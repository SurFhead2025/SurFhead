#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from tqdm import tqdm
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: Optional[np.array]
    image_path: str
    image_name: str
    mask: Optional[np.array] 
    mask_path: str
    mask_name: str    
    normal: Optional[np.array]
    normal_path: str
    normal_name: str
    mask_face: Optional[np.array] 
    mask_face_path: str
    mask_face_name: str  
    width: int
    height: int
    bg: np.array = np.array([0, 0, 0])
    timestep: Optional[int] = None
    camera_id: Optional[int] = None

class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    point_cloud: Optional[BasicPointCloud]
    ply_path: Optional[str]
    val_cameras: list = []
    train_meshes: dict = {}
    test_meshes: dict = {}
    tgt_train_meshes: dict = {}
    tgt_test_meshes: dict = {}

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # breakpoint()
        width, height = image.size

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    # breakpoint()
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if 'camera_angle_x' in contents:
            fovx_shared = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            file_path = frame["file_path"]
            if extension not in frame["file_path"]:
                file_path += extension
            
            cam_name = os.path.join(path, file_path)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            #! c2w = [X, Y, Z, T] XdotY=0, XdotZ=0, YdotZ=0
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            # breakpoint()
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            
            mask_path = image_path.replace(file_path, frame["fg_mask_path"])
            mask_name = Path(mask_path).stem
            
            
            mask_face_path = image_path.replace('images', 'binary_facer')
            # breakpoint()
            mask_face_name = Path(mask_face_path).stem
            
            if 'w' in frame and 'h' in frame:
                image = None
                mask = None
                mask_face = None
                width = frame['w']
                height = frame['h']
            else:
                image = Image.open(image_path)
                im_data = np.array(image.convert("RGBA"))
                norm_data = im_data / 255.0
                arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                width, height = image.size

            if 'camera_angle_x' in frame:
                fovx = frame["camera_angle_x"]
            else:
                fovx = fovx_shared
            fovy = focal2fov(fov2focal(fovx, width), height)

            timestep = frame["timestep_index"] if 'timestep_index' in frame else None
            camera_id = frame["camera_index"] if 'camera_id' in frame else None
            
            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, mask=mask, normal=None,
                mask_path=mask_path, mask_name=mask_name,   
                mask_face=mask_face, mask_face_path=mask_face_path, mask_face_name=mask_face_name,  
                image_path=image_path, image_name=image_name, 
                normal_path=None, normal_name=None,
                width=width, height=height, 
                timestep=timestep, camera_id=camera_id))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMeshesFromTransforms(path, transformsfile):
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            # breakpoint()
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos



def readMeshesFromCorpuses(path, transformsmode):
    
    if transformsmode == 'train':
        corpus_chunk = FaceTalk_train_corpuses
    elif transformsmode == 'val':
        corpus_chunk = FaceTalk_val_corpuses    
    elif transformsmode == 'test':
        corpus_chunk = FaceTalk_test_corpuses
    
    mesh_infos = {}
    
    outer_idx = 0
    #? 'translation', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'shape', 'expr', 'static_offset'
    # for corpus in corpus_chunk:
    for corpus_idx, corpus in tqdm(enumerate(corpus_chunk), total=len(corpus_chunk)):
    
        with open(os.path.join(path, corpus, 'transforms.json')) as json_file:
            contents = json.load(json_file)
            
            frames = contents["frames"]
            shape_shared = np.array(contents["shape"], dtype=np.float32).squeeze()
            # for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            for idx, frame in enumerate(frames):
                flame_param = dict()
                # if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                #     continue
                # breakpoint()
                flame_param['translation'] = np.zeros((1,3), dtype=np.float32)
                pose_correctives = np.array(frame['pose'], dtype=np.float32)
                flame_param['rotation'] = pose_correctives[:,:3]
                flame_param['neck_pose'] = pose_correctives[:,3:6]
                flame_param['jaw_pose'] = pose_correctives[:,6:9]
                flame_param['eyes_pose'] = pose_correctives[:,9:] #! 6
                flame_param['shape'] = shape_shared  #! GA (300,)
                flame_param['expr'] = np.array(frame['expression'], dtype=np.float32) #! GA (1,100)
                flame_param['static_offset'] = np.zeros((1,5143,3), dtype=np.float32)
                # breakpoint()
                # flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
                mesh_infos[outer_idx] = flame_param
                outer_idx += 1
            
    return mesh_infos

    with open(os.path.join(path, transformsmode)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        
        mesh_infos = {}
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            if not 'timestep_index' in frame or frame["timestep_index"] in mesh_infos:
                continue

            flame_param = dict(np.load(os.path.join(path, frame['flame_param_path']), allow_pickle=True))
            mesh_infos[frame["timestep_index"]] = flame_param
    return mesh_infos

def readCamerasMeshesFromCorpuses(path, transformsmode, white_background, extension=".png", GLOBAL_IDX=0):
    cam_infos = []
    mesh_infos = {}
    # breakpoint()
    height = width = 512
    camera_id = 0 #! monocular setting
    # outer_id = 0
    
    if transformsmode == 'train':
        corpus_chunk = FaceTalk_train_corpuses
    elif transformsmode == 'val':
        corpus_chunk = FaceTalk_val_corpuses    
    elif transformsmode == 'test':
        corpus_chunk = FaceTalk_test_corpuses
    if transformsmode == 'val':
        GLOBAL_IDX = 0 #! pause for each validation division bc same with train division
    
    GLOBAL_IDX_from = GLOBAL_IDX
    
    # for corpus in corpus_chunk:
    print(f"Reading {len(corpus_chunk)} corpuses from {transformsmode} division")   
    for corpus_idx, corpus in tqdm(enumerate(corpus_chunk), total=len(corpus_chunk)):
        with open(os.path.join(path, corpus, 'transforms.json')) as json_file:
            contents = json.load(json_file)
            if 'intrinsics' in contents:
                intrinsics_shared = contents["intrinsics"]
                _fx, _fy, _, _ = intrinsics_shared
                
                fx_real = -1 * _fx / 2.0 * width #!FaceTalk convention fx fy must be flipped - -> +
                fy_real = -1 * _fy / 2.0 * height
                fovx = focal2fov(fx_real, width) #! checked
                fovy = focal2fov(fy_real, height) #! checked
            # breakpoint()
            if 'cam' in contents:   
                w2c_shared = np.array(contents["cam"]).squeeze()
                
                w2c_shared[0, :] *= -1 #! x axis
                w2c_shared[1, :] *= -1 #! y axis
                # w2c_shared[:3, 1] *= -1 #! T
                # change from MakeHuman axes (X left Y up, Z forward) 
                # to COLMAP (X right Y down, Z forward)
                R = np.transpose(w2c_shared[:3,:3])
                T = w2c_shared[:3, 3]
            frames = contents["frames"]
            shape_shared = np.array(contents["shape"], dtype=np.float32).squeeze()
            # for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            for idx, frame in enumerate(frames):
                file_path = frame["file_path"]
                bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
                image_path = os.path.join(path, corpus, file_path)
                image_name = Path(f'{corpus}_{os.path.basename(file_path)}').stem   
                
                mask_path = image_path.replace('images', 'masks')
                mask_name = image_name
                
                normal_path = image_path.replace('images', 'normals')
                normal_name = image_name
                
                # breakpoint()
                normal = None; mask = None #! auxiliary information
                if 'w' in frame and 'h' in frame:
                    image = None
                    # mask = None
                    width = frame['w']
                    height = frame['h']
                else:
                    image = Image.open(image_path)
                    im_data = np.array(image.convert("RGBA"))
                    norm_data = im_data / 255.0
                    arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                    image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
                    width, height = image.size
                    
                    # mask = None
                    
                timestep = GLOBAL_IDX
                
                cam_infos.append(CameraInfo(
                        uid=GLOBAL_IDX, R=R, T=T, FovY=fovy, FovX=fovx, bg=bg, image=image, mask=mask, normal=normal,
                        mask_path=mask_path, mask_name=mask_name,   
                        image_path=image_path, image_name=image_name, 
                        normal_path=normal_path, normal_name=normal_name,
                        width=width, height=height, 
                        timestep=timestep, camera_id=camera_id))
                
                flame_param = dict()
                
                flame_param['translation'] = np.zeros((1,3), dtype=np.float32)
                pose_correctives = np.array(frame['pose'], dtype=np.float32)
                flame_param['rotation'] = pose_correctives[:,:3]
                # breakpoint()
                flame_param['neck_pose'] = pose_correctives[:,3:6] #!?????
                flame_param['jaw_pose'] = pose_correctives[:,6:9]
                flame_param['eyes_pose'] = pose_correctives[:,9:] #! 6
                flame_param['shape'] = shape_shared  #! GA (300,)
                flame_param['expr'] = np.array(frame['expression'], dtype=np.float32) #! GA (1,100)
                flame_param['static_offset'] = np.zeros((1,5143,3), dtype=np.float32)
                # breakpoint()
                mesh_infos[timestep] = flame_param 
                GLOBAL_IDX += 1
    
    number_of_frames = GLOBAL_IDX - GLOBAL_IDX_from
    
    print(f"Total {number_of_frames} frames read from {transformsmode} division")
    # breakpoint()
    if transformsmode == 'val':
        return cam_infos, mesh_infos#, GLOBAL_IDX
    else:
        return cam_infos, mesh_infos, GLOBAL_IDX
    
   



def readMakeHumanInfo(path, white_background, eval, extension=".png", target_path=""):
    print("Reading Training Corpuses and Meshes")
    
    GLOBAL_IDX  = 0
    if target_path != "":
        train_cam_infos, train_mesh_infos = readCamerasMeshesFromCorpuses(target_path, "train", white_background, extension)
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromCorpuses(target_path, "train")
    else:#!
        train_cam_infos, train_mesh_infos, GLOBAL_IDX = readCamerasMeshesFromCorpuses(path, "train", white_background, extension, GLOBAL_IDX)
        tgt_train_mesh_infos = {}
    
    # print("Reading Training Meshes")
    # train_mesh_infos = readMeshesFromCorpuses(path, "train")
    # if target_path != "":
    #     print("Reading Target Meshes (Training Division)")
    #     tgt_train_mesh_infos = readMeshesFromCorpuses(target_path, "train")
    # else:
    #     tgt_train_mesh_infos = {}
    
    print("Reading Validation Corpuses")
    if target_path != "":
        val_cam_infos, _ = readCamerasMeshesFromCorpuses(target_path, "val", white_background, extension) #!
    else:
        val_cam_infos, _ = readCamerasMeshesFromCorpuses(path, "val", white_background, extension, GLOBAL_IDX)
        #! Do not update GLOBAL_IDX
    
    print("Reading Test Corpuses and Meshes")
    if target_path != "":
        train_cam_infos, train_mesh_infos = readCamerasMeshesFromCorpuses(target_path, "test", white_background, extension)
        print("Reading Target Meshes (test Division)")
        tgt_test_mesh_infos = readMeshesFromCorpuses(target_path, "test")
    else:
        test_cam_infos, test_mesh_infos, GLOBAL_IDX = readCamerasMeshesFromCorpuses(path, "test", white_background, extension, GLOBAL_IDX)
        tgt_test_mesh_infos = {}
    # breakpoint()
    # print("Reading Test CorpureadMeshesFromCorpuses")
    # if target_path != "":
    #     test_cam_infos = readCamerasFromCorpuses(target_path, "test", white_background, extension)
    # else:
    #     test_cam_infos = readCamerasFromCorpuses(path, "test", white_background, extension)
    
    # print("Reading Test Meshes")
    # test_mesh_infos = readMeshesFromCorpuses(path, "test")
    # if target_path != "":
    #     print("Reading Target Meshes (Test Division)")
    #     tgt_test_mesh_infos = readMeshesFromCorpuses(target_path, "test")
    # else:
    #     tgt_test_mesh_infos = {}
    
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info

def readDynamicNerfInfo(path, white_background, eval, extension=".png", target_path=""): #!
    print("Reading Training Transforms")
    if target_path != "":
        train_cam_infos = readCamerasFromTransforms(target_path, "transforms_train.json", white_background, extension)
    else:
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    
    print("Reading Training Meshes")
    train_mesh_infos = readMeshesFromTransforms(path, "transforms_train.json")
    if target_path != "":
        print("Reading Target Meshes (Training Division)")
        tgt_train_mesh_infos = readMeshesFromTransforms(target_path, "transforms_train.json")
    else:
        tgt_train_mesh_infos = {}
    
    print("Reading Validation Transforms")
    if target_path != "":
        val_cam_infos = readCamerasFromTransforms(target_path, "transforms_val.json", white_background, extension)
    else:
        val_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension)
    
    print("Reading Test Transforms")
    if target_path != "":
        test_cam_infos = readCamerasFromTransforms(target_path, "transforms_test.json", white_background, extension)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    print("Reading Test Meshes")
    test_mesh_infos = readMeshesFromTransforms(path, "transforms_test.json")
    if target_path != "":
        print("Reading Target Meshes (Test Division)")
        tgt_test_mesh_infos = readMeshesFromTransforms(target_path, "transforms_test.json")
    else:
        tgt_test_mesh_infos = {}
    
    if target_path != "" or not eval:
        train_cam_infos.extend(val_cam_infos)
        val_cam_infos = []
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
        train_mesh_infos.update(test_mesh_infos)
        test_mesh_infos = {}

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=None,
                           train_cameras=train_cam_infos,
                           val_cameras=val_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=None,
                           train_meshes=train_mesh_infos,
                           test_meshes=test_mesh_infos,
                           tgt_train_meshes=tgt_train_mesh_infos,
                           tgt_test_meshes=tgt_test_mesh_infos)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "DynamicNerf" : readDynamicNerfInfo,
    "Blender" : readNerfSyntheticInfo,
    "MakeHuman" : readMakeHumanInfo,
}

# FaceTalk_train_corpuses = ['bareteeth','cheeks_in', 'eyebrow', 'high_smile', \
#     'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme', 'mouth_middle', 'mouth_open', \
#         'mouth_side', 'mouth_up', 'sentence01', 'sentence02', 'sentence03', 'sentence04', ]
#             # 'sentence05', 'sentence06', 'sentence07', 'sentence08', 'sentence09', 'sentence10', \
#             #     'sentence11', 'sentence12', 'sentence13', 'sentence14', 'sentence15', 'sentence16', \
#             #         'sentence17', 'sentence18', 'sentence19', 'sentence20', 'sentence21', 'sentence22', \
#             #             'sentence23', 'sentence24', 'sentence25', 'sentence26', 'sentence27', 'sentence28', \
#             #                 'sentence29', 'sentence30', 'sentence31', 'sentence32', 'sentence33', 'sentence34', \
#             #                     'sentence35', 'sentence36', 'sentence37' ]
# FaceTalk_val_corpuses = ['bareteeth'] #! 무조건 첫번째 트레인 첫번쨰 코퍼스부터 순차적으로 사용해야함.
# FaceTalk_test_corpuses = ['sentence38','sentence39']#'sentence40']


FaceTalk_train_corpuses = ['sentence01', 'sentence02', 'sentence03', 'sentence04', \
            'sentence05', 'sentence06', 'sentence07', 'sentence08', 'sentence09', 'sentence10', \
                'sentence11', 'sentence12', 'sentence13', 'sentence14', 'sentence15', 'sentence16', \
                    'sentence17', 'sentence18', 'sentence19', 'sentence20', 'sentence21', 'sentence22', \
                        'sentence23', 'sentence24', 'sentence25', 'sentence26', 'sentence27', 'sentence28', \
                            'sentence29', 'sentence30', 'sentence31', 'sentence32','sentence33', 'sentence34', \
                                'sentence35', 'sentence36', 'sentence37','sentence38','sentence39','sentence40']

FaceTalk_val_corpuses = ['sentence01', 'sentence02']

FaceTalk_test_corpuses = ['bareteeth','cheeks_in', 'eyebrow', 'high_smile', \
                            'lips_back', 'lips_up', 'mouth_down', 'mouth_extreme', 'mouth_middle', 'mouth_open', \
                                'mouth_side', 'mouth_up']