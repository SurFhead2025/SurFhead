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

import torch
from torch.utils.data import DataLoader
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import concurrent.futures
import multiprocessing
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
# from utils.general_utils import colormap

from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, FlameGaussianModel
from mesh_renderer import NVDiffRenderer
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.image_utils import apply_depth_colormap, frames2video
from scene import SpecularModel
#! import F
import torch.nn.functional as F

import open3d as o3d
from torchvision.utils import save_image as si
try:
    # breakpoint()
    mesh_renderer = NVDiffRenderer()
except:
    print("Cannot import NVDiffRenderer. Mesh rendering will be disabled.")
    mesh_renderer = None

from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap

def colormap(img, cmap='jet'):
    # Normalize the image data to 0-1 range
    norm = Normalize(vmin=img.min(), vmax=img.max())
    img_normalized = norm(img)
    
    # Get the colormap
    cmap = get_cmap(cmap)
    
    # Apply the colormap
    img_colormap = cmap(img_normalized)
    
    # Remove the alpha channel
    img_colormap = img_colormap[:, :, :3]
    
    # Convert to torch tensor and permute dimensions to match (C, H, W) format
    img_colormap_torch = torch.from_numpy(img_colormap).float().permute(2, 0, 1)
    # breakpoint()
    return img_colormap_torch

def write_data(path2data):
    for path, data in path2data.items():
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in [".png", ".jpg"]:
            data = data.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            Image.fromarray(data).save(path)
        elif path.suffix in [".obj"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".txt"]:
            with open(path, "w") as f:
                f.write(data)
        elif path.suffix in [".npz"]:
            np.savez(path, **data)
        elif path.suffix in [".npy"]:
            np.save(path, data)
        else:
            raise NotImplementedError(f"Unknown file type: {path.suffix}")
        
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math, random
class DummyCamera:
    def __init__(self, projection_matrix, world_view_transform, W, H, FoVx, FoVy):
        # self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
        # self.R = R
        # self.T = T
        self.projection_matrix = projection_matrix
        self.world_view_transform = world_view_transform
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0,0,0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = W
        self.image_height = H
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.original_mask = None

def LookAtPoseSampler(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
    h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
    v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
    v = torch.clamp(v, 1e-5, math.pi - 1e-5)

    theta = h
    v = v / math.pi
    phi = torch.arccos(1 - 2*v)

    camera_origins = torch.zeros((batch_size, 3), device=device)

    camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
    camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
    camera_origins[:, 1:2] = radius*torch.cos(phi)

    # forward_vectors = math_utils.normalize_vecs(-camera_origins)
    def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
        '''
        # Normalize vector lengths.
        '''
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True))
    
    forward_vectors = normalize_vecs(lookat_position - camera_origins)
    
    
    forward_vector = normalize_vecs(forward_vectors)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=camera_origins.device).expand_as(forward_vector)

    right_vector = -normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=camera_origins.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=camera_origins.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = camera_origins
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world

    
def render_set(dataset, name, iteration, views, gaussians, pipeline, background, render_mesh, extract_mesh, random_camera=0, specular= None):
    if dataset.select_camera_id != -1:
        name = f"{name}_camera_{dataset.select_camera_id}"
    iter_path = Path(dataset.model_path) / name / f"ours_{iteration}"
    
    gts_path = iter_path / "gt"
    mask_path = iter_path / "mask"
    
    render_path = iter_path / "renders"
    render_alpha_path = iter_path / "render_alphas"
    render_depth_path = iter_path / "render_depths"
    render_analytic_normal_path = iter_path / "render_analytic_normals"
    render_tangent_normal_path = iter_path / "render_tangent_normals"
    
    if render_mesh:
        render_mesh_path = iter_path / "renders_mesh"
    if extract_mesh:
        mesh_path = iter_path / "meshes"
        revolver = 80
        gaussians.select_mesh_by_timestep(views[revolver*16].timestep)
        gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=background)   
         
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(render_alpha_path, exist_ok=True)
    os.makedirs(render_depth_path, exist_ok=True)
    os.makedirs(render_analytic_normal_path, exist_ok=True)
    os.makedirs(render_tangent_normal_path, exist_ok=True)
    if extract_mesh:
        os.makedirs(mesh_path, exist_ok=True)
    
    if render_mesh:
        os.makedirs(render_mesh_path, exist_ok=True)
    
    views_loader = DataLoader(views, batch_size=None, shuffle=False, num_workers=8)
    max_threads = multiprocessing.cpu_count()
    print('Max threads: ', max_threads)
    worker_args = []
    ONLY_IMAGE = False
    
    # breakpoint()
    if extract_mesh:
        if random_camera != 0:
            views_random = []
            # ghost_cams = []
            FoVx = views[0].FoVx
            FoVy = views[0].FoVy
            image_width = views[0].image_width
            image_height = views[0].image_height
            projection_matrix = views[0].projection_matrix
            camera_positions = torch.stack([torch.linalg.inv(i.world_view_transform.transpose(0,1))[:3,3] for i in views[:16]], dim=0)#.mean(0)
            # camera_mean_position = camera_positions.mean(0)
            camera_mean_norm = camera_positions.norm(dim=-1).mean()
            pitch_range = 0.8
            yaw_range = 0.8
            lookat_position = torch.tensor([0, 0, 0], dtype=torch.float32, device='cpu')
            for i in range(random_camera):
                pitch_glitch = random.uniform(-pitch_range, pitch_range)
                yaw_glitch = random.uniform(-yaw_range, yaw_range)
                
                # ghost_cam = LookAtPoseSampler(3.14/2+ yaw_glitch, 3.14/2 + pitch_glitch, lookat_position, \
                #                 horizontal_stddev=0, vertical_stddev=0, radius = camera_mean_norm, batch_size=1, device='cpu')[0]
                ghost_c2w = LookAtPoseSampler(3.14/2+ yaw_glitch,  3.14/2+pitch_glitch, lookat_position, \
                                horizontal_stddev=0, vertical_stddev=0, radius = camera_mean_norm, batch_size=1, device='cpu')[0]
                # breakpoint()
                ghost_w2c = torch.linalg.inv(ghost_c2w).transpose(0,1) #! glm
                views_random.append(DummyCamera(projection_matrix, ghost_w2c, image_width, image_height, FoVx, FoVy))
            # gaussExtractor.reconstruction(views[:16]+views_random)
            gaussExtractor.reconstruction(views[revolver*16:revolver*16+16]+views_random)
        else:
            gaussExtractor.reconstruction(views[:16])
        name = 'fuse_for_toggle.ply'
        sdf_trunc = -1.0
        mesh_res = 4096
        voxel_size = -1.0
        num_cluster = 1 #50
        depth_trunc = -1.0
        
        depth_trunc = (gaussExtractor.radius * 2.0) if depth_trunc < 0  else depth_trunc
        # depth_trunc = 3
        voxel_size = (depth_trunc / mesh_res) if voxel_size < 0 else voxel_size
        sdf_trunc = 5.0 * voxel_size if sdf_trunc < 0 else sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        o3d.io.write_triangle_mesh(os.path.join(mesh_path, name), mesh)
        print("mesh saved at {}".format(os.path.join(mesh_path, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(mesh_path, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(mesh_path, name.replace('.ply', '_post.ply'))))
        exit()
        if False:
        # for idx, view in enumerate(views[:16]+views_random):
            aa = mesh_renderer.render_from_camera(torch.tensor(np.array(mesh_post.vertices),dtype=torch.float32)[None].cuda(),torch.tensor(np.array(mesh_post.triangles),dtype=torch.float32).cuda(),view)['rgba']
            # if gaussians.binding is not None:
            #     gaussians.select_mesh_by_timestep(view.timestep)
            
            K = 600_000  # Temporarily fixed
            han_window_iter = iteration * 2 / (K + 1)
            if pipeline.SGs:
                if pipeline.train_kinematic:
                    dir_pp = (gaussians.get_blended_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    normal = gaussians.get_normal
                    normal_normalised = F.normalize(normal,dim=-1).detach()
                    # breakpoint()
                    
                    normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                    if pipeline.rotSH:
                        dir_pp_normalized = (gaussians.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                else:
                    dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                    dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                    normal = gaussians.get_normal
                    normal_normalised = F.normalize(normal,dim=-1).detach()
                    # breakpoint()
                    
                    normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                    if pipeline.rotSH:
                        # breakpoint()
                        dir_pp_normalized = (gaussians.face_R_mat.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
            
            
                if pipeline.spec_only_eyeball:
                    # breakpoint()
                    mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.eyeballs)
                    points_indices = torch.nonzero(mask).squeeze(1)
                    
                    specular_color_eyeballs = specular.step(gaussians.get_sg_features[points_indices], dir_pp_normalized[points_indices], normal[points_indices].detach(), sg_type = pipeline.sg_type)
                    
                
                    
                    specular_color = specular_color_eyeballs
                
                else:
                    specular_color = specular.step(gaussians.get_sg_features, dir_pp_normalized, normal.detach(), sg_type = pipeline.sg_type)
                
            else:
                specular_color = None
            render_bucket = render(view, gaussians, pipeline, background, 
                                backface_culling_smooth=dataset.backface_culling_smooth,
                                backface_culling_hard=dataset.backface_culling_hard,
                                iter=han_window_iter,
                                specular_color= specular_color,
                                    spec_only_eyeball = pipeline.spec_only_eyeball)
            
            si(render_bucket["render"], os.path.join(mesh_path, name.replace('.ply', f'{idx}_render_image.png')))
            si(aa[0].permute(2,0,1), os.path.join(mesh_path, name.replace('.ply', f'{idx}_render.png')))
            # breakpoint()
        # exit()
    for idx, view in enumerate(tqdm(views_loader, desc="Rendering progress")):
        if gaussians.binding is not None:
            gaussians.select_mesh_by_timestep(view.timestep)
        
        K = 600_000  # Temporarily fixed
        han_window_iter = iteration * 2 / (K + 1)
        if pipeline.SGs:
            if pipeline.train_kinematic:
                dir_pp = (gaussians.get_blended_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = gaussians.get_normal
                normal_normalised = F.normalize(normal,dim=-1).detach()
                # breakpoint()
                
                normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                if pipeline.rotSH:
                    dir_pp_normalized = (gaussians.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
            else:
                dir_pp = (gaussians.get_xyz - view.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = gaussians.get_normal
                normal_normalised = F.normalize(normal,dim=-1).detach()
                # breakpoint()
                
                normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                if pipeline.rotSH:
                    dir_pp_normalized = (gaussians.face_R_mat.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
           
           
            if pipeline.spec_only_eyeball:
                # breakpoint()
                mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.eyeballs)
                points_indices = torch.nonzero(mask).squeeze(1)
                
                specular_color_eyeballs = specular.step(gaussians.get_sg_features[points_indices], dir_pp_normalized[points_indices], normal[points_indices].detach(), sg_type = pipeline.sg_type)
                
              
                
                specular_color = specular_color_eyeballs
               
            else:
                specular_color = specular.step(gaussians.get_sg_features, dir_pp_normalized, normal.detach(), sg_type = pipeline.sg_type)
               
        else:
            specular_color = None
        render_bucket = render(view, gaussians, pipeline, background, 
                               backface_culling_smooth=dataset.backface_culling_smooth,
                               backface_culling_hard=dataset.backface_culling_hard,
                               iter=han_window_iter,
                               specular_color= specular_color,
                                spec_only_eyeball = pipeline.spec_only_eyeball)
        # breakpoint()
        
            
        rendering = render_bucket["render"]
        
        gt = view.original_image[0:3, :, :]
        if not ONLY_IMAGE:
            gt_mask = view.original_mask[0:1, :, :]
            try:
                gt_normal = view.original_normal[0:3, :, :]
                normal_path = iter_path / 'normal'
                os.makedirs(normal_path, exist_ok=True)
            except:
                gt_normal = None
            
            if render_mesh:
                out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, view)
                rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                rgb_mesh = rgba_mesh[:3, :, :]
                alpha_mesh = rgba_mesh[3:, :, :]
                mesh_opacity = 0.5
                rendering_mesh = rgb_mesh * alpha_mesh * mesh_opacity + gt.to(rgb_mesh) * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
        if not ONLY_IMAGE:
            
            path2data = {
                Path(render_path) / f'{idx:05d}.png': rendering,
                Path(gts_path) / f'{idx:05d}.png': gt,
                Path(mask_path) / f'{idx:05d}.png': gt_mask.repeat(3,1,1),
                Path(render_alpha_path) / f'{idx:05d}.png': render_bucket['surfel_rend_alpha'].repeat(3,1,1),
                Path(render_analytic_normal_path) / f'{idx:05d}.png': render_bucket["surfel_surf_normal"] * 0.5 + 0.5,
                Path(render_analytic_normal_path) / f'{idx:05d}.npy': render_bucket["surfel_surf_normal"].detach().cpu().numpy(),
                Path(render_tangent_normal_path) / f'{idx:05d}.png': (render_bucket["surfel_rend_normal"] * 0.5 + 0.5) * gt_mask.repeat(3,1,1).cuda() + (1 - gt_mask.repeat(3,1,1).cuda()),
                Path(render_tangent_normal_path) / f'{idx:05d}.npy': render_bucket["surfel_rend_normal"].detach().cpu().numpy(),
            }
        else:
            path2data = {
                Path(render_path) / f'{idx:05d}.png': rendering,
                Path(gts_path) / f'{idx:05d}.png': gt,
                Path(render_tangent_normal_path) / f'{idx:05d}.png': render_bucket["surfel_rend_normal"] * 0.5 + 0.5,
            }
        if not ONLY_IMAGE:
            if gt_normal is not None:
                path2data[Path(normal_path) / f'{idx:05d}.png'] = gt_normal
                path2data[Path(normal_path) / f'{idx:05d}.npy'] = (gt_normal * 2 - 1).detach().cpu().numpy()
            
            depth = render_bucket["surfel_surf_depth"]
            # depth = depth * gt_mask.cuda()
            # norm = depth.max()
            # depth = depth / norm
            # depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
            
            alpha = render_bucket['surfel_rend_alpha']
            valid_depth_area = depth[gt_mask > 0.1]
            max_depth_value = valid_depth_area.max()
            min_depth_value = valid_depth_area.min()
            
            norm = max_depth_value - min_depth_value
            depth[alpha < 0.1] = max_depth_value #! fill bg with max depth
            depth = (depth - min_depth_value) / norm
            # from torchvision.utils import save_image as si
            # breakpoint()
            # depth = depth / norm
            depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
            depth = depth * gt_mask.repeat(3,1,1).cpu() + (1 - gt_mask.repeat(3,1,1).cpu())
            path2data[Path(render_depth_path) / f'{idx:05d}.png'] = depth
            
            if render_mesh:
                path2data[Path(render_mesh_path) / f'{idx:05d}.png'] = rendering_mesh

        worker_args.append(path2data)
        # breakpoint()
        if len(worker_args) == max_threads or idx == len(views_loader) - 1:
            with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
                futures = [executor.submit(write_data, args) for args in worker_args]
                for future in futures:
                    future.result()
            worker_args = []
    

    # frames2video(render_path, f"{iter_path}/renders.mp4")
    # frames2video(gts_path, f"{iter_path}/gt.mp4")
    # frames2video(render_tangent_normal_path, f"{iter_path}/renders_normal.mp4")

        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_val : bool, skip_test : bool, \
    render_mesh: bool, extract_mesh: bool, random_camera : int = 0):
    with torch.no_grad():
        if dataset.bind_to_mesh:
            if os.path.basename(dataset.source_path).startswith("FaceTalk"):
                # breakpoint()
                n_shape = 100    
                n_expr = 50
            else:
                n_shape = 300
                n_expr = 100   
            # gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset)
            gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,\
                train_normal = False, n_shape=n_shape, n_expr=n_expr, train_kinematic=pipeline.train_kinematic, \
                DTF = pipeline.DTF, invT_Jacobian=pipeline.invT_Jacobian,
                detach_eyeball_geometry = pipeline.detach_eyeball_geometry)
        else:
            gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        specular = None
        if pipeline.SGs:
            specular = SpecularModel()
            specular.load_weights(dataset.model_path)
        if dataset.target_path != "":
             name = os.path.basename(os.path.normpath(dataset.target_path))
             # when loading from a target path, test cameras are merged into the train cameras
             render_set(dataset, f'{name}', scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)
        else:
            if not skip_train:
                # if random_camera !=0:
                render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)
                # else:
                    # render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_mesh, extract_mesh)
            
            if not skip_val:
                render_set(dataset, "val", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

            if not skip_test:
                render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_mesh, extract_mesh, random_camera, specular)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_mesh", action="store_true")
    parser.add_argument("--extract_mesh", action="store_true")
    parser.add_argument("--random_camera", default=20, type=int)
    # parser.add_argument("--render_normal", action="store_true")
    # parser.add_argument("--render_depth", action="store_true")
    # parser.add_argument("--render_neigh_normal", action="store_true")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_val, args.skip_test,\
        args.render_mesh, args.extract_mesh, args.random_camera)