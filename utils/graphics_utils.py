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
import torch.nn.functional as F
import torch
import math
import numpy as np
from typing import NamedTuple


def get_dist_weight(centroids, xyz):
    # centroids = centroid #? N, 4, 3
    # breakpoint()
    p = xyz.unsqueeze(dim=1).repeat(1,4,1).detach() #? N, 4, 3
    beta = 4
    distances = torch.norm(centroids - p, dim=-1) #? N, 4
    weights = torch.exp(-beta * distances)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights #? N, 4

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0,1), mode='constant', value=w)

def compute_face_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return face_normals

def compute_face_orientation(verts, faces, return_scale=True, scale_dim = 1, return_type = 'GA', E_inverse=None):
    assert return_scale
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()
    # breakpoint()
    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

      # will have artifacts without negation
    if return_type == 'GA':
        a0 = safe_normalize(v1 - v0)
        # a1 = safe_normalize(v2 - v0)
        # a2 = safe_normalize(torch.cross(a0, a1, dim=-1))
        a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
        a2 = safe_normalize(torch.cross(a1, a0, dim=-1)) #! no negation by right hand drill-law
        orientation = torch.cat([a0[..., None], a2[..., None], a1[..., None]], dim=-1)
        #! 0 2 1 == base height perpendi
        
        #* original
        # a0 = safe_normalize(v1 - v0)
        # a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
        # a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))  # will have artifacts without negation

        # orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)
        #* original
        if return_scale:
            # breakpoint()
            s0 = length(v1 - v0) #! a0 axis
            s1 = dot(a2, (v2 - v0)).abs() #! a2 axis
            if scale_dim == 1:
                # scale = (s0 * s1) / 2
                scale = (s0 + s1) / 2
            elif scale_dim == 2:
                s_dot5 = torch.one_like(s0).to(s0)
                scale = torch.cat([s0,s_dot5,s1],-1)
                # scale = torch.cat([s0,s1],-1)

        return orientation, scale
    elif return_type == 'DTF_GA':
        # breakpoint()
        a0 = safe_normalize(v1 - v0)
        # a1 = safe_normalize(v2 - v0)
        # a2 = safe_normalize(torch.cross(a0, a1, dim=-1))
        a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
        a2 = safe_normalize(torch.cross(a1, a0, dim=-1)) #! no negation by right hand drill-law
        E_prime = torch.cat([a0[..., None], a2[..., None], a1[..., None]], dim=-1)
        #! orientation @ E = E_prime
        orientation = torch.bmm(E_prime, E_inverse)
        return orientation
    elif return_type == 'DTF':
        # breakpoint()
        assert E_inverse is not None
        if E_inverse is not None:
            pass
        else:
             #! last column is same with surfel's perpendicular
            # breakpoint()
            e0 = torch.tensor([[1,0,0]], device='cuda',dtype=torch.float32) * 0.0057
            e1 = F.normalize(torch.tensor([[0,1,0]], device='cuda',dtype=torch.float32),dim=-1) * 0.0057
            #! 60 degree
            e2 = torch.tensor([[0,0,1]], device='cuda',dtype=torch.float32) 

            E_inverse = torch.inverse(torch.cat([e0[..., None], e1[..., None], e2[..., None]], dim=-1))
            E_inverse = E_inverse.repeat(r0.shape[0],1,1)
        r0 = v1 - v0 #safe_normalize(v1 - v0) #! has length
        r1 =  v2 - v0 # safe_normalize(v2 - v0)
        r2 = safe_normalize(torch.cross(r0, v2 - v0, dim=-1))
        E_prime = torch.cat([r0[..., None], r1[..., None], r2[..., None]], dim=-1)
        #! orientation @ E = E_prime
        orientation = torch.bmm(E_prime, E_inverse)
        # orientation = torch.cat([r0[..., None], r1[..., None], r2[..., None]], dim=-1)

        
        
        return orientation#, orientation #! affine, nothing dummy

def compute_E_inverse_GA(verts, faces):
    # assert return_scale
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    
    # r0 = v1 - v0
    # r1 = v2 - v0 
    # r2 = safe_normalize(torch.cross(r0, v2 - v0, dim=-1))
    
    a0 = safe_normalize(v1 - v0)
        # a1 = safe_normalize(v2 - v0)
    # a2 = safe_normalize(torch.cross(a0, a1, dim=-1))
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = safe_normalize(torch.cross(a1, a0, dim=-1)) #! no negation by right hand drill-law
    E = torch.cat([a0[..., None], a2[..., None], a1[..., None]], dim=-1)
    E_inverse = torch.linalg.inv(E)
    return E_inverse


def compute_E_inverse(verts, faces):
    # assert return_scale
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    
    r0 = v1 - v0
    r1 = v2 - v0 
    r2 = safe_normalize(torch.cross(r0, v2 - v0, dim=-1))
    E = torch.cat([r0[..., None], r1[..., None], r2[..., None]], dim=-1)
    E_inverse = torch.linalg.inv(E)
    return E_inverse

def compute_vertex_normals(verts, faces):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    v_normals = torch.zeros_like(verts)
    N = verts.shape[0]
    v_normals.scatter_add_(1, i0[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i1[..., None].repeat(N, 1, 3), face_normals)
    v_normals.scatter_add_(1, i2[..., None].repeat(N, 1, 3), face_normals)

    v_normals = torch.where(dot(v_normals, v_normals) > 1e-20, v_normals, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda'))
    v_normals = safe_normalize(v_normals)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_normals))
    return v_normals

#! from here


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth_pcd2normal(xyz):
    hd, wd, _ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    # xyz_normal = torch.cross(bottom_to_top, left_to_right, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    # breakpoint()
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def get_rays(width, height, intrinsic, camrot):
    px, py = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32))

    pixelcoords = torch.stack((px, py), dim=-1).cuda()  # H x W x 2
    raydir = get_dtu_raydir(pixelcoords, intrinsic, camrot, dir_norm=True)
    return raydir