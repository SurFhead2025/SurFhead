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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, get_rays

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, bg, image_width, image, image_height, image_path,
                 image_name, mask, mask_path, mask_name,  uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                 timestep=None, data_device = "cuda", normal_path=None, normal_name=None, normal=None, mask_face_path=None, mask_face_name=None, mask_face=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.bg = bg
        self.image = image
        self.image_width = image_width
        self.image_height = image_height
        self.image_path = image_path
        self.image_name = image_name
        self.mask = mask
        self.mask_path = mask_path
        self.mask_name = mask_name
        self.timestep = timestep

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)  #.cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)  #.cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.normal_path = normal_path
        self.normal_name = normal_name  
        self.normal = normal
        
        self.mask_face_path = mask_face_path
        self.mask_face_name = mask_face_name  
        self.mask_face = mask_face
        
        
    def get_calib_matrix_nerf(self):
        focal = fov2focal(self.FoVx, self.image_width)  # original focal length
        intrinsic_matrix = torch.tensor([[focal, 0, self.image_width / 2], [0, focal, self.image_height / 2], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1)#.contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self):
        intrinsic_matrix, extrinsic_matrix = self.get_calib_matrix_nerf()

        viewdirs = get_rays(self.image_width, self.image_height, intrinsic_matrix, extrinsic_matrix[:3,:3])
        return viewdirs
    
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, timestep):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.timestep = timestep

