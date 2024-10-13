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

from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, get_minimum_axis, flip_align_view
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from pytorch3d.transforms import quaternion_multiply
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
# from utils.graphics_utils import BasicPointCloud
from utils.graphics_utils import BasicPointCloud, get_dist_weight
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F
from torch.autograd import Function
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, unitquat_to_rotmat, special_procrustes
from pytorch3d.ops import knn_points

    
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        self.blend_weight_activation = torch.sigmoid

    def __init__(self, sh_degree : int, sg_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.tight_visibility_mask = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        # for binding GaussianModel to a mesh
        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        self.binding = None  # gaussian index to face index
        self.binding_counter = None  # number of points bound to each face
        self.timestep = None  # the current timestep
        self.num_timesteps = 1  # required by viewers

        self.face_trans_mat = None
        
        #! for normal correction
        self.train_normal = False
        self._normal = torch.zeros(0)
        self._normal2 = torch.zeros(0)
        
        self.train_kinematic = False
        self.train_kinematic_dist = False
        #! have to check kinematic and kinematic dist not both

        self.blended_Jacobian = None
        # self.scaling_aniso = False
        self.DTF = False
        self.invT_Jacobian = False
        self.densification_type = 'arithmetic_mean'
        #! for spec gs
        self._features_sg = torch.empty(0)
        self.max_sg_degree = sg_degree
        self.detach_eyeball_geometry = False
        self.detach_boundary = False

        self.blended_Jacobian = None
        self.blended_R = None
        self.blended_U = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.tight_visibility_mask,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.binding,
        self.binding_counter,
        self.max_radii2D, 
        xyz_gradient_accum, 
        tight_visibility_mask,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.tight_visibility_mask = tight_visibility_mask
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
       

    
    @property
    def get_sg_features(self):
        return self._features_sg

    @property
    def get_scaling(self):
        if self.binding is None:
            return self.scaling_activation(self._scaling)
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_scaling is None:
                self.select_mesh_by_timestep(0)

            scaling = self.scaling_activation(self._scaling)
          
            return scaling * self.face_scaling[self.binding]
    
    @property
    def get_rotation(self):
        if self.binding is None:
            return self.rotation_activation(self._rotation)
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_orien_quat is None:
                self.select_mesh_by_timestep(0)

            # always need to normalize the rotation quaternions before chaining them
            # if self.detach_eyeball_geometry:

            rot = self.rotation_activation(self._rotation)
                
            face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
            return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))

        
    
    @property
    def get_normal(self):
        if self.binding is None:
            return self.rotation_activation(self._rotation)
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_orien_quat is None:
                self.select_mesh_by_timestep(0)

            # breakpoint()
            rot = self.rotation_activation(self._rotation)[self.binding]
            face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
            full_rot = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot))) #? same as get rotation
            normal = build_rotation(full_rot)[..., 2:]
            if self.DTF: #! Rr
             
               
                # rot = self.rotation_activation(self._rotation)[self.binding]
                # face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
                # full_rot = quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot))) #? same as get rotation
                # normal = build_rotation(full_rot)[..., 2:]
                if self.invT_Jacobian:
                    if self.train_kinematic or self.train_kinematic_dist:
                    # if False:
                        blended_Jacobian = self.blended_Jacobian
                        transmat_inv = torch.linalg.inv(blended_Jacobian).permute(0,2,1)
                    else:
                        transmat_inv = torch.linalg.inv(self.face_trans_mat[self.binding]).permute(0,2,1)
                    normal_deformed = torch.bmm(transmat_inv, normal).squeeze(-1)
                
                else:
                    raise NotImplementedError
                    if self.train_kinematic:
                        normal_deformed = torch.bmm(self.blended_Jacobian, normal).squeeze(-1)
                    else:
                        normal_deformed = torch.bmm(self.face_trans_mat[self.binding], normal).squeeze(-1)
                    
                return normal_deformed
               
            else:
                return normal.squeeze(-1)
    
    
        
    def get_blended_jacobian(self):
        
        def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
            #! this convention 
            #! https://blog.naver.com/PostView.naver?blogId=richscskia&logNo=222179474476
            #! https://discuss.pytorch.org/t/polar-decomposition-of-matrices-in-pytorch/188458/2
            # breakpoint()
            U, S, Vh = torch.linalg.svd(m)
            U_new = torch.bmm(U, Vh) #! Unitary
            P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S).to(dtype = m.dtype)), Vh) #! PSD
            # P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S)), Vh)
            return U_new, P
        
        binding_faces = self.face_adjacency.cuda()[self.binding]

        if self.train_kinematic:

            blend_weights = F.normalize(self.blend_weight_activation(self.blend_weight), dim=-1, p=1)
           
        elif self.train_kinematic_dist:
            self.centroids = self.face_center[self.face_adjacency.cuda()[self.binding]]
            blend_weights = get_dist_weight(self.centroids, self.get_xyz_cano)
        
        if self.detach_eyeball_geometry:
            eyeball_mask = torch.isin(self.binding, self.flame_model.mask.f.eyeballs)

            # print(eyeball_mask.sum(), 'EYEBALLS::Check the number of eyeballs')
            eyeball_indices = torch.nonzero(eyeball_mask).squeeze(1)
            # breakpoint()
            blend_weights[eyeball_indices] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=blend_weights.device)
            
            # blend_weights[]
        #! "right" Cauchy-Green
        Q = self.face_trans_mat
        # R, U = polar_decomp(Q)
    
        R_ = self.face_R_mat[binding_faces].detach(); U_ = self.face_U_mat[binding_faces].detach()
        R_rotvec = rotmat_to_rotvec(R_).detach()
        # R_rotvec = self.R_rotvec[binding_faces]
        # U_mat = self.U_mat[binding_faces]
        # U_mat = U[b]
        blended_R_rotvec = torch.sum(blend_weights.unsqueeze(-1) * R_rotvec, dim=1)
        blended_R = rotvec_to_rotmat(blended_R_rotvec)
        
        blended_U = torch.sum(blend_weights.unsqueeze(-1).unsqueeze(-1)*U_, dim=1)
        
        
        # R_rotvec = self.R_rotvec[binding_faces]
        # U_mat = self.U_mat[binding_faces]
        # blended_R_rotvec = torch.sum(blend_weights.unsqueeze(-1) * R_rotvec, dim=1)
        
        
        # blended_U = torch.sum(blend_weights.unsqueeze(-1).unsqueeze(-1)*U_mat, dim=1)
        # blended_R = rotvec_to_rotmat(blended_R_rotvec)
        blended_Jacobians = torch.bmm(blended_R, blended_U)
        if self.detach_eyeball_geometry:
            # breakpoint()
            blended_Jacobians[eyeball_indices] = Q[self.binding][eyeball_indices]
        return blended_Jacobians, blended_R, blended_U
    
    
        

    @property
    def get_blended_xyz(self):
        if self.binding is None:
            return self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
     

            
            if self.DTF:
            # if True:
                # breakpoint()
                # if self.blended_Jacobian is None:
                if self.blended_Jacobian is None or self.blended_R is None or self.blended_U is None:
                    blended_Jacobian, blended_R, blended_U = self.get_blended_jacobian()

                    self.blended_Jacobian = blended_Jacobian
                    self.blended_R = blended_R
                    self.blended_U = blended_U

                global_scaling = self.face_scaling[self.binding]
                xyz_cano = global_scaling[...,None]*torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None])#.squeeze(-1)
                # xyz_posed = torch.bmm(self.blended_Jacobian, xyz_cano).squeeze(-1) + self.face_center[self.binding]
                xyz_posed = torch.bmm(self.blended_Jacobian, xyz_cano).squeeze(-1) + self.face_center[self.binding]
            
                return xyz_posed
            else:
                raise NotImplementedError
    @property
    def get_xyz_cano(self):
        if self.binding is None:
            return self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
     

            
            if self.DTF:
            
                global_scaling = self.face_scaling[self.binding]
                # breakpoint()
                xyz_cano = global_scaling*\
                    torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1) + \
                        self.face_center[self.binding]
                # breakpoint()
                # xyz_posed = torch.bmm(self.blended_Jacobian, xyz_cano).squeeze(-1) + self.face_center[self.binding]
                # xyz_posed = torch.bmm(blended_Jacobian, xyz_cano).squeeze(-1) + self.face_center[self.binding]
            
                return xyz_cano#.squeeze(-1)
            else:
                raise NotImplementedError
            
    @property
    def get_xyz(self):
        if self.binding is None:
            return self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
    
            
            
            if self.DTF:#! JRr + T_posed
           
                global_scaling = self.face_scaling[self.binding]
                xyz_cano = global_scaling[...,None]*torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None])#.squeeze(-1)
                xyz_posed = torch.bmm(self.face_trans_mat[self.binding], xyz_cano).squeeze(-1) + self.face_center[self.binding]
                return xyz_posed 

            else:
              
                    
                xyz = torch.bmm(self.face_orien_mat[self.binding], self._xyz[..., None]).squeeze(-1)
                global_scaling = self.face_scaling[self.binding]
                return xyz * global_scaling + self.face_center[self.binding]
            
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_blend_weight(self):
        if self.train_kinematic:
            return self.blend_weight_activation(self.blend_weight)
        else:
            return self.blend_weight
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def select_mesh_by_timestep(self, timestep):
        raise NotImplementedError

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : Optional[BasicPointCloud], spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        if pcd == None:
            assert self.binding is not None
            num_pts = self.binding.shape[0]
            fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda()
            fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        sg_features = torch.zeros((fused_color.shape[0], self.max_sg_degree)).float().cuda()

        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        if self.binding is None:
            dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        else:
            
            scales = torch.log(torch.ones((self.get_xyz.shape[0], 2), device="cuda"))
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self._features_sg = nn.Parameter(sg_features.requires_grad_(True))

        if self.train_normal:
            self._normal = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
            self._normal2 = nn.Parameter(torch.zeros_like(self._xyz).requires_grad_(True))
        
        if self.train_kinematic:

            # identity = inverse_sigmoid(torch.ones((self._xyz.shape[0],1), device="cuda")) #!1
            # neighbour = inverse_sigmoid(torch.ones((self._xyz.shape[0],3)*0.01, device="cuda"))  #!3
            identity = inverse_sigmoid(0.9 * torch.ones((self._xyz.shape[0],1), device="cuda")) #!1
            neighbour = inverse_sigmoid(0.05 * torch.ones((self._xyz.shape[0],3), device="cuda"))

            # identity = inverse_sigmoid(torch.ones((self._xyz.shape[0],1), device="cuda")) #!1
            # neighbour = inverse_sigmoid(torch.zeros((self._xyz.shape[0],3), device="cuda"))
            boundary_mask = torch.isin(self.binding, self.flame_model.mask.f.boundary)
            boundary_indices = torch.nonzero(boundary_mask).squeeze(1)

            if self.detach_boundary:
                identity[boundary_indices] = inverse_sigmoid((1-(1e-10)) * torch.ones((self._xyz[boundary_indices].shape[0],1), device="cuda"))
                neighbour[boundary_indices] = inverse_sigmoid((1e-10) * torch.ones((self._xyz[boundary_indices].shape[0],3), device="cuda"))
                
            bw = torch.cat((identity, neighbour), 1)  #.cuda()
            # self.blend_weight = nn.Parameter(.requires_grad_(True))
            # breakpoint()
            self.blend_weight = nn.Parameter(bw.requires_grad_(True))
            #! self.blend_weight.shape == M, 4
            # self.blended_Jacobian = 
        # breakpoint()
        if self.train_kinematic_dist:
            # self.centroids = self.face_center[self.face_adjacency] #? N,4,3
            self.centroids = self.face_center[self.face_adjacency.cuda()[self.binding]]
            self.blend_weight = get_dist_weight(self.centroids, self.get_xyz_cano)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.tightpruning__mask = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.tight_visibility_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda")  # 초기화

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._features_sg], 'lr': training_args.feature_lr, "name": "f_sg"},
        ]
        if self.train_normal:
            l.append({'params': [self._normal], 'lr': training_args.normal_lr, "name": "normal"})
            l.append({'params': [self._normal2], 'lr': training_args.normal_lr, "name": "normal2"})
        if self.train_kinematic or self.train_kinematic_dist:
            l.append({'params': [self.blend_weight], 'lr': training_args.blend_weight_lr, "name": "blend_weight"})
        # breakpoint()
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.blend_weight_scheduler_args = get_expon_lr_func(lr_init=training_args.blend_weight_lr*self.spatial_lr_scale,
                                                    lr_final=(training_args.blend_weight_lr/100.0)*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if False:
            xyz_done = False; blend_weight_done = False
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                    xyz_done = True
                    # return lr
                if param_group["name"] == "blend_weight":
                    lr = self.blend_weight_scheduler_args(iteration)
                    param_group['lr'] = lr
                    blend_weight_done = True
                    
                if (xyz_done and blend_weight_done):
                    return 
        else:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "xyz":
                    lr = self.xyz_scheduler_args(iteration)
                    param_group['lr'] = lr
                    return lr
            # if para
            

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        if self.train_normal:
            l.extend(['nx2', 'ny2', 'nz2'])
        if self.train_kinematic or self.train_kinematic_dist:
            # for i in range(self.blend_weight.shape[1]):
            #     l.append()
            # l.append('blend_weight')
            for i in range(4):
                l.append('blend_weight_{}'.format(i))
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._features_sg.shape[1]):
            l.append('f_sg_{}'.format(i))
        if self.binding is not None:
            for i in range(1):
                l.append('binding_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        # breakpoint()
        xyz = self._xyz.detach().cpu().numpy()
        if self.train_normal:
            normals = self._normal.detach().cpu().numpy()
            normals2 = self._normal2.detach().cpu().numpy()
        else:    
            normals = np.zeros_like(xyz)
        if self.train_kinematic or self.train_kinematic_dist:
            # blend_weight = self.blend_weight[self.binding].detach().cpu().numpy()
            blend_weight = self.blend_weight.detach().cpu().numpy()

        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        f_sg = self._features_sg.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        # import pdb;pdb.set_trace()
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        if self.train_normal:
            attributes = np.concatenate((xyz, normals, normals2, f_dc, f_rest, opacities, scale, rotation, f_sg), axis=1)
        else:
            # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
            if self.train_kinematic or self.train_kinematic_dist:
                attributes = np.concatenate((xyz, normals, blend_weight, f_dc, f_rest, opacities, scale, rotation, f_sg), axis=1)
            else:
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_sg), axis=1)
        if self.binding is not None:
            binding = self.binding.detach().cpu().numpy()
            attributes = np.concatenate((attributes, binding[:, None]), axis=1)
        # breakpoint()
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        # breakpoint()
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, **kwargs):
        plydata = PlyData.read(path)
        # breakpoint()
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        sg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_sg_")]
        f_sgs = np.zeros((xyz.shape[0], len(sg_names)))
        for idx, attr_name in enumerate(sg_names):
            f_sgs[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_sg = nn.Parameter(torch.tensor(f_sgs, dtype=torch.float, device="cuda").requires_grad_(True))
     
        self.active_sh_degree = self.max_sh_degree

        # optional fields
        binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
        if len(binding_names) > 0:
            binding_names = sorted(binding_names, key = lambda x: int(x.split('_')[-1]))
            binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
            for idx, attr_name in enumerate(binding_names):
                binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
            self.binding = torch.tensor(binding, dtype=torch.int32, device="cuda").squeeze(-1)
        if self.train_normal:
            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            normal2 = np.stack((np.asarray(plydata.elements[0]["nx2"]),
                            np.asarray(plydata.elements[0]["ny2"]),
                            np.asarray(plydata.elements[0]["nz2"])),  axis=1)
            self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
            self._normal2 = nn.Parameter(torch.tensor(normal2, dtype=torch.float, device="cuda").requires_grad_(True))
        if self.train_kinematic or self.train_kinematic_dist:
            # breakpoint()
            #! M x 4
            blend_weight_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("blend_weight")]
            blend_weight_names = sorted(blend_weight_names, key = lambda x: int(x.split('_')[-1]))
            blend_weights = np.zeros((xyz.shape[0], len(blend_weight_names)))
            for idx, attr_name in enumerate(blend_weight_names):
                blend_weights[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # breakpoint()
            # restored_blend_weight = torch.full((self.face_center.shape[0], 4), -1, dtype=blend_weights.dtype)
            # restored_blend_weight[self.binding] = blend_weights
            # self.blend_weight = nn.Parameter(restored_blend_weight.to("cuda").requires_grad_(True))
            self.blend_weight = nn.Parameter(torch.tensor(blend_weights, dtype=torch.float).to("cuda").requires_grad_(True))

            # blend_weight = np.asarray(plydata.elements[0]["blend_weight"])[..., np.newaxis]
            # restored_blend_weight = torch.full((self.face_center.shape[0], 3), -1, dtype=blend_weight.dtype)
            # restored_blend_weight[self.binding] = blend_weight
            # self.blend_weight = nn.Parameter(restored_blend_weight.to("cuda").requires_grad_(True))
            
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if len(group["params"]) != 1 or group["params"][0].shape[0] != mask.shape[0]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    import torch
    from pytorch3d.ops import knn_points

    def recalculate_points_laplacian(self, K=5, fill_value=1e10):
        print('::Recalculate_points_laplacian... Wait for seconds to a single minute:: \n')
        
        p1 = self.get_xyz_cano.unsqueeze(0)  # (N, 1, D)
        p2 = self.get_xyz_cano.unsqueeze(0)  # (1, N, D)
        
        knn_dist, knn_idx,_ = knn_points(p1, p2, K=K, return_sorted=True)
        # breakpoint()
        N = self.get_xyz.shape[0]
        
        # Step 2: 인접 행렬 생성
        A_points = torch.zeros((N, N), dtype=torch.int, device=self.get_xyz.device)
        row_indices = torch.arange(N, device=knn_idx.device).unsqueeze(1).expand(-1, K)
        A_points[row_indices, knn_idx.squeeze(0)] = 1
        
        # 대칭화
        A_points = A_points | A_points.T
        
        # Step 3: 라플라시안 행렬 계산
        D_points = torch.diag_embed(A_points.sum(dim=-1))
        L_points = D_points - A_points  # 라플라시안 행렬 (정규화 없음)
        L_points = L_points / A_points.sum(dim=-1, keepdim=True)
        self.L_points = L_points
        # return L_points
# 이 코드를 클래스 내의 메서드로 사용합니다.


# 이 코드를 클래스 내의 메서드로 사용합니다.

    # def recalculate_points_laplacian(self):
        print('::Recalculate_points_laplacian... Wait for seconds to a single minute:: \n')
        N = self.get_xyz.shape[0]
            # face_points = 
        face_sharing = self.flame_model.face_sharing #! M x M
        M = face_sharing.shape[0]
        self.A_points = torch.zeros((N, N), dtype=torch.int32)

        self.A_normals = torch.zeros((N,N,3), dtype=torch.float)
        # breakpoint()   
        face_to_points = {i: [] for i in range(M)}
        for point_idx, face_idx in enumerate(self.binding):
            face_to_points[face_idx.item()].append(point_idx)
        #! 각 face에 속한 points인덱스의 리스트
        # NxN 매트릭스 채우기
        # normals = self.get_normal
        for i in range(N):
            point_face = self.binding[i]  # 포인트 i가 속한 페이스 인덱스
            shared_faces = torch.nonzero(face_sharing[point_face]).squeeze().tolist()
            if isinstance(shared_faces, int):
                shared_faces = [shared_faces]
            shared_points = set()
            for face in shared_faces:
                shared_points.update(face_to_points[face])
            # self.A_points 채우기
            for j in shared_points:
                self.A_points[i, j] = 1
                # self.A_normals[i, j] = normals[i]
        self.D_points = torch.diag_embed(self.A_points.sum(dim=-1))
        
        self.L_points = self.D_points - self.A_points #! points laplacian without normalization
        self.L_points = self.L_points / self.A_points.sum(dim=-1)[...,None]
        # breakpoint()
        sparse_indices = self.L_points.nonzero().t()
        sparse_values = self.L_points[self.L_points != 0].float()
        sparse_size = self.L_points.shape
        self.L_sparse = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size).cuda()
        return self.L_sparse
        # indices = self.A_points.nonzero().t()
        # self.D_points = torch.diag_embed(self.A_points.sum(dim=-1))
    
        # A_points sparse tensor
        
        # values = self.A_points[self.A_points != 0].float()
        # size = self.A_points.shape
        # A_sparse = torch.sparse_coo_tensor(indices, values, size)
        
        # # D_points sparse tensor
        # D_values = self.D_points[self.D_points != 0].float()
        # D_sparse = torch.sparse_coo_tensor(indices, D_values, size)
        
        # # L_points sparse tensor
        # L_sparse = D_sparse - A_sparse
        
        # # Normalize L_sparse
        # A_sum = torch.sparse.sum(A_sparse, dim=-1).to_dense()
        # A_sum = A_sum[..., None]  # Expand dimensions to match indices shape
        
        # row_indices = indices[0]
        # normalized_values = L_sparse._values() / A_sum[row_indices]
        
        # self.L_points = torch.sparse_coo_tensor(indices, normalized_values, size)
        print('::Recalculate_points_laplacian Has Done::')

        # print('::Recalculate_points_laplacian Has Done!::')


    def prune_points(self, mask):
        # breakpoint()
        if self.binding is not None: #! 이거떄매 계속 살아날수도있음
            # make sure each face is bound to at least one point after pruning
            binding_to_prune = self.binding[mask]
            counter_prune = torch.zeros_like(self.binding_counter)
            counter_prune.scatter_add_(0, binding_to_prune, torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"))
            mask_redundant = (self.binding_counter - counter_prune) > 0
            mask[mask.clone()] = mask_redundant[binding_to_prune]
        
        valid_points_mask = ~mask #! 살아남는 값들
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_sg = optimizable_tensors["f_sg"]
        # breakpoint()
        #! 이미 다들 0
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask] #! 무조건 처음보다 갯수가 작아지기 때문에
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_points_mask]

        self.tight_visibility_mask = self.tight_visibility_mask[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            self.binding_counter.scatter_add_(0, self.binding[mask], -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"))
            self.binding = self.binding[valid_points_mask]
            
        if self.train_normal:
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]
            
        if self.train_kinematic or self.train_kinematic_dist:
            self.blend_weight = optimizable_tensors['blend_weight']
            
            
            # print('!!Calculating Lap. with passing in the tight session!!\n')
            
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if group["name"] not in tensors_dict:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_feature_sg,
                              new_normal=None, new_normal2=None, new_blend_weight=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "f_sg": new_feature_sg}

        if self.train_normal:
            d["normal"] = new_normal
            d["normal2"] = new_normal2
        if self.train_kinematic or self.train_kinematic_dist:
            d["blend_weight"] = new_blend_weight
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_sg = optimizable_tensors["f_sg"]
        
        #! after densification -> reinitialize
        self.tight_visibility_mask = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        if self.train_normal:
            self._normal = optimizable_tensors["normal"]
            self._normal2 = optimizable_tensors["normal2"]
        if self.train_kinematic or self.train_kinematic_dist:
            self.blend_weight = optimizable_tensors['blend_weight']

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # n_init_points = self.get_xyz.shape[0]
        # # Extract points that satisfy the gradient condition
        # padded_grad = torch.zeros((n_init_points), device="cuda")
        # padded_grad[:grads.shape[0]] = grads.squeeze()
        # selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        # padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        # padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        # selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)

        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        # means =torch.zeros((stds.size(0), 3),device="cuda")
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        if self.binding is not None:
            selected_scaling = self.get_scaling[selected_pts_mask]
         
            face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
            new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N,1) / (0.8*N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_feature_sg = self._features_sg[selected_pts_mask].repeat(N, 1)

        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))
        
        if self.train_normal:
            new_normal = self._normal[selected_pts_mask].repeat(N,1)
            new_normal2 = self._normal2[selected_pts_mask].repeat(N,1)
        else:
            new_normal = None
            new_normal2 = None
        
        if self.train_kinematic or self.train_kinematic_dist:
            # breakpoint()
            if False:
                before_blend_weight = self.blend_weight[selected_pts_mask]
                after_blend_weight = torch.tensor([[0.9,0.05,0.05,0.05]]).repeat(before_blend_weight.shape[0],1).to(before_blend_weight)
                new_blend_weight = torch.cat([before_blend_weight,after_blend_weight],dim=0)
            else:
                new_blend_weight = self.blend_weight[selected_pts_mask].repeat(N,1)
        else:
            new_blend_weight = None
            # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal, new_normal2)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_feature_sg, new_normal, new_normal2, new_blend_weight)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False) #! N, 1 # already positive
        # selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        # selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # # breakpoint()
        # #! If the norm of the gradient ∥dL/dx∥2
        # #! is above a predefined threshold
        # #! τx, the Gaussian is chosen as the candidate for densification.
        # try:
        #     selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                         torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # except:
        #     breakpoint()
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_feature_sg = self._features_sg[selected_pts_mask]

        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask]
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))
        if self.train_normal:
            new_normal = self._normal[selected_pts_mask]
            new_normal2 = self._normal2[selected_pts_mask]
        else:
            new_normal = None
            new_normal2 = None
        if self.train_kinematic or self.train_kinematic_dist:
            if False:
                before_blend_weight = self.blend_weight[selected_pts_mask]
                after_blend_weight = torch.tensor([[0.9,0.05,0.05,0.05]]).repeat(before_blend_weight.shape[0],1).to(before_blend_weight)
                new_blend_weight = torch.cat([before_blend_weight,after_blend_weight],dim=0)
            else:
                new_blend_weight = self.blend_weight[selected_pts_mask]
        else:
            new_blend_weight = None
            
            # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal, new_normal2)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_feature_sg,new_normal, new_normal2, new_blend_weight)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, \
              detach_eyeball_geometry=False):#, tight_pruning=False):
        # if tight_pruning:
        #     killing_mask = torch.logical_not(self.tight_visibility_mask)
        #     # breakpoint()
        #     print(f'By tight: {killing_mask.sum()} / Whole: {killing_mask.shape[0]}')
        #     self.prune_points(killing_mask) #!여기서 self.xyz_gradient_accum 사이즈가(valid) 바뀜 
            
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0
        
        # grads_abs = self.xyz_gradient_accum_abs / self.denom
        # grads_abs[grads_abs.isnan()] = 0.0
        
        # breakpoint()
        # if self.densification_type == 'arithmetic_mean':
        #     grads = self.xyz_gradient_accum / self.denom
        #     grads_abs = self.xyz_gradient_accum_abs / self.denom
        # elif self.densification_type == 'geometric_mean2':
        #     grads = (self.xyz_gradient_accum / self.denom) ** 0.5
        #     grads_abs = (self.xyz_gradient_accum_abs / self.denom) ** 0.5
        # else:
        #     raise NotImplementedError
        
        # grads[grads.isnan()] = 0.0
        # grads_abs[grads_abs.isnan()] = 0.0
        
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        
        if detach_eyeball_geometry:
            eyeball_mask = torch.isin(self.binding, self.flame_model.mask.f.eyeballs)

            print(eyeball_mask.sum(), 'EYEBALLS::Check the number of eyeballs')
            eyeball_indices = torch.nonzero(eyeball_mask).squeeze(1)
            grads[eyeball_indices] = 0.0
            # grads_abs[eyeball_indices] = 0.0
        # if detach_teeth_geometry:
        #     teeth_mask = torch.isin(self.binding, self.flame_model.mask.f.teeth)
        #     print(teeth_mask.sum(), 'TEETH')
        #     teeth_indices = torch.nonzero(teeth_mask).squeeze(1)
        #     grads[teeth_indices] = 0.0
            # grads_abs[teeth_indices] = 0.0
            
        # ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        # Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)


        # self.densify_and_clone(grads, max_grad, extent)#! 여기서는 postfix 때문에 xyz_gradient_accum 모든값을 0으로 만듦
        # #! 
        # self.densify_and_split(grads, max_grad, extent)#! postfix(to zero) + pruning(size)

        before = self._xyz.shape[0]
        # self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        self.densify_and_clone(grads, max_grad, extent)
        clone = self._xyz.shape[0]
        # self.densify_and_split(grads, max_grad, grads_abs, Q, extent)
        self.densify_and_split(grads, max_grad, extent)
        split = self._xyz.shape[0]

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
            # prune_mask torch.Size([11458])
            # breakpoint()
       
            
            #! 살리는게 하나도 없으면
            #! 다 True가 되어서 다죽임
       
        self.prune_points(prune_mask)

        # self.tight_visibility_mask = torch.zeros_like(self.tight_visibility_mask)
        prune = self._xyz.shape[0]
        # torch.cuda.empty_cache()
        print(f'Densification :: Clone: {clone - before} / Split: {split - clone} / Prune: {split - prune} / TOTAL ADD: {prune - before} \n')
        print(f'Before: {before} / After: {prune}')
        torch.cuda.empty_cache()
        return clone - before, split - clone, split - prune
    
        



    def add_densification_stats(self, viewspace_point_tensor, update_filter, amplify_teeth_grad=False):
        # breakpoint()
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        if amplify_teeth_grad:
            # breakpoint()
            teeth_mask = torch.isin(self.binding, self.flame_model.mask.f.teeth)
            # teeth_lower_mask = torch.isin(self.binding, self.flame_model.mask.f.teeth_lower)
            teeth_indices = torch.nonzero(teeth_mask).squeeze(1)
            
            viewspace_point_tensor.grad[teeth_indices] *= 20
            # breakpoint()
        # if self.densification_type == 'arithmetic_mean':
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        # self.xyz_gradient_accum_abs_max[update_filter] = torch.max(self.xyz_gradient_accum_abs_max[update_filter], torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True))
     
        # self.denom[update_filter] += 2
        
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        # self.denom[update_filter] += 1
        
        # if visibility_filter_tight is not None:
        #     # breakpoint()
        #     self.tight_visibility_mask = torch.logical_or(self.tight_visibility_mask, visibility_filter_tight)
            # print(self.tight_visibility_mask.sum(),'SUM')