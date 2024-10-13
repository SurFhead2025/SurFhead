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
import math
from typing import Union
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh
from utils.mesh_utils import world_to_camera, compute_face_normals
from utils.loss_utils import hann_window
from utils.general_utils import build_rotation
from utils.point_utils import depth_to_normal


def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, \
    bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, \
    backface_culling_smooth = False, backface_culling_hard = False, iter = 0, specular_color = None, spec_only_eyeball = False
        ):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    iter_ = iter
    render_bucket = {}
    # asset_bucket = {}
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    eyeball_mask = torch.isin(pc.binding, pc.flame_model.mask.f.eyeballs)
    eyeball_indices = torch.nonzero(eyeball_mask).squeeze(1)
    

    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug,
        tight_pruning_threshold = pipe.tight_pruning_threshold#*
    )
    # breakpoint()
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if pipe.train_kinematic:
        means3D = pc.get_blended_xyz
        # if pipe.detach_eyeball_geometry:
        #     means3D[eyeball_indices] = means3D[eyeball_indices].detach()
    else:
        means3D = pc.get_xyz
        # if pipe.detach_eyeball_geometry:
        #     means3D[eyeball_indices] = means3D[eyeball_indices].detach()
    means2D = screenspace_points
    
    opacity = pc.get_opacity
    

    
    only_facial = False
    only_boundary = False
    if only_facial:
        # if True:
        # breakpoint()
        facial_mask = torch.isin(pc.binding, pc.flame_model.mask.f.face)
        facial_indices = torch.nonzero(facial_mask).squeeze(1)
    if only_boundary:
        boundary_mask = torch.isin(pc.binding, pc.flame_model.mask.f.boundary)
        boundary_indices = torch.nonzero(boundary_mask).squeeze(1)
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
      
        rotations = pc.get_rotation
        # if pipe.detach_eyeball_geometry:
        #     rotations[eyeball_indices] = rotations[eyeball_indices].detach()
        

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # print(rotations[eyeball_indices][0],'rotations_eb')
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pipe.rotSH:
                if pipe.train_kinematic:
                    dir_pp_normalized = (pc.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                else:
                    breakpoint()
                    dir_pp_normalized = (pc.face_R_mat.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                    
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:#!
            shs = pc.get_features#!N,16,3
    else:
        colors_precomp = override_color
        
    if pipe.SGs:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda())
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree,
                        shs_view,
                        dir_pp_normalized)
        if specular_color is None:
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # print('1')
            # print(sh2rgb.mean())
            if spec_only_eyeball:
                specular_color_full = torch.zeros_like(sh2rgb).to(sh2rgb)
                # breakpoint()
                specular_color_full[eyeball_indices] = specular_color
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + specular_color_full
            else:
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + specular_color
        
        shs = None
   
    
    try:
        means3D.retain_grad()
    except:
        pass
    
    if pipe.DTF: #! to consensus with the convention with glm, matrices must be transposed as input. 
   
        if pipe.train_kinematic:
            if pipe.invT_Jacobian:
                Jacobians = pc.blended_Jacobian.permute(0,2,1)#[pc.binding]
                Jacobians_inv = torch.linalg.inv(Jacobians.permute(0,2,1))#.permute(0,2,1)
            else:#! revive for the ablation
                raise NotImplementedError
                Jacobians = pc.blended_Jacobian.permute(0,2,1)#[pc.binding]
                Jacobians_inv = Jacobians#.permute(0,2,1)
         
        else:
            if pipe.invT_Jacobian:
                # breakpoint()
                Jacobians = pc.face_trans_mat[pc.binding].permute(0,2,1)
                Jacobians_inv = torch.linalg.inv(pc.face_trans_mat)[pc.binding]#.permute(0,2,1).permute(0,2,1)
                #! double transpose is identity -> J^-T for normal vector
            else:#! revive for the ablation
                raise NotImplementedError
                Jacobians = pc.face_trans_mat[pc.binding].permute(0,2,1)#.detach()
                Jacobians_inv = Jacobians
    else:
        
        Jacobians = torch.eye(3)[None].repeat(means3D.shape[0],1,1).to(means3D)
        Jacobians_inv = Jacobians

    if False:
        # breakpoint()
        from utils.quaternion_utils import init_predefined_omega
        #! for check where is the frontal hemisphere
        n_of_eye = 32
        num_theta = 4; num_phi = 8
        omega, omega_la, omega_mu = init_predefined_omega(num_theta, num_phi, type = 'frontal')
        # omega = omega.view(1, num_theta, num_phi, 3).to('cuda')#! incoming direction; lobe direction
        # omega_la = omega_la.view(1, num_theta, num_phi, 3).to('cuda')#! incoming tangent direction
        # omega_mu = omega_mu.view(1, num_theta, num_phi, 3).to('cuda')#! incoming bitangent direction
        
        omega = omega.view(-1, 3).to('cuda')
        #! REDCOLOR
        colour = torch.tensor([1,0,0]).view(1,3).repeat(n_of_eye,1).to('cuda').float()
        # colour_G = torch.tensor([0,0,0]).view(1,3).repeat(n_of_eye,1).to('cuda')
        # colour_B = torch.tensor([0,0,0]).view(1,3).repeat(n_of_eye,1).to('cuda')
        # colour = torch.cat([colour_R, colour_G, colour_B], dim=0).to('cuda')
        Jacobians = torch.eye(3)[None].repeat(means3D.shape[0],1,1).to(means3D)
        rendered_image = rasterizer(                                                                                   

            means3D = omega*0.1,
            means2D = means2D[:n_of_eye],
            shs = None,
            colors_precomp = colour,
            opacities = torch.ones_like(opacity)[:n_of_eye]*1,
            scales = torch.ones_like(scales)[:n_of_eye] *0.01,
            rotations = rotations[:n_of_eye],
            cov3D_precomp = cov3D_precomp,
            Jacobians = Jacobians[:n_of_eye],
            Jacobians_inv = Jacobians_inv[:n_of_eye])[0]#*
        from torchvision.utils import save_image as si; si(rendered_image,'omega.png')
        si(rendered_image,'omega_image.png')
        breakpoint()
        
    
    rendered_image, radii, allmap, n_contrib_pixel, top_weights, top_depths, visible_points_tight = rasterizer(                                                                                   

        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        Jacobians = Jacobians,
        Jacobians_inv = Jacobians_inv)#*
    
    
    render_bucket.update({"render": rendered_image, #* 0-1
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "visibility_filter_tight": visible_points_tight,
            "top_weights": top_weights,
            "top_depths": top_depths,
            "n_contrib_pixel": n_contrib_pixel,#*
            })
    
  
    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    render_normal = allmap[2:5]

    #! world normal
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T.cuda())).permute(2,0,1)

    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_intersect = allmap[0:1]
   
    render_depth_expected = (render_depth_intersect / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

    render_dist = allmap[6:7]
    

    #? psedo surface attributes
    #? surf depth is either median or expected by setting depth_ratio to 1 or 0
    #? for bounded scene, use median depth, i.e., depth_ratio = 1; 
    #? for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    if pipe.depth_ratio == -1:
        assert False, "insersect depth is not valid, set 1(median) or 0(mean)"
        surf_depth = render_depth_intersect
    else:
        surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    # breakpoint()
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()
    
    render_bucket.update({
            'surfel_rend_alpha': render_alpha,
            'surfel_rend_normal': render_normal,
            'surfel_rend_dist': render_dist,
            'surfel_surf_depth': surf_depth,
            'surfel_surf_normal': surf_normal,
    })
    if pipe.SGs:
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda())
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree,
                        shs_view,
                        dir_pp_normalized)
        # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) + mlp_color
        #! 여기서 분리 
        colors_precomp_diffuse = torch.clamp_min(sh2rgb + 0.5 , 0.0)
        
        rendered_diffuse = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp_diffuse,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            Jacobians = Jacobians,
            Jacobians_inv = Jacobians_inv)[0]
        
        if specular_color is None:
            colors_precomp_specular = torch.zeros_like(colors_precomp_diffuse).to(colors_precomp_diffuse)
        else:
            # if spec_only_eyeball:
            #     # specular_color_full = torch.zeros_like(sh2rgb).to(sh2rgb)
            #     # specular_color_full[eyeball_indices] = specular_color
            #     colors_precomp_specular = specular_color_full
            # else:
            colors_precomp_specular = specular_color

        
        # breakpoint()
        if spec_only_eyeball:
            
            rendered_specular = rasterizer(
                means3D = means3D[eyeball_indices],
                means2D = means2D[eyeball_indices],
                shs = shs,
                colors_precomp = colors_precomp_specular,
                opacities = opacity[eyeball_indices],
                scales = scales[eyeball_indices],
                rotations = rotations[eyeball_indices],
                cov3D_precomp = cov3D_precomp,
                Jacobians = Jacobians[eyeball_indices],
                Jacobians_inv = Jacobians_inv[eyeball_indices])[0]
        else:
            rendered_specular = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp_specular,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp,
                Jacobians = Jacobians,
                Jacobians_inv = Jacobians_inv)[0]
            
        render_bucket.update({
                'rend_diffuse': rendered_diffuse,
                'rend_specular': rendered_specular,
        })
        if only_facial: 
            rendered_facial = rasterizer(
                    means3D = means3D[facial_indices],
                    means2D = means2D[facial_indices],
                    shs = None,
                    colors_precomp = colors_precomp[facial_indices],
                    opacities = opacity[facial_indices],
                    scales = scales[facial_indices],
                    rotations = rotations[facial_indices],
                    cov3D_precomp = None,
                    Jacobians = Jacobians[facial_indices],
                    Jacobians_inv = Jacobians_inv[facial_indices])[0]
            breakpoint()
        if only_boundary:
            from torchvision.utils import save_image as si
            rendered_boundary = rasterizer(
                    means3D = means3D[boundary_indices],
                    means2D = means2D[boundary_indices],
                    shs = None,
                    colors_precomp = colors_precomp[boundary_indices],
                    opacities = opacity[boundary_indices],
                    scales = scales[boundary_indices],
                    rotations = rotations[boundary_indices],
                    cov3D_precomp = None,
                    Jacobians = Jacobians[boundary_indices],
                    Jacobians_inv = Jacobians_inv[boundary_indices])[0]
            breakpoint()
    return render_bucket

 