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
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, laplacian_loss, laplacian_loss_U, get_effective_rank

from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel, SpecularModel
from utils.general_utils import safe_state, colormap
import uuid
from tqdm import tqdm


from utils.image_utils import psnr, error_map, visualize_gaussians_with_tensor
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from kornia.losses import inverse_depth_smoothness_loss

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    
    if pipe.SGs:
        # tb_writer = prepare_output_and_logger(dataset)
        specular_mlp = SpecularModel(sg_type=pipe.sg_type)
        specular_mlp.train_setting(opt)
    else:
        specular_mlp = None
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:#!
        
        train_normal = False
        if os.path.basename(dataset.source_path).startswith("FaceTalk"):
            # breakpoint()
            n_shape = 100    
            n_expr = 50
        else:
            n_shape = 300
            n_expr = 100    
        
        cal_laplacian = False
       
    
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.sg_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,\
            train_normal = train_normal, n_shape=n_shape, n_expr=n_expr, train_kinematic=pipe.train_kinematic, train_kinematic_dist = pipe.train_kinematic_dist,\
            DTF = pipe.DTF, invT_Jacobian=pipe.invT_Jacobian,
                densification_type=opt.densification_type, detach_eyeball_geometry = pipe.detach_eyeball_geometry,detach_boundary = pipe.detach_boundary)
        try:
            mesh_renderer = NVDiffRenderer()
        except:
            mesh_renderer = None
            print("Mesh renderer not available")
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt) #! from here all trainble parameters are set(e.g., flame_codes, xyz, scaling, dynamic_offset, opacity)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # breakpoint()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
   
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    
    
    
    for iteration in range(first_iter, opt.iterations + 1):     
      
        # han_window_iter = iteration * 2/(opt.iterations + 1)  
        han_window_iter = iteration /(opt.iterations + 1)  

        if network_gui.conn == None:
            network_gui.try_connect()
            
        # breakpoint()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        kick_in_SGs = 2500 #! This works best for growing up SHs
        

        if iteration > kick_in_SGs and pipe.SGs:
            if pipe.train_kinematic or pipe.train_kinematic_dist:
                dir_pp = (gaussians.get_blended_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = gaussians.get_normal
                normal_normalised = F.normalize(normal,dim=-1).detach()
                # breakpoint()
                
                normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                if pipe.rotSH:
                    dir_pp_normalized = (gaussians.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
            else:
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1).cuda())
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                normal = gaussians.get_normal
                normal_normalised = F.normalize(normal,dim=-1).detach()
                # breakpoint()
                
                normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                if pipe.rotSH:
                    assert pipe.DTF
                    dir_pp_normalized = (gaussians.face_R_mat[gaussians.binding].permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
  
           
            if pipe.spec_only_eyeball:
                # breakpoint()
                mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.eyeballs)
                points_indices = torch.nonzero(mask).squeeze(1)
                
                specular_color_eyeballs = specular_mlp.step(gaussians.get_sg_features[points_indices], dir_pp_normalized[points_indices], normal[points_indices].detach(), sg_type = pipe.sg_type)
                
                
                specular_color = specular_color_eyeballs
            else:
                specular_color = specular_mlp.step(gaussians.get_sg_features, dir_pp_normalized, normal.detach(), sg_type = pipe.sg_type)
    
        else:
            specular_color = None
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                            backface_culling_smooth = dataset.backface_culling_smooth,
                                            backface_culling_hard = dataset.backface_culling_hard,
                                            iter = han_window_iter,
                                            specular_color= specular_color,
                                            spec_only_eyeball = pipe.spec_only_eyeball)
                                        
        
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
       
        
        # visibility_filter_tight = render_pkg.get("visibility_filter_tight", None)
        
        gt_image = viewpoint_cam.original_image.cuda()
        
       
        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim
        
        if iteration == 1 or ((iteration - 1) % opt.densification_interval == 0 \
            and (iteration - 1) >= opt.densify_from_iter):
            if opt.lambda_laplacian != 0:
                pre_compute_laplacian = gaussians.recalculate_points_laplacian()
        #! from 2dgs
        # regularization
        if os.path.basename(dataset.source_path).startswith("FaceTalk"):
            lambda_normal = opt.lambda_normal if iteration > 70000 else 0.0
            lambda_dist = opt.lambda_dist if iteration > 30000 else 0.0
        else:
            if opt.iterations < 600000:
                lambda_normal = opt.lambda_normal if iteration > 70000 else 0.0
                lambda_dist = opt.lambda_dist if iteration > 30000 else 0.0
            else:
                lambda_normal = opt.lambda_normal if iteration > 140000 else 0.0
                lambda_dist = opt.lambda_dist if iteration > 60000 else 0.0

        if opt.lambda_eye_alpha != 0:
            # normal = gaussians.get_normal
            mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.eyeballs)
            
            points_indices = torch.nonzero(mask).squeeze(1)
            losses['eye_alpha'] = ((gaussians.get_opacity[points_indices] - 1) ** 2).mean() * opt.lambda_eye_alpha

        if opt.lambda_blend_weight != 0:
        
            losses['blend_weight'] = opt.lambda_blend_weight * \
                F.relu(F.normalize(gaussians.get_blend_weight[visibility_filter],dim=-1,p=1)[...,1:] - 0.1).norm(dim=1).mean()
          
        if opt.lambda_normal_norm != 0 and (pipe.DTF):
            if pipe.train_kinematic or pipe.train_kinematic_dist:
                view_dir = gaussians.get_blended_xyz - viewpoint_cam.camera_center.cuda()
            else:
                view_dir = gaussians.get_xyz - viewpoint_cam.camera_center.cuda()
            normal = gaussians.get_normal#[visibility_filter]
            normal_normalised=  F.normalize(normal,dim=-1).detach()
            normal = normal * ((((view_dir * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
            normal = normal[visibility_filter]
            losses['normal_norm'] \
                 = torch.abs(normal.norm(dim=1) - 1).mean() * opt.lambda_normal_norm
             
        if lambda_normal != 0:
           
            rend_normal = render_pkg['surfel_rend_normal']
            surf_normal = render_pkg['surfel_surf_normal']
            
            if pipe.rm_bg:
                gt_mask = viewpoint_cam.original_mask.cuda()
                # surf_normal = F.normalize(surf_normal * gt_mask.repeat(3,1,1), dim =0)
                surf_normal = surf_normal * gt_mask.repeat(3,1,1)
                
                rend_normal = rend_normal * gt_mask.repeat(3,1,1)
                # if False:
                rend_normal = F.normalize(rend_normal, dim = 0)
            else:
                # surf_normal = render_pkg['surfel_surf_normal']
                pass
            
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = lambda_normal * (normal_error).mean()
            losses['surfel_normal_loss'] = normal_loss

        if lambda_dist != 0:
            rend_dist = render_pkg["surfel_rend_dist"]
            dist_loss = lambda_dist * (rend_dist).mean()
            losses['surfel_dist_loss'] = dist_loss
        
        if opt.lambda_ids != 0:
            # breakpoint()
            depth = render_pkg["surfel_surf_depth"]
            if opt.half_res:
                
                losses['ids'] = inverse_depth_smoothness_loss(F.interpolate(depth[None],scale_factor=0.5)
                                        , F.interpolate(image[None],scale_factor=0.5)) * opt.lambda_ids
            else:
                losses['ids'] = inverse_depth_smoothness_loss(depth[None], image[None]) * opt.lambda_ids
        
        
        if gaussians.binding != None:
                
            if opt.lambda_xyz != 0:
                if opt.metric_xyz:
                    losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            
                else:
                    losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                # breakpoint()
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
        

                
        losses['total'] = sum([v for k, v in losses.items()])
        
        losses['total'].backward()
        if pipe.detach_eyeball_geometry:
            mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.eyeballs)


            points_indices = torch.nonzero(mask).squeeze(1)
            gaussians._xyz.grad[points_indices] = 0
            gaussians._rotation.grad[points_indices] = 0
            if pipe.train_kinematic:
                gaussians.blend_weight.grad[points_indices] = 0
            
        if pipe.detach_boundary:
            boundary_mask = torch.isin(gaussians.binding, gaussians.flame_model.mask.f.boundary)
            boundary_indices = torch.nonzero(boundary_mask).squeeze(1)
            if pipe.train_kinematic:
                gaussians.blend_weight.grad[boundary_indices] = 0

        iter_end.record()
       
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if lambda_dist != 0:
                ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            else:
                ema_dist_for_log = 0.0
            if lambda_normal != 0:
                ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            else:
                ema_normal_for_log = 0.0

            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                if 'alpha' in losses:
                    postfix["alpha"] = f"{losses['alpha']:.{7}f}"
                if 'surfel_normal_loss' in losses:
                    postfix["surfel_normal_loss"] = f"{ema_normal_for_log:.{7}f}"
                    # postfix["surfel_normal_loss"] = f"{losses['surfel_normal_loss']:.{7}f}"
                if 'surfel_dist_loss' in losses:
                    # postfix["surfel_dist_loss"] = f"{losses['surfel_dist_loss']:.{7}f}"
                    postfix["surfel_dist_loss"] = f"{ema_dist_for_log:.{7}f}"
                if 'normal' in losses:
                    postfix["normal"] = f"{losses['normal']:.{7}f}"
             
                if 'normal_norm' in losses:
                    postfix['normal_norm'] = f"{losses['normal_norm']:.{7}f}"
       
                if 'lap_lbs' in losses:
                    postfix['lap_lbs'] = f"{losses['lap_lbs']:.{7}f}"
                if 'convex_eyeballs' in losses:
                    postfix['convex_eyeballs'] = f"{losses['convex_eyeballs']:.{7}f}"
                if 'eye_alpha' in losses:
                    postfix['eye_alpha'] = f"{losses['eye_alpha']:.{7}f}"
            
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            render_args = (pipe, background, 1.0, None, dataset.backface_culling_smooth, dataset.backface_culling_hard, han_window_iter)
            
       
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), \
                            testing_iterations, scene, render, render_args, specular_mlp)
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if pipe.SGs and iteration > kick_in_SGs:
                    specular_mlp.save_weights(args.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, pipe.amplify_teeth_grad)
               
        
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    #! 10000 12000 ...
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold,
                                                pipe.detach_eyeball_geometry)

                

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    #! 10000 60000 120000 ...
                    
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
                if pipe.SGs and iteration > kick_in_SGs:
                    specular_mlp.optimizer.step()
                    specular_mlp.optimizer.zero_grad()
                    specular_mlp.update_learning_rate(iteration)
                    # breakpoint()
            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, specular_mlp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        if 'alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/alpha_loss', losses['alpha'].item(), iteration)
    
        if 'normal' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_loss', losses['normal'].item(), iteration)
        if 'surfel_normal_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_normal_loss', losses['surfel_normal_loss'].item(), iteration)
        if 'surfel_dist_loss' in losses:
            tb_writer.add_scalar('train_loss_patches/surfel_dist_loss', losses['surfel_dist_loss'].item(), iteration)

        if 'blend_weight' in losses:
            tb_writer.add_scalar('train_loss_patches/blend_weight', losses['blend_weight'].item(), iteration)
        if 'normal_norm' in losses:
            tb_writer.add_scalar('train_loss_patches/normal_norm', losses['normal_norm'].item(), iteration)
     
        if 'convex_eyeballs' in losses:
            tb_writer.add_scalar('train_loss_patches/convex_eyeballs', losses['convex_eyeballs'].item(), iteration)
        if 'eye_alpha' in losses:
            tb_writer.add_scalar('train_loss_patches/eye_alpha', losses['eye_alpha'].item(), iteration)

        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    testing_iterations_rough = [_iter * 2 for _iter in testing_iterations]
    if iteration in testing_iterations: #or iteration in [5000, 10000]:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    
                    
                    
              
                    specular_color=None
                    try:
                        if renderArgs[0].train_kinematic or renderArgs[0].train_kinematic_dist:
                            dir_pp = (scene.gaussians.get_blended_xyz - viewpoint.camera_center.repeat(
                                scene.gaussians.get_features.shape[0], 1).cuda())
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            normal_normalised=  F.normalize(scene.gaussians.get_normal,dim=-1).detach()
                            normal = scene.gaussians.get_normal
                            normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                            if renderArgs[0].rotSH:
                                dir_pp_normalized = (scene.gaussians.blended_R.permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)

                        else:
                            dir_pp = (scene.gaussians.get_xyz - viewpoint.camera_center.repeat(
                                scene.gaussians.get_features.shape[0], 1).cuda())
                            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                            normal_normalised=  F.normalize(scene.gaussians.get_normal,dim=-1).detach()
                            normal = scene.gaussians.get_normal
                            normal = normal * ((((dir_pp_normalized * normal_normalised).sum(dim=-1) < 0) * 1 - 0.5) * 2)[...,None]
                            if renderArgs[0].rotSH:
                                dir_pp_normalized = (scene.gaussians.face_R_mat[scene.gaussians.binding].permute(0,2,1) @ dir_pp_normalized[...,None]).squeeze(-1)
                                
                        if renderArgs[0].spec_only_eyeball:
                            mask = torch.isin(scene.gaussians.binding, scene.gaussians.flame_model.mask.f.eyeballs)

                            points_indices = torch.nonzero(mask).squeeze(1)
                            
                            specular_color_eyeballs = specular_mlp.step(scene.gaussians.get_sg_features[points_indices], dir_pp_normalized[points_indices], normal[points_indices], sg_type = renderArgs[0].sg_type)
                            
                          
                            specular_color = specular_color_eyeballs
                        
                        else:
                            specular_color = specular_mlp.step(scene.gaussians.get_sg_features, dir_pp_normalized, normal.detach(), sg_type = renderArgs[0].sg_type)
                        # print('Specular Forwarded')
                    except:
                        pass
                    # breakpoint()
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, specular_color, renderArgs[0].spec_only_eyeball)
                    
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_mask = torch.clamp(viewpoint.original_mask.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        #! from NRFF
                    
                        try:
                            diffuse = render_pkg["rend_diffuse"]
                            specular = render_pkg['rend_specular']
                            tb_writer.add_images(config['name'] + "_{}/render_diffuse".format(vis_ct), diffuse[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular".format(vis_ct), specular[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_2x".format(vis_ct), specular[None] * 2, global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/render_specular_4x".format(vis_ct), specular[None] * 4, global_step=iteration)
                        except:
                            pass
                
                        #! from 2dgs
                        
                        depth = render_pkg["surfel_surf_depth"]
                        alpha = render_pkg['surfel_rend_alpha']
                        # breakpoint()
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
                        tb_writer.add_images(config['name'] + "_{}/surfel_depth".format(vis_ct), depth[None], global_step=iteration)
                
                            
                        try:
                            rend_alpha = render_pkg['surfel_rend_alpha']
                            rend_normal = render_pkg["surfel_rend_normal"] * 0.5 + 0.5
                            surf_normal = (render_pkg["surfel_surf_normal"] * 0.5 + 0.5) * gt_mask.repeat(3,1,1).cuda + (1 - gt_mask.repeat(3,1,1).cuda())
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_render_normal".format(vis_ct), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_normal".format(vis_ct), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_{}/surfel_surfel_alpha".format(vis_ct), rend_alpha[None], global_step=iteration)
                          
                            rend_dist = render_pkg["surfel_rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass
                        
                        vis_ct += 1
                        
                        
                    image = viewpoint.original_mask_face.cuda() * image
                    gt_image = viewpoint.original_mask_face.cuda() * gt_image
                    # breakpoint()
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/scale_histogram", torch.mean(scene.gaussians.get_scaling, dim=-1), iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            try:
                tb_writer.add_histogram('scene/blend_weight_primary',\
                    F.normalize(scene.gaussians.get_blend_weight, dim=-1)[...,:1], iteration)
            except:
                pass
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=30_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
