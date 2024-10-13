#!/bin/bash

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <gpu_id>"
    exit 1
fi

# Extract the GPU ID from the command-line argument
GPU_ID=$1
SUBJECT=304 #! example subject id 
prefix_data='NeRSemble_data_path'
port=60000 #! example port number

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla] \
--port ${port} --eval --white_background --bind_to_mesh --lambda_normal 0.05 --lambda_dist 100.0 --depth_ratio 1 \
--interval 60000 \
--iterations 300000 \
--densify_until_iter 150000 \
--densify_from_iter 5000 \
--opacity_reset_interval 30000 \
--densification_interval 1000 \
--position_lr_max_steps 300000 \
--rm_bg --amplify_teeth_grad \
--detach_eyeball_geometry --lambda_eye_alpha 0.1 --SGs --sg_type 'asg' --spec_only_eyeball \

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian] \
--port ${port} --eval --white_background --bind_to_mesh --lambda_normal 0.05 --lambda_dist 100.0 --depth_ratio 1 \
--interval 60000 \
--iterations 300000 \
--densify_until_iter 150000 \
--densify_from_iter 5000 \
--opacity_reset_interval 30000 \
--densification_interval 1000 \
--position_lr_max_steps 300000 \
--rm_bg --amplify_teeth_grad \
--detach_eyeball_geometry --lambda_eye_alpha 0.1 --SGs --sg_type 'asg' --spec_only_eyeball \
--DTF --invT_Jacobian --lambda_normal_norm 0.01 

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS] \
--port ${port} --eval --white_background --bind_to_mesh --lambda_normal 0.05 --lambda_dist 100.0 --depth_ratio 1 \
--interval 60000 \
--iterations 300000 \
--densify_until_iter 150000 \
--densify_from_iter 5000 \
--opacity_reset_interval 30000 \
--densification_interval 1000 \
--position_lr_max_steps 300000 \
--rm_bg --amplify_teeth_grad \
--detach_eyeball_geometry --lambda_eye_alpha 0.1 --SGs --sg_type 'asg' --spec_only_eyeball \
--DTF --invT_Jacobian --lambda_normal_norm 0.01 \
--train_kinematic --rotSH --detach_boundary


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS][-eyeballs] \
--port ${port} --eval --white_background --bind_to_mesh --lambda_normal 0.05 --lambda_dist 100.0 --depth_ratio 1 \
--interval 60000 \
--iterations 300000 \
--densify_until_iter 150000 \
--densify_from_iter 5000 \
--opacity_reset_interval 30000 \
--densification_interval 1000 \
--position_lr_max_steps 300000 \
--rm_bg --amplify_teeth_grad \
--DTF --invT_Jacobian --lambda_normal_norm 0.01 \
--train_kinematic --rotSH --detach_boundary \

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS][+light_asg] \
--port ${port} --eval --white_background --bind_to_mesh --lambda_normal 0.05 --lambda_dist 100.0 --depth_ratio 1 \
--interval 60000 \
--iterations 300000 \
--densify_until_iter 150000 \
--densify_from_iter 5000 \
--opacity_reset_interval 30000 \
--densification_interval 1000 \
--position_lr_max_steps 300000 \
--rm_bg --amplify_teeth_grad \
--detach_eyeball_geometry --lambda_eye_alpha 0.1 --SGs --sg_type 'lasg' --spec_only_eyeball \
--DTF --invT_Jacobian --lambda_normal_norm 0.01 \
--train_kinematic --rotSH --detach_boundary


