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


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla] \
--skip_train --skip_val \
--white_background --bind_to_mesh --depth_ratio 1 \
--rm_bg \
--SGs --sg_type 'asg' --spec_only_eyeball \

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian] \
--skip_train --skip_val \
--white_background --bind_to_mesh --depth_ratio 1 \
--rm_bg \
--SGs --sg_type 'asg' --spec_only_eyeball \
--DTF --invT_Jacobian


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS] \
--skip_train --skip_val \
--white_background --bind_to_mesh --depth_ratio 1 \
--rm_bg \
--SGs --sg_type 'asg' --spec_only_eyeball \
--DTF --invT_Jacobian \
--train_kinematic --rotSH --detach_boundary


OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS][-eyeballs] \
--skip_train --skip_val \
--white_background --bind_to_mesh --depth_ratio 1 \
--rm_bg \
--DTF --invT_Jacobian \
--train_kinematic --rotSH --detach_boundary

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU_ID python render.py \
-s ${prefix_data}/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/[300K]UNION10EMOEXP_${SUBJECT}_[vanilla][+Jacobian][+JBS][+light_asg] \
--skip_train --skip_val \
--white_background --bind_to_mesh --depth_ratio 1 \
--rm_bg \
--SGs --sg_type 'lasg' --spec_only_eyeball \
--DTF --invT_Jacobian \
--train_kinematic --rotSH --detach_boundary


