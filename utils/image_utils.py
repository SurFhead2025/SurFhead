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
from matplotlib import cm
import numpy as np
import cv2

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def error_map(img1, img2):
    error = (img1 - img2).mean(dim=0) / 2 + 0.5
    cmap = cm.get_cmap("seismic")
    error_map = cmap(error.cpu())
    return torch.from_numpy(error_map[..., :3]).permute(2, 0, 1)

def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

def erode(img_in, erode_size=4):
    img_out = np.copy(img_in)
    kernel = np.ones((erode_size, erode_size), np.uint8)
    img_out = cv2.erode(img_out, kernel, iterations=1)

    return img_out



import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from io import BytesIO

def visualize_gaussians_with_tensor(image_tensor, depth_row, weight_row, row_to_draw, W, K, vmax=0.1):
    fig, ax = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [1, 3, 1]})
    
    # 이미지 시각화 및 가로선 그리기
    img = image_tensor.permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC
    if img.shape[2] == 4:
        img = img[:, :, :3]  # 알파 채널 제거
    ax[0].imshow(img)
    ax[0].axhline(y=row_to_draw, color='r', linestyle='--')
    ax[0].set_title('Rendered Image with Horizontal Line')
    ax[0].axis('off')
    
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = plt.cm.Blues

    scatter_plots = []

    # 깊이 값 시각화
    for k in range(K):
        valid_mask = depth_row[:, k] != -1  # -1이 아닌 유효한 값만 선택
        if valid_mask.sum() > 0:
            weights = weight_row[:, k][valid_mask]
            colors = cmap(norm(weights))
            colors[weights > vmax] = [0, 0, 0, 1]  # weights가 vmax보다 크면 검은색
            # sizes = weights * 100  # 가중치에 따라 점의 크기 조절
            sizes = 80
            # scatter = ax[1].scatter(np.arange(valid_mask.sum()), depth_row[:, k][valid_mask], c=colors, s=sizes, alpha=0.7, label=f'Gaussian {K}' if k == 0 else "")
            scatter = ax[1].scatter(np.arange(0, W)[valid_mask], depth_row[:, k][valid_mask], c=colors, s=sizes, alpha=0.7, label=f'Gaussian {k}' if k == 0 else "")
          
            scatter_plots.append(scatter)
    
    ax[1].set_title('Row-wise Region Gaussian Depth and Weights Visualization')
    ax[1].set_xlabel('Width')
    ax[1].set_ylabel('Depth')
    ax[1].set_xlim(0, W)
    ax[1].legend()
    
    # 컬러바 추가
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax[1], label=f'Normalized Weights (0-{vmax})')
    
    ax[1].grid(True)
    
    # 가로줄 바 추가
    valid_weight_sum = np.sum(np.where(weight_row != -1, weight_row, 0), axis=1)
    bar_cmap = plt.cm.coolwarm  # 빨간색에서 파란색으로 변경
    bar_colors = bar_cmap(valid_weight_sum)
    
    # 가로줄 바 시각화
    ax[2].imshow([bar_colors], aspect='auto', extent=[0, W, 0, 1])  # extent를 0에서 W로 설정
    ax[2].set_xlim(0, W)  # x축 범위를 0에서 W로 설정
    ax[2].set_xticks(np.linspace(0, W, num=5))
    ax[2].set_xticklabels(np.linspace(0, W, num=5, dtype=int))
    ax[2].set_title('Sum of Weights')
    ax[2].set_xlabel('Width')
    ax[2].set_yticks([])
    
    # 가로줄 바 컬러바 추가
    sm_bar = plt.cm.ScalarMappable(cmap=bar_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    sm_bar.set_array([])
    fig.colorbar(sm_bar, ax=ax[2], orientation='horizontal', label='Sum of Weights')
    
    plt.tight_layout()
    
    # Figure를 tensor로 변환하여 반환
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = mpimg.imread(buf)
    plt.close(fig)
    return torch.tensor(image).permute(2, 0, 1)  


import cv2
import os

# 입력 이미지 경로와 출력 비디오 경로 설정
# render_path = 'path_to_images'
# output_path = 'path_to_output/renders.mp4'
def frames2video(render_path, output_path):
    # 이미지 파일 리스트 가져오기
    images = [img for img in os.listdir(render_path) if img.endswith(".png")]
    images.sort()  # 파일 이름에 따라 정렬 (이미지 순서 보장)

    # 첫 번째 이미지로부터 프레임 크기 가져오기
    frame = cv2.imread(os.path.join(render_path, images[0]))
    height, width, layers = frame.shape

    # 비디오 작성기 초기화
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
    video = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

    # 모든 이미지를 비디오에 추가
    for image in images:
        frame = cv2.imread(os.path.join(render_path, image))
        video.write(frame)

    # 비디오 작성기 해제
    video.release()
    
# 이미지 파일 리스트 가져오기
# images = [img for img in os.listdir(render_path) if img.endswith(".png")]
# images.sort()  # 파일 이름에 따라 정렬 (이미지 순서 보장)

# # 첫 번째 이미지로부터 프레임 크기 가져오기
# frame = cv2.imread(os.path.join(render_path, images[0]))
# height, width, layers = frame.shape

# # 비디오 작성기 초기화
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정
# video = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))

# # 모든 이미지를 비디오에 추가
# for image in images:
#     frame = cv2.imread(os.path.join(render_path, image))
#     video.write(frame)

# # 비디오 작성기 해제
# video.release()