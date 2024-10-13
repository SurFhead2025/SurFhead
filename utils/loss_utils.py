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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from utils.image_utils import erode
from pytorch3d.ops import knn_points


def get_effective_rank(scale, temp=1):
    D = (scale*scale)**(1/temp)
    _sum = D.sum(dim=1, keepdim=True)
    pD = D / _sum
    try:
        entropy = -torch.sum(pD*torch.log(pD), dim=1)
        erank = torch.exp(entropy) 
    except Exception as e:
        print(e)
        pass
    return erank

def prior_normal_loss(local_coord, normal, normal_prior):
    # local_coord: (N, 3), normal: (N, 3), normal_prior: (N, 3)
    weight = torch.exp(-torch.sum(local_coord ** 2, dim=-1) / 0.01)
    # weight = torch.exp()
    # loss = torch.mean(torch.abs(torch.sum(normal * normal_prior, dim=-1)))
    # loss = torch.mean(weight * torch.abs(torch.sum(normal * normal_prior, dim=-1)))
    # import pdb; pdb.set_trace()
    loss = torch.mean(weight * (1 - torch.sum(normal * normal_prior, dim=-1)))
    return loss

def sparse_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)

    # breakpoint()
    normal = normal * 2 - 1 #! to -1 to 1 & normalize to 1
    normal_ref = normal_ref * 2 - 1
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

# def laplacian_loss(L, V):
#     #! https://github.com/sbharadwajj/flare/blob/ed15fa10bf0ff06859d41c4fd4002c8fb7ba19e7/flare/losses/geometry.py#L8
    
#     # breakpoint()
#     loss = L.mm(V)
#     loss = loss.norm(dim=1) ** 2
#     return loss.mean()

def laplacian_loss(L, V):
    #! https://github.com/sbharadwajj/flare/blob/ed15fa10bf0ff06859d41c4fd4002c8fb7ba19e7/flare/losses/geometry.py#L8
    
    # breakpoint()
    loss = L.mm(V)
    loss = loss.norm(dim=1) ** 2
    return loss.mean()


def laplacian_loss_U(L, V):
    # L이 sparse tensor인 경우 dense로 변환
    L = L.to_dense()
    
    # Einstein summation notation을 사용하여 loss 계산
    loss = torch.einsum('nm,mab->nab', L, V)
    
    # Frobenius norm을 배치 차원(첫 번째 차원)을 제외한 나머지 차원에 대해 계산
    frobenius_norm = torch.norm(loss, p='fro', dim=(1, 2))
    
    # 각 배치에 대한 Frobenius norm의 평균을 계산하여 스칼라 손실로 변환
    scalar_loss = frobenius_norm.mean()
    
    return scalar_loss

def normals_laplacian_loss(A, N, xyz, searcher):
    #! https://github.com/sbharadwajj/flare/blob/ed15fa10bf0ff06859d41c4fd4002c8fb7ba19e7/flare/losses/geometry.py#L8

    #* ver.1
    breakpoint()
    neighbor_counts = A.sum(dim=1).int()

    # 가장 인접한 이웃이 많은 수를 max_neighbors로 설정
    max_neighbors = neighbor_counts.max().item()
    K = neighbor_counts.min().item()
    # Adjacency matrix를 사용하여 인덱스 생성
    adj_indices = A.nonzero(as_tuple=True)
    row_indices, col_indices = adj_indices

    # 인접 포인트들을 NxMaxNeighbors, 3 형태로 변환
    adj_points = torch.full((A.shape[0], max_neighbors, 3), 10.0).cuda()
    scatter_indices = torch.arange(row_indices.size(0))

    # 각 포인트의 인접 포인트들을 adj_points에 채웁니다
    adj_points[row_indices, scatter_indices % max_neighbors] = xyz[col_indices]

    # KNN을 adj_points에 적용
    _, idx, _ = knn_points(xyz.unsqueeze(1), adj_points, K=K, return_nn=False)
    loss = 1 - torch.abs(torch.sum(N[idx.squeeze(1)] * N.unsqueeze(1).expand(-1, K, -1), dim=-1)) #! N, K

    # return loss.mean(dim=1).mean()
    #* ver.2
    K = 5
    # xyz = xyz.cpu().detach().numpy()
    # searcher.add(xyz) 
    
    # D, idx = searcher.search(xyz, k=K+1) 
    # idx = idx[:, 1:]  # remove self
    _, idx, _ = knn_points(xyz.unsqueeze(1), xyz.unsqueeze(1), K=K, return_nn=False)
    # breakpoint()
    loss = 1 - torch.abs(torch.sum(N[idx.squeeze(1)] * N.unsqueeze(1).expand(-1, K, -1), dim=-1))
    # loss = 1 - torch.abs(torch.sum(N[idx] * N.unsqueeze(1).expand(-1, K, -1), dim=-1))
    return loss.mean(dim=1).mean()
    #  N[idx.squeeze(1)]

    # L_points = L
    # L_points_indices = L_points._indices()
    # L_points_values = L_points._values()

    # # L_points를 각 차원에 대해 복사하여 NxNx3 sparse matrix 생성
    # # 새로운 인덱스 생성
    # expanded_indices_0 = L_points_indices[0].repeat_interleave(3).cuda()
    # expanded_indices_1 = L_points_indices[1].repeat_interleave(3).cuda()
    # expanded_indices_2 = torch.arange(3).repeat(L_points_indices.size(1)).cuda()

    # # 새로운 값 생성
    # expanded_values = L_points_values.repeat_interleave(3) * N[L_points_indices[1]].view(-1, 3).flatten()

    # # sparse_coo_tensor 생성
    # expanded_indices = torch.stack([expanded_indices_0, expanded_indices_1, expanded_indices_2])
    # breakpoint()
    # N_ = L.shape[0]
    # # expaneded_size = torch.zeros(N,N,3).shape
    # L_points_expanded = torch.sparse_coo_tensor(expanded_indices, expanded_values, (N_, N_, 3))#.to_dense(
    
    # loss = 1 - torch.sum(torch.abs(torch.sparse.mm(L_points_expanded, N.T)), dim=1)
    # K = 10
    # xyz = xyz.cpu().detach().numpy()
    # searcher.add(xyz) 
    
    # D, I = searcher.search(xyz, k=K+1) 
    # I = I[:, 1:]  # remove self
    # near_normals = N[I]
    # self_normals = N.unsqueeze(1).expand(-1, K, -1)
    # # breakpoint()
    # cossim = 1 - torch.abs(torch.sum(near_normals * self_normals, dim=-1))
    # loss = torch.mean(cossim)
    # Gather the points according to the index matrix
    # L_normal = N[L.long()]
    # # breakpoint()
    # #! N == N,3
    # #! L_normal == N,N,3
    # loss = 1 - torch.sum(torch.abs(torch.matmul(L_normal,N.T)), dim=1) #! N, N
    # loss = loss.sum()/L.sum()
    # loss = loss.norm(dim=1) ** 2
    return loss#.mean()

def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def hann_window(x, offset, power):
    return ((1 - torch.cos(torch.pi * torch.clamp(x + offset, 0, 1))) / 2) ** power
#! if x==0 and from 90d~180d, clamp -> 0 -> 9
#! if x==0 and from 