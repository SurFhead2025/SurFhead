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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
cos = torch.nn.CosineSimilarity(dim=0) #! 값이 높을수록 좋음
def readImages(method_dir, return_normal=False):
    if return_normal:
        
        renders_dir = method_dir / 'epoch_60' / 'rgb'
        gt_dir = method_dir / 'epoch_60' / 'rgb_gt'
        render_normal_dir = method_dir / 'epoch_60' / 'normal'
        normal_dir = method_dir / 'epoch_60' / 'normal_gt'
        mask_dir = method_dir / 'epoch_60' / 'object_mask'
        
        renders = []
        gts = []
        normals = []
        render_normals = []
        masks = []
        image_names = []
        for fname in os.listdir(renders_dir):
            # breakpoint()
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            # 
            # normal = np.load(normal_dir / fname.replace('png','npy'))
            normal = Image.open(normal_dir / fname)
            render_normal = Image.open(render_normal_dir / fname)
            # render_normal = np.load(render_normal_dir / fname.replace('png','npy'))
            
            mask = Image.open(mask_dir / fname)
            # breakpoint()
            normals.append(tf.to_tensor(normal).unsqueeze(0)[:, :3, :, :].cuda())
            render_normals.append(tf.to_tensor(render_normal).unsqueeze(0)[:, :3, :, :].cuda())
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            masks.append(tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :].cuda())
            
            image_names.append(fname)
        return renders, gts, render_normals, normals, masks, image_names
    else:
        renders = []
        gts = []
        image_names = []
        for fname in os.listdir(renders_dir):
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
        return renders, gts, image_names
    

def evaluate(model_paths, use_mask = False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}
            
                
            test_dir = Path(scene_dir) / "eval"
            if 'FaceTalk' in scene_dir:
                return_normal = True
            else:
                return_normal = False
            for corpus in os.listdir(test_dir):
                print("Corpus:", corpus)

                # full_dict[scene_dir][method] = {}
                # per_view_dict[scene_dir][method] = {}
                # full_dict_polytopeonly[scene_dir][method] = {}
                # per_view_dict_polytopeonly[scene_dir][method] = {}

                corpus_dir = test_dir / corpus
                # gt_dir = method_dir/ "gt"
                # renders_dir = method_dir / "renders"
                if return_normal:
                    renders, gts, render_normals, normals, masks, image_names = readImages(corpus_dir, return_normal)
                else:
                    renders, gts, image_names = readImages(method_dir, return_normal)
                if use_mask:
                    # gts = 
                    breakpoint()
                ssims = []
                psnrs = []
                lpipss = []
                if return_normal:
                    normal_cossims = []
                # breakpoint()

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx]))
                    if return_normal:
                        # breakpoint()
                        unit_normal = normals[idx][0] * 2 - 1
                        unit_rend_normal = render_normals[idx][0] * 2 - 1
                        mask = masks[idx][0]

                        # 전경 영역만 선택
                        foreground_mask = mask > 0.5  # 마스크가 0.5보다 큰 영역을 전경으로 간주
                        foreground_mask = foreground_mask.squeeze(0)
                        # breakpoint()
                        foreground_unit_normal = unit_normal[:, foreground_mask]
                        foreground_unit_rend_normal = unit_rend_normal[:, foreground_mask]

                        # 코사인 유사도 계산
                        cossim = torch.mean(cos(foreground_unit_normal,foreground_unit_rend_normal))
                        normal_cossims.append(cossim.item())  # t


                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                if return_normal:
                    print(" NORMAL_COSSIM: {:>12.7f}".format(torch.tensor(normal_cossims).mean(), ".5"))

                full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                if return_normal:
                    full_dict[scene_dir].update({'NORMAL_COSSIM': torch.tensor(normal_cossims).mean().item()})
                else:
                    full_dict[scene_dir].update({'NORMAL_COSSIM': {name: lp for lp, name in zip(torch.tensor(normal_cossims).tolist(), image_names)}})
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--use_mask', action='store_true', help='Use mask if this flag is set')
    args = parser.parse_args()
    evaluate(args.model_paths, args.use_mask)
