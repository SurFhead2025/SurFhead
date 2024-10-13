from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np

lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
cos = torch.nn.CosineSimilarity(dim=0)

def readImages(method_dir, return_normal=False, image_name=None):
    renders_dir = method_dir / 'renders'
    gt_dir = method_dir / 'gt'
    if return_normal:
        normal_dir = method_dir / 'normal'
        render_normal_dir = method_dir / 'render_tangent_normals'
        mask_dir = method_dir / 'mask'
        
        render = Image.open(renders_dir / image_name)
        gt = Image.open(gt_dir / image_name)
        normal = np.load(normal_dir / image_name.replace('png', 'npy'))
        render_normal = np.load(render_normal_dir / image_name.replace('png', 'npy'))
        mask = Image.open(mask_dir / image_name)
        
        normal = torch.tensor(normal).unsqueeze(0)[:, :3, :, :].cuda()
        render_normal = torch.tensor(render_normal).unsqueeze(0)[:, :3, :, :].cuda()
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        mask = tf.to_tensor(mask).unsqueeze(0)[:, :1, :, :].cuda()
        
        return render, gt, render_normal, normal, mask
    else:
        render = Image.open(renders_dir / image_name)
        gt = Image.open(gt_dir / image_name)
        render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        
        return render, gt

def evaluate(model_paths, use_mask=False):
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
                
            test_dir = Path(scene_dir) / "test"
            return_normal = 'FaceTalk' in scene_dir
            for method in os.listdir(test_dir):
                print("Method:", method)
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                renders_dir = method_dir / 'renders'
                image_names = os.listdir(renders_dir)
                
                ssims = []
                psnrs = []
                lpipss = []
                normal_cossims = []
                
                for idx in tqdm(range(len(image_names)), desc="Metric evaluation progress"):
                    torch.cuda.empty_cache()
                    image_name = image_names[idx]
                    
                    if return_normal:
                        render, gt, render_normal, normal, mask = readImages(method_dir, return_normal, image_name)
                    else:
                        render, gt = readImages(method_dir, return_normal, image_name)
                    
                    ssims.append(ssim(render, gt))
                    psnrs.append(psnr(render, gt))
                    lpipss.append(lpips(render, gt))
                    
                    if return_normal:
                        unit_normal = normal[0]
                        unit_rend_normal = render_normal[0]
                        mask = mask[0]
                        foreground_mask = mask > 0.5
                        foreground_mask = foreground_mask.squeeze(0)
                        foreground_unit_normal = unit_normal[:, foreground_mask]
                        foreground_unit_rend_normal = unit_rend_normal[:, foreground_mask]
                        cossim = torch.mean(cos(foreground_unit_normal, foreground_unit_rend_normal))
                        normal_cossims.append(cossim.item())

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                if return_normal:
                    print(" NORMAL_COSSIM: {:>12.7f}".format(torch.tensor(normal_cossims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})
                if return_normal:
                    full_dict[scene_dir][method].update({'NORMAL_COSSIM': torch.tensor(normal_cossims).mean().item()})
                    per_view_dict[scene_dir][method].update({'NORMAL_COSSIM': {name: lp for lp, name in zip(torch.tensor(normal_cossims).tolist(), image_names)}})
                    
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--use_mask', action='store_true', help='Use mask if this flag is set')
    args = parser.parse_args()
    evaluate(args.model_paths, args.use_mask)
