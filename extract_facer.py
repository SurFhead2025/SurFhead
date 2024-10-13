import torch
import facer
import ssl
import os
from PIL import Image
import json
from tqdm import tqdm
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.utils import save_image as si
device = 'cuda' if torch.cuda.is_available() else 'cpu'

 # image: 1 x 3 x h x w

face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/celebm/448', device=device) # optional "farl/celebm/448"
# breakpoint()
whos = [165, 175, 302, 304, 218]   
whos = [175, 302, 304, 218,  '210', '074' ,]#!140, 264, 306,253
whos = [304, 302, 218, 210, '074', 140, 253, 264, 165, 175]
whos = ['306']
import numpy as np
import cv2

#! all semantics
# with torch.inference_mode():
#     # who = 165
#     for who in whos:
#         head_path \
#             = f'/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_{who}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
#         if who == '306':
#             head_path = f'/home/nas4_dataset/3D/GaussianAvatars/UNION10_{who}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
#         # os.makedirs(os.path.join(head_path, 'kpts5'))
#         image_bunch_path = '/'.join(head_path.split('/')[:-1])
#         whole_image_path_list = []
#         # breakpoint()
#         print(f'{who} START!')
#         for i in ['train', 'val', 'test']:
#             meta_path = f'transforms_{i}.json'
            
#             # whole_image_path_list
#             with open(os.path.join(head_path, meta_path)) as asd:
#                 meta = json.load(asd)
                
                
#             frames = meta['frames']
#             for _f in tqdm(frames, desc = f'{who}_{i}'):
#                 frame_name = _f['file_path'].split('../')[-1]
                
#                 load_path = os.path.join(image_bunch_path,frame_name)
#                 # img = np.array(Image.open(load_path))
#                 # faces = app.get(img)
#                 # faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] 
#                 # kpts = faces['kps']
#                 try:
#                     image = facer.hwc2bchw(facer.read_hwc(load_path)).to(device=device) 
#                 except:
#                     breakpoint()
#                 faces = face_detector(image)
#                 save_path = load_path.replace('images','celeba_facer')
#                 os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
#                 try:
#                     faces = face_parser(image, faces)
#                     seg_logits = faces['seg']['logits']
#                     seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
#                     n_classes = seg_probs.size(1)
#                     vis_seg_probs = seg_probs.argmax(dim=1).float()#/n_classes*255
#                     # mask = (vis_seg_probs[0] != 0) & (vis_seg_probs[0] != 18)
#                     # binary_mask = mask.float()
                    
                    
#                     # breakpoint()
                    
#                     # si(binary_mask,save_path)
#                     cv2.imwrite(save_path, vis_seg_probs[0].cpu().numpy().astype(np.uint8))
                    
#                 except:
#                     # save_path = load_path.replace('images','binary_facer')
#                     dummy_mask = torch.ones_like(image).float()/255 #! regarding skin
#                     print(load_path, '  ::  OUT!!!')
#                     si(dummy_mask,save_path)
#                     # breakpoint()
                    


                
#                 # np.save(save_path, kpts)
#         # faces = app.get(img)
#         print(f'Done for {who}!')
        
        
#* binary
with torch.inference_mode():
    # who = 165
    for who in whos:
        head_path \
            = f'/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_{who}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
        if who == '306':
            head_path = f'/home/nas4_dataset/3D/GaussianAvatars/UNION10_{who}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
        # os.makedirs(os.path.join(head_path, 'kpts5'))
        image_bunch_path = '/'.join(head_path.split('/')[:-1])
        whole_image_path_list = []
        # breakpoint()
        print(f'{who} START!')
        for i in ['train', 'val', 'test']:
            meta_path = f'transforms_{i}.json'
            
            # whole_image_path_list
            with open(os.path.join(head_path, meta_path)) as asd:
                meta = json.load(asd)
                
                
            frames = meta['frames']
            for _f in tqdm(frames, desc = f'{who}_{i}'):
                frame_name = _f['file_path'].split('../')[-1]
                
                load_path = os.path.join(image_bunch_path,frame_name)
                # img = np.array(Image.open(load_path))
                # faces = app.get(img)
                # faces = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] 
                # kpts = faces['kps']
                try:
                    image = facer.hwc2bchw(facer.read_hwc(load_path)).to(device=device) 
                except:
                    breakpoint()
                faces = face_detector(image)
                save_path = load_path.replace('images','binary_facer')
                os.makedirs('/'.join(save_path.split('/')[:-1]), exist_ok=True)
                try:
                    faces = face_parser(image, faces)
                    seg_logits = faces['seg']['logits']
                    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
                    n_classes = seg_probs.size(1)
                    vis_seg_probs = seg_probs.argmax(dim=1).float()#/n_classes*255
                    mask = (vis_seg_probs[0] != 0) & (vis_seg_probs[0] != 18)
                    binary_mask = mask.float()
                    
                    
                    # breakpoint()
                    
                    si(binary_mask,save_path)
                except:
                    # save_path = load_path.replace('images','binary_facer')
                    dummy_mask = torch.ones_like(image).float()
                    print(load_path, '  ::  OUT!!!')
                    si(dummy_mask,save_path)
                    # breakpoint()


                
                # np.save(save_path, kpts)
        # faces = app.get(img)
        print(f'Done for {who}!')
        
        
        