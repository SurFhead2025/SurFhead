import json, os
from pathlib import Path
import argparse
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def swap_and_remap_timestamps(train_data, test_data, val_data, val_camera_index, goto_test_corpus_name, goto_train_corpus_name):
    # 특정 콜퍼스를 스왑하기 위한 프레임 분류
    # train_frames_to_test = [frame for frame in train_data['frames'] if goto_test_corpus_name in frame['file_path']]
    # test_frames_to_train = [frame for frame in test_data['frames'] if goto_train_corpus_name in frame['file_path']]
    
    # assert len(train_frames_to_test) % 15 == 0
    # assert len(test_frames_to_train) % 16 == 0
    
    # 트레인 데이터셋에서 스왑할 프레임 제거
    remaining_test_frames = []
    test_frames_to_train = []
    
    remaining_train_frames = []
    train_frames_to_test = []
    
    val_frames = []
    
    n_of_whole_frames = len(train_data['frames']) + len(test_data['frames']) + len(val_data['frames'])
    for frame in train_data['frames']:
        if goto_test_corpus_name not in frame['file_path']: #! 계속 남아있는애
            if frame['camera_index'] != val_camera_index:
                remaining_train_frames.append(frame)
            else:
                print("Impossible")
                assert False
                val_frames.append(frame)
                # print("Impossible")
                # assert False
            if frame['camera_index'] == 0:
                for i in val_data['frames']:
                    if i['flame_param_path'] == frame['flame_param_path']:
                        val_frames.append(i)
                        break
                    
        else:#! 테스트로 보내질 애들 보낼떄는 val이랑 같이 16개 만들어서 보내야함.
            
            train_frames_to_test.append(frame)
            if frame['camera_index'] == val_camera_index-1:
                for i in val_data['frames']:
                    if i['flame_param_path'] == frame['flame_param_path']:
                        train_frames_to_test.append(i)
                        # print(1)
                        break
                        
    # assert len(train_data['frames']) == len(remaining_train_frames) + len(train_frames_to_test) + len(val_frames)
    # breakpoint()
    for frame in test_data['frames']:
        if goto_train_corpus_name not in frame['file_path']: #! 아마 없음.
            print("Impossible")
            assert False
            remaining_test_frames.append(frame)
        else:#! 트레인으로 보내질 애들 train val로 쪼개서 보내야함
            if frame['camera_index'] != val_camera_index:
                
                test_frames_to_train.append(frame)
            else:
                val_frames.append(frame)
    # for frame in val_data['frames']:
    #     if goto_test_corpus_name in frame['file_path']:a
    #         assert frame['camera_index'] == val_camera_index
    #         test_frames_to_swap.append(frame)
    # remaining_train_frames = [frame for frame in train_data['frames'] if train_corpus_name not in frame['file_path'] and frame['camera_index'] != val_camera_index]
    # # 테스트 데이터셋에서 스왑할 프레임 제거
    # remaining_test_frames = [frame for frame in test_data['frames'] if test_corpus_name not in frame['file_path']]

    # 스왑된 프레임 추가
    assert len(remaining_test_frames) == 0
    remaining_train_frames.extend(test_frames_to_train)
    remaining_test_frames.extend(train_frames_to_test)
    # breakpoint()
    assert n_of_whole_frames == len(remaining_train_frames) + len(remaining_test_frames) + len(val_frames)
    
    # 데이터셋의 프레임 수 확인
    
    if len(remaining_train_frames) % 15 != 0:
        raise ValueError("Train dataset frame count is not divisible by 15.")
    if len(remaining_test_frames) % 16 != 0:
        raise ValueError("Test dataset frame count is not divisible by 16.")

    # 벨리데이션 데이터셋 구성
    # val_frames = [frame for frame in train_data['frames'] if frame['camera_index'] == val_camera_index]

    # 트레인 데이터셋 타임스탬프 및 카메라 인덱스 재구성 (15개의 카메라)
    num_train_timesteps = len(remaining_train_frames) // 15
    for i in range(num_train_timesteps):
        val_frames[i]['timestep_index'] = i
        val_frames[i]['camera_index'] = val_camera_index
        
        for j in range(15):
            index = i * 15 + j
            remaining_train_frames[index]['timestep_index'] = i
            if j >= val_camera_index:
                remaining_train_frames[index]['camera_index'] = j + 1
            else:
                remaining_train_frames[index]['camera_index'] = j
           
    for i in range(len(val_frames)):
        val_frames[i]['timestep_index'] = i
        val_frames[i]['camera_index'] = val_camera_index
    
    # 테스트 데이터셋 타임스탬프 및 카메라 인덱스 재구성 (16개의 카메라)
    num_test_timesteps = len(remaining_test_frames) // 16
    # for i in range(num_test_timesteps):
    
    for i in range(0, num_test_timesteps):
        for j in range(16):
            index = i * 16 + j
            remaining_test_frames[index]['timestep_index'] = num_train_timesteps+ i
            remaining_test_frames[index]['camera_index'] = j
    # breakpoint()
    # 벨리데이션 데이터셋 타임스탬프 재구성
    # num_val_timesteps = len(val_frames) // 1
    # for i in range(num_val_timesteps):
    #     val_frames[i]['timestep_index'] = i
    #     val_frames[i]['camera_index'] = val_camera_index

    # 데이터 업데이트
    # train_data['frames'] = remaining_train_frames
    # test_data['frames'] = remaining_test_frames
    # train_timesteps = len(remaining_train_frames) // 15
    assert num_train_timesteps == len(val_frames)
    
    val_data_new = {
        "frames": val_frames,
        "cx": val_data["cx"],
        "cy": val_data["cy"],
        "fl_x": val_data["fl_x"],
        "fl_y": val_data["fl_y"],
        "h": val_data["h"],
        "w": val_data["w"],
        "camera_angle_x": val_data["camera_angle_x"],
        "camera_angle_y": val_data["camera_angle_y"],
        "camera_indices": [val_camera_index],
        "timestep_indices": np.arange(num_train_timesteps).tolist()
        
    }
    test_data_new = {
        "frames": remaining_test_frames,
        "cx": test_data["cx"],
        "cy": test_data["cy"],
        "fl_x": test_data["fl_x"],
        "fl_y": test_data["fl_y"],
        "h": test_data["h"],
        "w": test_data["w"],
        "camera_angle_x": test_data["camera_angle_x"],
        "camera_angle_y": test_data["camera_angle_y"],
        "camera_indices": np.arange(16).tolist(),
        "timestep_indices": np.arange(num_train_timesteps, num_train_timesteps + num_test_timesteps).tolist()
        
    }
    train_data_new = {
        "frames": remaining_train_frames,
        "cx": train_data["cx"],
        "cy": train_data["cy"],
        "fl_x": train_data["fl_x"],
        "fl_y": train_data["fl_y"],
        "h": train_data["h"],
        "w": train_data["w"],
        "camera_angle_x": train_data["camera_angle_x"],
        "camera_angle_y": train_data["camera_angle_y"],
        "camera_indices": np.arange(16).tolist().pop(val_camera_index),
        "timestep_indices": np.arange(num_train_timesteps).tolist()
        
    }
    return train_data_new, test_data_new, val_data_new

def main(header, train_file_path, test_file_path, val_file_path, goto_test_corpus_name, goto_train_corpus_name, val_camera_index):
    train_data = load_json(train_file_path)
    test_data = load_json(test_file_path)
    val_data = load_json(val_file_path)
    updated_train_data, updated_test_data, updated_val_data = swap_and_remap_timestamps(train_data, test_data, val_data, val_camera_index, goto_test_corpus_name, goto_train_corpus_name)
    
    real_header = '/'.join(header.split('/')[:-1])
    dir_orig = header.split('/')[-1]
    # breakpoint()
    id = dir_orig.split('_')[1]
    dir_new = f'JSL_{id}_{goto_test_corpus_name}_as_test_{dir_orig}'
    
    output_train_file_path = os.path.join(real_header, dir_new, 'transforms_train.json')
    output_test_file_path = os.path.join(real_header, dir_new, 'transforms_test.json')
    output_val_file_path = os.path.join(real_header, dir_new, 'transforms_val.json')
    
    copy_from_flame_path = os.path.join(real_header, dir_orig, 'canonical_flame_param.npz')
    copy_to_flame_path  = os.path.join(real_header, dir_new, 'canonical_flame_param.npz')
    
    #! copy file
    os.makedirs(os.path.join(real_header, dir_new), exist_ok=True)
    import shutil
    shutil.copy(copy_from_flame_path, copy_to_flame_path)
    
    
    
    save_json(updated_train_data, output_train_file_path)
    save_json(updated_test_data, output_test_file_path)
    save_json(updated_val_data, output_val_file_path)
    
    print(f"New train dataset saved to {output_train_file_path}")
    print(f"New test dataset saved to {output_test_file_path}")
    print(f"New validation dataset saved to {output_val_file_path}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Swap corpora between train and test datasets, remap timestamps, and create validation dataset.')
    # parser.add_argument('--train_file_path', type=Path, required=True, help='Path to the train dataset JSON file.')
    # parser.add_argument('--test_file_path', type=Path, required=True, help='Path to the test dataset JSON file.')
    # parser.add_argument('--train_corpus_name', type=str, required=True, help='Corpus name to be moved from train to test.')
    # parser.add_argument('--test_corpus_name', type=str, required=True, help='Corpus name to be moved from test to train.')
    # parser.add_argument('--val_camera_index', type=int, required=True, help='Camera index to be used for validation dataset.')
   
    
    # args = parser.parse_args() 
    goto_test_corpus_name = 'EMO-1'    #*from train #! may be fixed
    #! 306
    # header = '/home/nas4_dataset/3D/GaussianAvatars/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_test_corpus_name = 'EMO-1' #*from train
    # goto_train_corpus_name = 'EXP-2' #*from test
    
    #! 264
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_264_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_test_corpus_name = 'EMO-1' #*from train #! may be fixed
    # goto_train_corpus_name = 'EXP-9' #*from test

    #! 165
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_165_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EMO-4' #*from test


    #! 104
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EXP-2' #*from test  

    #! 253
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EMO-4' #*from test  

    #! 218
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_218_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EXP-9' #*from test  


    # #! 175
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_175_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EXP-5' #*from test 
    
    
    # #! 074
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EMO-4' #*from test 
    
    # #! 210
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_210_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EXP-5' #*from test 
    
    # #! 238
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_238_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EMO-4' #*from test 
    
    #! 302
    # header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    # goto_train_corpus_name = 'EMO-2' #*from test 
    
    #! 304
    header = '/home/nas4_dataset/3D/GaussianAvatars/cluster/ikarus/sqian/project/dynamic-head-avatars/code/multi-view-head-tracker/export/UNION10_304_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine'
    goto_train_corpus_name = 'EXP-2' #*from test 

    #! 140 is not needed



    
    train_file_path = os.path.join(header, 'transforms_train.json')
    test_file_path = os.path.join(header, 'transforms_test.json')
    val_file_path = os.path.join(header, 'transforms_val.json')
    
    val_camera_index = 8 #! may be fixed
    # header_wo_dir = '/'.join(header.split('/')[:-1])
    main(header, train_file_path, test_file_path, val_file_path, goto_test_corpus_name, goto_train_corpus_name, val_camera_index)
