import os
import cv2
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from configs.configs import get_configs_avenue
from data.train_dataset import AbnormalDatasetGradientsTrain
from data.test_dataset import AbnormalDatasetGradientsTest

def pack_list_to_npy(file_list, save_path, shape=(160, 320, 3)):
    # Initialize gigantic monolithic file on SSD (0 bytes in true RAM!)
    mem = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.uint8, shape=(len(file_list), shape[0], shape[1], shape[2]))
    for i, path in enumerate(file_list):
        img = cv2.imread(path)
        if img is None:
            print(f"FAILED TO LOAD: {path}")
            continue
        if img.shape[:2] != shape[:2]:
            img = cv2.resize(img, (shape[1], shape[0]))
        if shape[2] == 1 and len(img.shape) == 3:
            img = img[:, :, :1]
        mem[i] = img
        if i % 1000 == 0:
            print(f"Packed {i}/{len(file_list)} into {save_path}...")
    mem.flush()
    print(f"Successfully minted monolithic binary: {save_path}!")

if __name__ == '__main__':
    args = get_configs_avenue()
    args.dataset = "avenue"
    
    print("Generating pure PyTorch logical structures...")
    train_dataset = AbnormalDatasetGradientsTrain(args)
    test_dataset = AbnormalDatasetGradientsTest(args)
    
    print("Executing monolithic compression for TRAIN...")
    pack_list_to_npy(train_dataset.data, 'train_data.npy', shape=(160, 320, 3))
    pack_list_to_npy(train_dataset.abnormal_data, 'train_abnormal.npy', shape=(160, 320, 3))
    pack_list_to_npy(train_dataset.gradients, 'train_grads.npy', shape=(160, 320, 3))
    pack_list_to_npy(train_dataset.masks_abnormal, 'train_masks.npy', shape=(160, 320, 1))
    
    print("Executing monolithic compression for TEST...")
    pack_list_to_npy(test_dataset.data, 'test_data.npy', shape=(160, 320, 3))
    pack_list_to_npy(test_dataset.gradients, 'test_grads.npy', shape=(160, 320, 3))
    
    print("Fully Extracted! System is ready to structurally bypass PyTorch Memory bugs!")
