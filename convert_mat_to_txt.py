import scipy.io
import numpy as np
import os
from glob import glob

src_dir = r'C:\Users\Anay\.gemini\antigravity\scratch\vad\data\avenue\gt_labels\ground_truth_demo\testing_label_mask'
dst_dir = r'C:\Users\Anay\.gemini\antigravity\scratch\vad\data\avenue\gt_txt_labels'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Pattern is X_label.mat, we want XX.txt
mat_files = glob(os.path.join(src_dir, '*_label.mat'))

for mat_path in mat_files:
    filename = os.path.basename(mat_path)
    video_idx = filename.split('_')[0]
    # format to 01.txt, 02.txt ...
    txt_filename = f"{int(video_idx):02d}.txt"
    txt_path = os.path.join(dst_dir, txt_filename)
    
    data = scipy.io.loadmat(mat_path)
    # Common keys in these MATs: 'label' or 'vol' or just the first non-header key
    keys = [k for k in data.keys() if not k.startswith('__')]
    if not keys:
        continue
    
    label_data = data['volLabel']
    
    data = scipy.io.loadmat(mat_path, squeeze_me=True)
    label_data = data['volLabel']
    
    # Iterate through each frame (element in object array)
    binary_labels = []
    if label_data.dtype == object:
        for frame_cell in label_data:
            # If the cell is not empty and has a max > 0, it's abnormal
            if hasattr(frame_cell, 'size') and frame_cell.size > 0:
                is_abnormal = 1 if np.max(frame_cell) > 0 else 0
            else:
                is_abnormal = 0
            binary_labels.append(is_abnormal)
    else:
        # It's already a numeric array
        binary_labels = (label_data.flatten() > 0).astype(int)
        
    np.savetxt(txt_path, binary_labels, fmt='%d')
    print(f"Converted {filename} -> {txt_filename} (Length: {len(binary_labels)})")
