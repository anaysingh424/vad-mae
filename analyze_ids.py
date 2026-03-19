import scipy.io
import numpy as np
import os

def analyze_video(mat_path):
    print(f"Analyzing {mat_path}...")
    mat = scipy.io.loadmat(mat_path)
    k = [k for k in mat.keys() if not k.startswith('__')][0]
    vol = mat[k]
    
    unique_per_frame = []
    for f in range(vol.shape[2]):
        u = np.unique(vol[:, :, f])
        # Filter out 0 if present
        u = u[u != 0]
        unique_per_frame.append(set(u))
    
    # Check transitions
    for i in range(1, len(unique_per_frame)):
        if unique_per_frame[i] != unique_per_frame[i-1]:
            print(f"Frame {i:4d}: {sorted(list(unique_per_frame[i-1]))} -> {sorted(list(unique_per_frame[i]))}")
            if i > 500: break 

analyze_video(r'C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat')
