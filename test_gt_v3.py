import scipy.io
import numpy as np
import os

mat_path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat"
mat = scipy.io.loadmat(mat_path)
vol = mat['vol']
print(f"Shape: {vol.shape}")
# Assuming T is the dimension with 1439
t_axis = np.where(np.array(vol.shape) == 1439)[0][0]
h_axis = 0 if t_axis != 0 else 1
w_axis = 1 if t_axis > 1 else 2

print(f"T axis: {t_axis}, H axis: {h_axis}, W axis: {w_axis}")

# Sum spatially to get frame-level labels
spatial_axes = (h_axis, w_axis)
frame_sums = vol.sum(axis=spatial_axes)
print(f"Frame sums (first 20): {frame_sums[:20]}")
print(f"Max possible sum: {vol.shape[h_axis] * vol.shape[w_axis]}")

labels = (frame_sums > 0).astype(np.int32)
print(f"Total frames: {len(labels)}, Anomalous: {labels.sum()}")
print(f"First 50 labels: {labels[:50]}")
