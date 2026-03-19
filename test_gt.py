import scipy.io
import numpy as np
import os

mat_path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat"
mat = scipy.io.loadmat(mat_path)
print(f"Keys: {mat.keys()}")
vol = mat.get('volLabel', mat.get('vol', None))
if vol is None:
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    vol = mat[key]
    print(f"Using key: {key}")

print(f"Shape: {vol.shape}, Dtype: {vol.dtype}")
if vol.ndim == 3:
    labels = (vol.sum(axis=(0, 1)) > 0).astype(np.int32)
elif vol.dtype == object:
    labels_list = []
    for j in range(vol.shape[1]):
        cell = vol[0, j]
        labels_list.append(1 if cell.sum() > 0 else 0)
    labels = np.array(labels_list, dtype=np.int32)
else:
    labels = (vol.reshape(-1) > 0).astype(np.int32)

print(f"First 50 labels: {labels[:50]}")
print(f"Total frames: {len(labels)}, Anomalous: {labels.sum()}")
