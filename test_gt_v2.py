import scipy.io  # type: ignore
import numpy as np  # type: ignore
import os

mat_path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat"
mat = scipy.io.loadmat(mat_path)
print(f"Keys: {[k for k in mat.keys() if not k.startswith('__')]}")
for key in [k for k in mat.keys() if not k.startswith('__')]:
    val = mat[key]
    print(f"Key: {key}, Shape: {val.shape}, Dtype: {val.dtype}, Min: {np.min(val)}, Max: {np.max(val)}, Unique: {np.unique(val)}")

vol = mat.get('volLabel', mat.get('vol', None))
if vol is None:
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    vol = mat[key]

if vol.ndim == 3:
    # H x W x T
    frame_sums = vol.sum(axis=(0, 1))
    print(f"Frame sums first 10: {frame_sums[:10]}")
    labels = (frame_sums > 0).astype(np.int32)
else:
    print("Not 3D")
    labels = []

print(f"Total frames: {len(labels)}, Anomalous: {sum(labels)}")
