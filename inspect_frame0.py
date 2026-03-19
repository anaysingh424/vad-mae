import scipy.io
import numpy as np

mat_path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat"
mat = scipy.io.loadmat(mat_path)
vol = mat['vol']
print(f"Shape: {vol.shape}, Dtype: {vol.dtype}")
frame0 = vol[:, :, 0]
print(f"Frame 0 sum: {frame0.sum()}")
print(f"Frame 0 unique: {np.unique(frame0)}")
print(f"Frame 0 mean: {frame0.mean()}")
