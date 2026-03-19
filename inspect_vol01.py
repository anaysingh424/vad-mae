import scipy.io
import numpy as np

path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol01.mat"
mat = scipy.io.loadmat(path)
print("Keys:", mat.keys())
for k in mat.keys():
    if not k.startswith('__'):
        vol = mat[k]
        print(f"Key: {k}, Shape: {vol.shape}, Dtype: {vol.dtype}")
        if vol.dtype == object:
            print("Object array details:")
            for j in range(min(5, vol.shape[1])):
                cell = vol[0, j]
                print(f"  Cell {j} sum: {cell.sum()}, Unique: {np.unique(cell)}")
        else:
            print(f"Min: {vol.min()}, Max: {vol.max()}, Unique: {np.unique(vol)[:10]}")
