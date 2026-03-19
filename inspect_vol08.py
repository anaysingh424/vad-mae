import scipy.io
import numpy as np

mat_path = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol\vol08.mat"
mat = scipy.io.loadmat(mat_path)
print(f"Keys: {[k for k in mat.keys() if not k.startswith('__')]}")
for key in [k for k in mat.keys() if not k.startswith('__')]:
    val = mat[key]
    print(f"Key: {key}, Shape: {val.shape}, Dtype: {val.dtype}, Min: {np.min(val)}, Max: {np.max(val)}")
