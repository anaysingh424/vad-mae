"""
Convert Avenue Dataset .mat ground truth files to per-frame label .txt files.
The .mat files contain a cell array 'volLabel' where each cell is a binary volume
(H x W x T) indicating anomalous pixels per frame. We collapse to per-frame labels
by checking if any anomalous pixel exists in a frame.
"""
import os
import scipy.io
import numpy as np

mat_dir = r"C:\Users\Anay\Downloads\Avenue_Dataset\Avenue Dataset\testing_vol"
out_dir = r"C:\Users\Anay\.gemini\antigravity\scratch\Avenue Dataset"

os.makedirs(out_dir, exist_ok=True)

mat_files = sorted(os.listdir(mat_dir))
for i, fname in enumerate(mat_files, 1):
    if not fname.endswith('.mat'):
        continue
    mat = scipy.io.loadmat(os.path.join(mat_dir, fname))
    # volLabel is shape (H,W,T) where T = num frames
    vol = mat.get('volLabel', mat.get('vol', None))
    if vol is None:
        # try first non-private key
        key = [k for k in mat.keys() if not k.startswith('__')][0]
        vol = mat[key]

    # Squeeze out any extra dimensions and collapse spatially  
    if vol.ndim == 3:
        # shape: H x W x T
        labels = (vol.sum(axis=(0, 1)) > 0).astype(np.int32)
    elif vol.dtype == object:
        # cell array from scipy: vol is (1, N) of arrays
        labels_list = []
        for j in range(vol.shape[1]):
            cell = vol[0, j]
            labels_list.append(1 if cell.sum() > 0 else 0)
        labels = np.array(labels_list, dtype=np.int32)
    else:
        labels = (vol.reshape(-1) > 0).astype(np.int32)

    vid_name = str(i).zfill(2)
    out_path = os.path.join(out_dir, f"{vid_name}.txt")
    np.savetxt(out_path, labels, fmt='%d')
    print(f"Saved {out_path} with {len(labels)} frames, {labels.sum()} anomalous")
print("Done")
