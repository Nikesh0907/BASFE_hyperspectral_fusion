import os
import numpy as np
import cv2 as cv
import scipy.io as sio

IGNORE_KEYS = {'__globals__','__header__','__version__'}

def load_first_cube(mat_path):
    mat = sio.loadmat(mat_path)
    for k,v in mat.items():
        if k in IGNORE_KEYS: continue
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] >= 3:
            arr = v.astype(np.float32)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin)/(vmax - vmin)
            return arr, k
    raise ValueError(f'No 3D cube found in {mat_path}')

def list_mats(dir_path):
    files = []
    if os.path.isdir(dir_path):
        for f in sorted(os.listdir(dir_path)):
            if f.lower().endswith('.mat'):
                files.append(os.path.join(dir_path, f))
    return files

def load_rgb_any(path):
    if path.lower().endswith('.mat'):
        cube,_ = load_first_cube(path)
        return cube
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'RGB/MSI file not found or unreadable: {path}')
    img = img.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min())/(img.max() - img.min())
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return img

def read_list(txt_path):
    with open(txt_path,'r') as f:
        return [line.strip() for line in f if line.strip()]