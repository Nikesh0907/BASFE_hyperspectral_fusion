import os, math, random, numpy as np
import cv2 as cv  # Added to support resize operations (was missing, caused NameError)
from tqdm.auto import tqdm
from .utils import load_first_cube, load_rgb_any, read_list, list_mats

def resolve_dir(root, candidates, required=True):
    for rel in candidates:
        full = os.path.join(root, rel)
        if os.path.isdir(full):
            return rel
    if required:
        raise FileNotFoundError(f"None of candidates exist under {root}: {candidates}")
    return None


def discover_files(config):
    mode = config['DATASET_MODE']
    root = config['ROOT_DIR']
    if mode == 'CAVE_MAT_DIRS':
        TRAIN_HR_HSI_DIR = resolve_dir(root, config['TRAIN_HR_HSI_DIR_CAND'])
        TRAIN_LR_HSI_DIR = resolve_dir(root, config['TRAIN_LR_HSI_DIR_CAND'])
        TRAIN_HR_MSI_DIR = resolve_dir(root, config['TRAIN_HR_MSI_DIR_CAND'])
        TEST_LR_HSI_DIR  = resolve_dir(root, config['TEST_LR_HSI_DIR_CAND'])
        TEST_HR_MSI_DIR  = resolve_dir(root, config['TEST_HR_MSI_DIR_CAND'])
        train_hr_hsi_files = list_mats(os.path.join(root, TRAIN_HR_HSI_DIR))
        train_lr_hsi_files = list_mats(os.path.join(root, TRAIN_LR_HSI_DIR))
        train_hr_msi_files = list_mats(os.path.join(root, TRAIN_HR_MSI_DIR))
        test_lr_hsi_files  = list_mats(os.path.join(root, TEST_LR_HSI_DIR))
        test_hr_msi_files  = list_mats(os.path.join(root, TEST_HR_MSI_DIR))
    else:
        train_root = os.path.join(root, config['TRAIN_DIR'])
        test_root  = os.path.join(root, config['TEST_DIR'])
        for p in [train_root, test_root]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f'Dataset dir missing: {p}')
        train_txt = os.path.join(train_root, config['TRAIN_TXT'])
        test_txt  = os.path.join(test_root, config['TEST_TXT'])
        train_list = read_list(train_txt)
        test_list  = read_list(test_txt)
        config['TEST_BASENAMES'] = test_list
        hsi_tr_dir = os.path.join(train_root, config['HSI_SUBDIR'])
        rgb_tr_dir = os.path.join(train_root, config['RGB_SUBDIR'])
        hsi_te_dir = os.path.join(test_root,  config['HSI_SUBDIR'])
        rgb_te_dir = os.path.join(test_root,  config['RGB_SUBDIR'])
        mat_exts = ['.mat']; rgb_exts = ['.mat','.png','.jpg','.jpeg','.tif']
        def find_file(base, folder, exts):
            for e in exts:
                cand = os.path.join(folder, base+e)
                if os.path.isfile(cand): return cand
            cand2 = os.path.join(folder, base)
            if os.path.isfile(cand2): return cand2
            raise FileNotFoundError(f'File for {base} not found in {folder}')
        train_hr_hsi_files = [find_file(b, hsi_tr_dir, mat_exts) for b in train_list]
        train_hr_msi_files = [find_file(b, rgb_tr_dir, rgb_exts) for b in train_list]
        train_lr_hsi_files = train_hr_hsi_files  # derived LR
        test_lr_hsi_files  = [find_file(b, hsi_te_dir, mat_exts) for b in test_list]
        test_hr_msi_files  = [find_file(b, rgb_te_dir, rgb_exts) for b in test_list]
        config['DERIVE_LR_FROM_HR'] = True
        if config.get('USE_GT') and not config.get('TEST_GT_HR_HSI_DIR'):
            config['TEST_GT_HR_HSI_DIR'] = os.path.join(config['TEST_DIR'], config.get('GT_SUBDIR','GT_HR'))
    if config['MAX_TRAIN_SCENES']:
        train_hr_hsi_files = train_hr_hsi_files[:config['MAX_TRAIN_SCENES']]
        train_lr_hsi_files = train_lr_hsi_files[:config['MAX_TRAIN_SCENES']]
        train_hr_msi_files = train_hr_msi_files[:config['MAX_TRAIN_SCENES']]
    if config['MAX_TEST_SCENES']:
        test_lr_hsi_files  = test_lr_hsi_files[:config['MAX_TEST_SCENES']]
        test_hr_msi_files  = test_hr_msi_files[:config['MAX_TEST_SCENES']]
    return train_hr_hsi_files, train_lr_hsi_files, train_hr_msi_files, test_lr_hsi_files, test_hr_msi_files


def infer_scale(hr_sample, msi_sample):
    scale_x = max(1, int(round(msi_sample.shape[1] / hr_sample.shape[1])))
    scale_y = max(1, int(round(msi_sample.shape[0] / hr_sample.shape[0])))
    scale = int((scale_x + scale_y)/2)
    return max(1, scale)


def build_patches(config, train_hr_hsi_files, train_lr_hsi_files, train_hr_msi_files, scale, hsi_bands, msi_bands):
    hrsize = config['PATCH_HR_SIZE']; stride = config['PATCH_STRIDE']
    patches_hr=[]; patches_lr=[]; patches_mr=[]
    cap_bytes = config['PATCH_MEMORY_CAP_MB'] * 1_000_000
    subsample = max(1, config['SUBSAMPLE_PATCH_RATE'])
    est_patch_bytes=None
    MODE = config['BUILD_PATCHES_MODE']
    rand_per_scene = config['RANDOM_PATCHES_PER_SCENE']
    limit_total = config['TOTAL_PATCHES_LIMIT']
    micro_cap = config.get('MICRO_MAX_PATCHES') if config.get('MICRO_DEBUG_MODE') else None
    if micro_cap:
        limit_total = min(limit_total, micro_cap) if limit_total else micro_cap
    scene_iter = tqdm(list(zip(train_hr_hsi_files, train_lr_hsi_files, train_hr_msi_files)), desc=f'Scenes(train) mode={MODE}', leave=True)
    for scene_index,(hr_path, lr_path, msi_path) in enumerate(scene_iter):
        scene_name = os.path.basename(os.path.splitext(hr_path)[0])
        hr_cube,_ = load_first_cube(hr_path)
        if config.get('DERIVE_LR_FROM_HR', False):
            target_lr = (max(1, hr_cube.shape[1]//scale), max(1, hr_cube.shape[0]//scale))
            lr_small = cv.resize(hr_cube, target_lr, interpolation=cv.INTER_AREA)
            lr_cube = lr_small
        else:
            lr_cube,_ = load_first_cube(lr_path)
        msi_cube = load_rgb_any(msi_path)
        if config['MSI_BANDS_SELECT']:
            msi_cube = msi_cube[:,:,config['MSI_BANDS_SELECT']]
        up_lr = cv.resize(lr_cube, (hr_cube.shape[1], hr_cube.shape[0]), interpolation=cv.INTER_CUBIC)
        h,w,_ = hr_cube.shape
        c_before = len(patches_hr)
        if MODE=='grid':
            row_starts = np.arange(0, h - hrsize, stride)
            col_starts = np.arange(0, w - hrsize, stride)
            local_added=0; stop_scene=False
            for i in row_starts:
                for j in col_starts:
                    if (local_added % subsample)==0:
                        patches_hr.append(hr_cube[i:i+hrsize, j:j+hrsize, :])
                        patches_lr.append(up_lr[i:i+hrsize, j:j+hrsize, :])
                        patches_mr.append(msi_cube[i:i+hrsize, j:j+hrsize, :])
                        if est_patch_bytes is None:
                            est_patch_bytes = (patches_hr[-1].nbytes + patches_lr[-1].nbytes + patches_mr[-1].nbytes)
                        total_bytes = len(patches_hr) * est_patch_bytes
                        if cap_bytes and total_bytes > cap_bytes:
                            stop_scene=True; break
                    local_added+=1
                    if limit_total and len(patches_hr)>=limit_total:
                        stop_scene=True; break
                if stop_scene: break
        else:
            max_i = h - hrsize; max_j = w - hrsize
            n_samples = rand_per_scene if rand_per_scene else 0
            if n_samples==0:
                approx_rows = max(1,(h - hrsize)//stride); approx_cols = max(1,(w - hrsize)//stride)
                n_samples = int(approx_rows*approx_cols)
            for k in range(n_samples):
                i = random.randint(0,max_i); j=random.randint(0,max_j)
                if (k % subsample)==0:
                    patches_hr.append(hr_cube[i:i+hrsize, j:j+hrsize, :])
                    patches_lr.append(up_lr[i:i+hrsize, j:j+hrsize, :])
                    patches_mr.append(msi_cube[i:i+hrsize, j:j+hrsize, :])
                    if est_patch_bytes is None:
                        est_patch_bytes = (patches_hr[-1].nbytes + patches_lr[-1].nbytes + patches_mr[-1].nbytes)
                    total_bytes = len(patches_hr) * est_patch_bytes
                    if cap_bytes and total_bytes > cap_bytes: break
                    if limit_total and len(patches_hr) >= limit_total: break
        if (cap_bytes and est_patch_bytes and (len(patches_hr)*est_patch_bytes) > cap_bytes) or (limit_total and len(patches_hr)>=limit_total):
            break
    if len(patches_hr)==0:
        raise RuntimeError('No patches collected; adjust configuration.')
    hrdata = np.stack(patches_hr, axis=0)
    lrdata = np.stack(patches_lr, axis=0)
    mrdata = np.stack(patches_mr, axis=0)
    if config['TRUNCATE_TO_CAP'] and est_patch_bytes and (hrdata.nbytes+lrdata.nbytes+mrdata.nbytes) > cap_bytes:
        triplet_bytes = (hrdata[0].nbytes + lrdata[0].nbytes + mrdata[0].nbytes)
        max_samples = int(cap_bytes // triplet_bytes)
        if max_samples < hrdata.shape[0]:
            idx = np.random.permutation(hrdata.shape[0])[:max_samples]
            hrdata = hrdata[idx]; lrdata = lrdata[idx]; mrdata = mrdata[idx]
    return hrdata, lrdata, mrdata
