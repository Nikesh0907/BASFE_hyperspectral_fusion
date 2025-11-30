import numpy as np


def rmse_psnr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    z = x.shape
    n = z[0] * z[1]
    temp = np.sum(np.sum((x - y) * (x - y), axis=0), axis=0) / n
    rmse_per_band = np.sqrt(temp)
    rmse_total = np.sqrt(np.sum(temp) / z[2])
    psnr = 10 * np.log10(1.0 / (rmse_total ** 2 + 1e-12))
    return rmse_total, psnr, rmse_per_band


def sam(x: np.ndarray, y: np.ndarray) -> float:
    num = np.sum(x * y, axis=2)
    den = np.sqrt(np.sum(x * x, axis=2) * np.sum(y * y, axis=2)) + 1e-12
    ang = np.arccos(np.clip(num / den, -1.0, 1.0))
    return float(np.mean(ang) * 180.0 / np.pi)


def ergas(x: np.ndarray, y: np.ndarray, scale: int) -> float:
    z = x.shape
    n = z[0] * z[1]
    mean_y = np.sum(np.sum(y, axis=0), axis=0) / n
    rmse_total, _, rmse_per_band = rmse_psnr(x, y)
    return float(100.0 / scale * np.sqrt(np.sum((rmse_per_band / (mean_y + 1e-12)) ** 2) / z[2]))


def mssim_cc(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = x.shape
    n = z[0] * z[1]
    mean_x = np.sum(np.sum(x, axis=0), axis=0) / n
    mean_y = np.sum(np.sum(y, axis=0), axis=0) / n
    c1 = 1e-4
    c2 = 1e-4
    ssim = np.zeros((z[2]))
    cc = np.zeros((z[2]))
    for i in range(z[2]):
        sig2_x = np.mean(x[:, :, i] ** 2) - mean_x[i] ** 2
        sig2_y = np.mean(y[:, :, i] ** 2) - mean_y[i] ** 2
        sigma_xy = np.mean(x[:, :, i] * y[:, :, i]) - mean_x[i] * mean_y[i]
        ssim[i] = ((2 * mean_x[i] * mean_y[i] + c1) * (2 * sigma_xy + c2)) / (
            (mean_x[i] ** 2 + mean_y[i] ** 2 + c1) * (sig2_x + sig2_y + c2)
        )
        cc[i] = np.sum((x[:, :, i] - mean_x[i]) * (y[:, :, i] - mean_y[i])) / (
            np.sqrt(np.sum((x[:, :, i] - mean_x[i]) ** 2) * np.sum((y[:, :, i] - mean_y[i]) ** 2) + 1e-12)
        )
    return float(np.mean(ssim)), float(np.mean(cc))
import os, json, numpy as np
import cv2 as cv
from math import log10
from tqdm.auto import tqdm
from .utils import load_first_cube, list_mats


def _metrics(pred, gt, scale):
    assert pred.shape == gt.shape
    H,W,L = pred.shape
    n = H*W
    temp = np.sum(np.sum((pred-gt)*(pred-gt),axis=0),axis=0)/n
    rmse_per_band = np.sqrt(temp)
    rmse_total = np.sqrt(np.sum(temp)/L)
    psnr = 10*log10(1.0/(rmse_total**2 + 1e-12))
    num = np.sum(pred*gt,axis=2)
    den = np.sqrt(np.sum(pred*pred,axis=2)*np.sum(gt*gt,axis=2))+1e-12
    sam = np.mean(np.arccos(np.clip(num/den, -1, 1))) * 180/np.pi
    mean_gt = np.sum(np.sum(gt,axis=0),axis=0)/n
    ergas = 100/scale*np.sqrt(np.sum((rmse_per_band / (mean_gt+1e-12))**2)/L)
    c1=.0001; c2=.0001
    ssim=[]
    mean_p = np.sum(np.sum(pred,axis=0),axis=0)/n
    cc=[]
    for i in range(L):
        sigma2_p = np.mean(pred[:,:,i]**2)-mean_p[i]**2
        sigma2_g = np.mean(gt[:,:,i]**2)-mean_gt[i]**2
        sigma_pg = np.mean(pred[:,:,i]*gt[:,:,i]) - mean_p[i]*mean_gt[i]
        ssim_i = ((2*mean_p[i]*mean_gt[i]+c1)*(2*sigma_pg+c2))/((mean_p[i]**2+mean_gt[i]**2+c1)*(sigma2_p+sigma2_g+c2))
        ssim.append(ssim_i)
        cc_num = np.sum((pred[:,:,i]-mean_p[i])*(gt[:,:,i]-mean_gt[i]))
        cc_den = np.sqrt(np.sum((pred[:,:,i]-mean_p[i])**2)*np.sum((gt[:,:,i]-mean_gt[i])**2))+1e-12
        cc.append(cc_num/cc_den)
    return {
        'RMSE': float(rmse_total), 'PSNR': float(psnr), 'SAM_deg': float(sam), 'ERGAS': float(ergas),
        'MSSIM': float(np.mean(ssim)), 'CC': float(np.mean(cc))
    }


def compute_metrics(reconstructed, config, scale):
    # If real GT is not enabled but pseudo-GT is requested, compute that directly
    if not config.get('USE_GT') and config.get('PSEUDO_GT_TEST_HSI'):
        print('GT not enabled; using pseudo-GT (upsampled test LR-HSI).')
        return _compute_pseudo_gt_metrics(reconstructed, config, scale)
    if not config.get('USE_GT'):
        print('GT not enabled; skipping metrics.')
        return {}
    gt_dir_rel = config.get('TEST_GT_HR_HSI_DIR')
    if config.get('USE_GT') and not gt_dir_rel:
        gt_dir_rel = os.path.join(config['TEST_DIR'], config.get('GT_SUBDIR','GT_HR'))
    # Support absolute path override; else treat as relative to ROOT_DIR
    if gt_dir_rel:
        gt_dir_full = gt_dir_rel if os.path.isabs(gt_dir_rel) else os.path.join(config['ROOT_DIR'], gt_dir_rel)
    else:
        gt_dir_full = None
    if not (gt_dir_full and os.path.isdir(gt_dir_full)):
        if config.get('PSEUDO_GT_TEST_HSI'):
            print('Real GT directory missing; using pseudo-GT (upsampled test LR-HSI).')
            return _compute_pseudo_gt_metrics(reconstructed, config, scale)
        print('GT directory missing; skipping metrics.')
        return {}
    test_bases = config.get('TEST_BASENAMES')
    gt_map = {}
    # Primary mapping: from Test.txt basenames
    if test_bases:
        for base in test_bases:
            cand = os.path.join(gt_dir_full, base + '.mat')
            if os.path.isfile(cand): gt_map[base]=cand
    # Fallback: list all mats if none matched
    if not gt_map:
        for f in list_mats(gt_dir_full):
            name = os.path.splitext(os.path.basename(f))[0]
            gt_map[name] = f
    results={}
    for scene in tqdm(reconstructed.keys(), desc='Metrics', leave=True):
        if scene in gt_map:
            gt_cube,_ = load_first_cube(gt_map[scene])
            H = min(gt_cube.shape[0], reconstructed[scene].shape[0])
            W = min(gt_cube.shape[1], reconstructed[scene].shape[1])
            C = min(gt_cube.shape[2], reconstructed[scene].shape[2])
            gt_crop = gt_cube[:H,:W,:C]; pred_crop = reconstructed[scene][:H,:W,:C]
            results[scene] = _metrics(pred_crop, gt_crop, scale)
        else:
            # Try loose match (lowercase, replace spaces/underscores)
            key_loose = scene.lower().replace(' ','_')
            match = None
            for k in gt_map.keys():
                if k.lower().replace(' ','_') == key_loose:
                    match = k; break
            if match:
                gt_cube,_ = load_first_cube(gt_map[match])
                H = min(gt_cube.shape[0], reconstructed[scene].shape[0])
                W = min(gt_cube.shape[1], reconstructed[scene].shape[1])
                C = min(gt_cube.shape[2], reconstructed[scene].shape[2])
                gt_crop = gt_cube[:H,:W,:C]; pred_crop = reconstructed[scene][:H,:W,:C]
                results[scene] = _metrics(pred_crop, gt_crop, scale)
            else:
                print(f'GT not found for scene {scene}')
    if results:
        out_path = os.path.join(config['RESULTS_DIR'], 'metrics.json')
        os.makedirs(config['RESULTS_DIR'], exist_ok=True)
        with open(out_path,'w') as f: json.dump(results, f, indent=2)
        print('Saved metrics.json')
    return results

def _compute_pseudo_gt_metrics(reconstructed, config, scale):
    test_lr_files = config.get('TEST_LR_HSI_FILES', [])
    if not test_lr_files:
        print('Pseudo-GT: no test LR-HSI files available.')
        return {}
    from .utils import load_first_cube
    lr_map = {os.path.splitext(os.path.basename(p))[0]: p for p in test_lr_files}
    results = {}
    for scene in reconstructed.keys():
        key = scene if scene in lr_map else None
        if not key:
            scene_norm = scene.lower().replace(' ','_')
            for k in lr_map.keys():
                if k.lower().replace(' ','_') == scene_norm:
                    key = k; break
        if not key:
            print(f'Pseudo-GT: LR-HSI not found for {scene}')
            continue
        try:
            lr_cube,_ = load_first_cube(lr_map[key])
            H,W,Cp = reconstructed[scene].shape
            up_lr = cv.resize(lr_cube, (W, H), interpolation=cv.INTER_CUBIC)
            C_use = min(Cp, up_lr.shape[2])
            pred_crop = reconstructed[scene][:,:,:C_use]
            up_lr_crop = up_lr[:,:,:C_use]
            results[scene] = _metrics(pred_crop, up_lr_crop, scale)
        except Exception as e:
            print(f'Pseudo-GT metrics failed for {scene}:', e)
    if results:
        out_path = os.path.join(config['RESULTS_DIR'], 'metrics_pseudo_gt.json')
        os.makedirs(config['RESULTS_DIR'], exist_ok=True)
        with open(out_path,'w') as f: json.dump(results, f, indent=2)
        print('Saved metrics_pseudo_gt.json')
    return results
