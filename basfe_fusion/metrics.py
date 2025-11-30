import os, json, numpy as np
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
        print('GT directory missing; skipping metrics.')
        return {}
    test_bases = config.get('TEST_BASENAMES')
    gt_map = {}
    if test_bases:
        for base in test_bases:
            cand = os.path.join(gt_dir_full, base + '.mat')
            if os.path.isfile(cand): gt_map[base]=cand
    else:
        for f in list_mats(gt_dir_full):
            gt_map[os.path.splitext(os.path.basename(f))[0]] = f
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
            print(f'GT not found for scene {scene}')
    if results:
        out_path = os.path.join(config['RESULTS_DIR'], 'metrics.json')
        os.makedirs(config['RESULTS_DIR'], exist_ok=True)
        with open(out_path,'w') as f: json.dump(results, f, indent=2)
        print('Saved metrics.json')
    return results
