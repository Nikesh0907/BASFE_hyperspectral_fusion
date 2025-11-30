import os, random, numpy as np, time, gc
from tqdm.auto import tqdm
from scipy.io import savemat
from .utils import load_first_cube, load_rgb_any


def prepare_test_scene(lr_path, msi_path, config):
    lr_cube,_ = load_first_cube(lr_path)
    msi_cube = load_rgb_any(msi_path)
    if config['MSI_BANDS_SELECT']:
        msi_cube = msi_cube[:,:,config['MSI_BANDS_SELECT']]
    up_lr = None
    # upsample LR to MSI spatial size
    up_lr = __import__('cv2').resize(lr_cube, (msi_cube.shape[1], msi_cube.shape[0]), interpolation=__import__('cv2').INTER_CUBIC)
    return up_lr, msi_cube


def reconstruct_test(model, test_lr_list, test_msi_list, config, hsi_bands, msi_bands):
    assert len(test_lr_list)==len(test_msi_list), 'Mismatch in test list counts'
    hrsize = config['PATCH_HR_SIZE']; EDGE = config['EDGE_OVERLAP']
    strider = hrsize - 2*EDGE
    if strider <= 0:
        raise ValueError(f'Computed strider={strider} <= 0. Lower EDGE_OVERLAP or increase PATCH_HR_SIZE.')
    stride_mult = max(1, int(config.get('RECON_STRIDE_MULT',1)))
    if stride_mult>1:
        strider *= stride_mult
    BASE_PRED_PIXEL_BUDGET = int(config.get('PRED_PIXEL_BUDGET',2048))
    recon_patch_cap = config.get('RECON_MAX_PATCHES')
    reconstructed = {}
    loop = tqdm(list(zip(test_lr_list, test_msi_list)), desc='Scenes(test)', leave=True)
    for lr_path, msi_path in loop:
        scene_name = os.path.splitext(os.path.basename(lr_path))[0]
        scene_start = time.time()
        up_lr, hr_msi = prepare_test_scene(lr_path, msi_path, config)
        H,W,_ = hr_msi.shape
        ii = np.arange(0, max(1, H - hrsize + 1), strider)
        jj = np.arange(0, max(1, W - hrsize + 1), strider)
        if len(ii)==0: ii = np.array([0])
        if len(jj)==0: jj = np.array([0])
        if ii[-1] + hrsize > H: ii[-1] = H - hrsize
        if jj[-1] + hrsize > W: jj[-1] = W - hrsize
        ii = np.unique(ii); jj = np.unique(jj)
        coords = [(i,j) for i in ii for j in jj]
        grid_total = len(coords)
        if recon_patch_cap and grid_total > recon_patch_cap:
            random.shuffle(coords); coords = coords[:recon_patch_cap]
        mr_patches=[]; lr_patches=[]
        for (i,j) in coords:
            mr_patches.append(hr_msi[i:i+hrsize, j:j+hrsize, :])
            lr_patches.append(up_lr[i:i+hrsize, j:j+hrsize, :])
        mrdatainput = np.stack(mr_patches, axis=0)
        lrdatainput = np.stack(lr_patches, axis=0)
        pred_pixels_per_patch = hrsize * hrsize
        PRED_BATCH = max(1, BASE_PRED_PIXEL_BUDGET // pred_pixels_per_patch)
        PRED_BATCH = min(PRED_BATCH, mrdatainput.shape[0])
        preds_list=[]
        for start in range(0, mrdatainput.shape[0], PRED_BATCH):
            end = min(start+PRED_BATCH, mrdatainput.shape[0])
            attempt_ok=False; local_pred_batch=end-start
            while not attempt_ok:
                try:
                    batch_preds = model.predict({'msi_input': mrdatainput[start:start+local_pred_batch], 'lr_input': lrdatainput[start:start+local_pred_batch]}, verbose=0)
                    attempt_ok=True; preds_list.append(batch_preds)
                except Exception as e:
                    if 'ResourceExhaustedError' in str(type(e)) and local_pred_batch>1:
                        local_pred_batch = max(1, local_pred_batch//2)
                        continue
                    raise
        preds = np.concatenate(preds_list, axis=0); del preds_list; gc.collect()
        out_cube = np.zeros((H,W,hsi_bands), dtype=np.float32)
        for idx,(i,j) in enumerate(coords):
            out_cube[i:i+hrsize, j:j+hrsize, :] = preds[idx]
        if EDGE>0 and stride_mult==1 and (not recon_patch_cap or len(coords)==grid_total):
            for idx,(i,j) in enumerate(coords):
                out_cube[i+EDGE:i+hrsize-EDGE, j+EDGE:j+hrsize-EDGE, :] = preds[idx, EDGE:-EDGE, EDGE:-EDGE, :]
        reconstructed[scene_name]=out_cube
        savemat(os.path.join(config['RESULTS_DIR'], f'{scene_name}_reconst.mat'), {f'reconst_{scene_name}': out_cube})
        print(f'Reconstructed {scene_name}: shape {out_cube.shape} time {(time.time()-scene_start):.2f}s patches={len(coords)}')
    return reconstructed
