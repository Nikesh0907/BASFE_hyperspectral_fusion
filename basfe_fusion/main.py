import os, argparse, json, numpy as np, tensorflow as tf
from . import config as cfg
from .dataset import discover_files, infer_scale, build_patches
from .utils import load_first_cube, load_rgb_any
from .model import build_model
from .train import train_model
from .reconstruct import reconstruct_test
from .metrics import compute_metrics


def enable_mixed_precision(conf):
    if conf['MIXED_PRECISION']:
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy(conf['MIXED_PRECISION'])
            mixed_precision.set_global_policy(policy)
            print('Mixed precision enabled:', policy)
        except Exception as e:
            print('Mixed precision failed:', e)


def prepare(conf):
    for d in ['LOG_DIR','RESULTS_DIR','CHECKPOINT_DIR']:
        os.makedirs(conf[d], exist_ok=True)
    np.random.seed(conf['SEED'])
    import random as _r; _r.seed(conf['SEED'])


def run(args):
    conf = cfg.load_config(force_fast=args.fast_test, root_dir=args.root_dir, use_gt=args.use_gt)
    prepare(conf)
    train_hr_hsi_files, train_lr_hsi_files, train_hr_msi_files, test_lr_hsi_files, test_hr_msi_files = discover_files(conf)
    print('Train counts -> HR-HSI', len(train_hr_hsi_files), 'LR-HSI', len(train_lr_hsi_files), 'MSI', len(train_hr_msi_files))
    hr_sample,_ = load_first_cube(train_hr_hsi_files[0])
    msi_sample = load_rgb_any(train_hr_msi_files[0])
    scale = infer_scale(hr_sample, msi_sample)
    H_BANDS = hr_sample.shape[2]
    M_BANDS = msi_sample.shape[2] if conf['MSI_BANDS_SELECT'] is None else len(conf['MSI_BANDS_SELECT'])
    adaptive_info = cfg.autotune_patch(conf, hr_sample)
    if adaptive_info:
        print('Adaptive sizing applied:', adaptive_info)
    cfg.summarize(conf)
    enable_mixed_precision(conf)

    # Mode control
    if args.mode in ('train','full'):
        hrdata, lrdata, mrdata = build_patches(conf, train_hr_hsi_files, train_lr_hsi_files, train_hr_msi_files, scale, H_BANDS, M_BANDS)
        model = build_model(conf['PATCH_HR_SIZE'], M_BANDS, H_BANDS)
        try:
            model.save(conf['SAVE_MODEL_PATH'])
        except Exception:
            pass
        train_model(model, hrdata, lrdata, mrdata, conf)
    else:
        # build model for subsequent steps (weights must exist if training omitted)
        model = build_model(conf['PATCH_HR_SIZE'], M_BANDS, H_BANDS)
        if os.path.isfile(conf['SAVE_MODEL_TRAINED_PATH']):
            model.load_weights(conf['SAVE_MODEL_TRAINED_PATH'])
            print('Loaded trained weights.')
        else:
            print('WARNING: trained model file not found; using random weights.')

    if args.mode in ('reconstruct','full','metrics'):
        reconstructed = reconstruct_test(model, test_lr_hsi_files, test_hr_msi_files, conf, H_BANDS, M_BANDS)
        if args.mode in ('metrics','full'):
            metrics = compute_metrics(reconstructed, conf, scale)
            if metrics:
                print('Metrics summary:', json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description='BASFE Fusion CLI')
    p.add_argument('--mode', choices=['train','reconstruct','metrics','full'], default='full', help='Pipeline step to run')
    p.add_argument('--root-dir', type=str, default=None, help='Override dataset root directory')
    p.add_argument('--fast-test', type=lambda x: str(x).lower() in ('1','true','yes','y'), default=None, help='Enable fast test preset (smoke).')
    p.add_argument('--use-gt', type=lambda x: str(x).lower() in ('1','true','yes','y'), default=None, help='Enable ground truth metrics computation.')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
