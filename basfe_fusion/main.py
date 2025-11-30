import os, argparse, json, numpy as np, tensorflow as tf
from . import config as cfg
from .dataset import discover_files, infer_scale, build_patches
from .utils import load_first_cube, load_rgb_any
from .model import build_model
from .train import train_model
from .reconstruct import reconstruct_test
from .metrics import compute_metrics


def enable_mixed_precision(conf):
    mp = conf.get('MIXED_PRECISION')
    if mp:
        # Allow common shorthand aliases
        alias_map = {
            'fp16': 'mixed_float16',
            'float16': 'mixed_float16',
            'bf16': 'mixed_bfloat16',
            'bfloat16': 'mixed_bfloat16'
        }
        policy_name = alias_map.get(str(mp).lower(), mp)
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy(policy_name)
            mixed_precision.set_global_policy(policy)
            print('Mixed precision enabled:', policy)
        except Exception as e:
            print('Mixed precision failed:', e, '\nRequested:', mp, 'Resolved policy:', policy_name)


def prepare(conf):
    for d in ['LOG_DIR','RESULTS_DIR','CHECKPOINT_DIR']:
        os.makedirs(conf[d], exist_ok=True)
    np.random.seed(conf['SEED'])
    import random as _r; _r.seed(conf['SEED'])


def run(args):
    # Optional noise suppression and XLA control
    if getattr(args, 'quiet', False):
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')  # suppress INFO and WARNING
    if getattr(args, 'disable_xla', False):
        os.environ['TF_XLA_FLAGS'] = '--xla_disable_gpu'
    conf = cfg.load_config(force_fast=args.fast_test, root_dir=args.root_dir, use_gt=args.use_gt)
    if args.gt_dir:
        # Accept absolute or relative; if relative, assume under ROOT_DIR
        conf['TEST_GT_HR_HSI_DIR'] = args.gt_dir
    if args.epochs is not None:
        try:
            conf['EPOCHS'] = int(args.epochs)
            print('Overriding EPOCHS to', conf['EPOCHS'])
        except Exception:
            print('Invalid --epochs value; using default in config.')
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
    p.add_argument('--epochs', type=int, default=None, help='Override number of training epochs')
    p.add_argument('--gt-dir', type=str, default=None, help='Override GT directory for metrics (absolute or relative)')
    p.add_argument('--quiet', action='store_true', help='Reduce TensorFlow/absl log verbosity')
    p.add_argument('--disable-xla', action='store_true', help='Disable XLA to avoid slow operation alarms')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
