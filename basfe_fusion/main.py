import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .dataset import discover_scene_paths, load_scene, extract_patches, tile_indices
from .model import build_basfe_model
from .io_utils import mat_save
from .metrics import rmse_psnr, sam, ergas, mssim_cc


def enable_quiet_logs():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_gpu():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Use only the first GPU to reduce memory pressure
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass


def train(args):
    enable_quiet_logs()
    configure_gpu()
    scenes = discover_scene_paths(args.root_dir, "Train")
    if not scenes:
        raise RuntimeError("No training scenes found under Train/HSI and Train/RGB")

    hrsize = args.hrsize
    stride = args.stride
    scale = args.scale

    hr_list = []
    lr_list = []
    mr_list = []
    hsi_bands = None
    msi_bands = None

    for s in scenes[: args.max_scenes or len(scenes)]:
        hrhsi, hrmsi, lrhsi_up = load_scene(s["hsi"], s["rgb"], scale=scale)
        if hsi_bands is None:
            hsi_bands = hrhsi.shape[2]
        if msi_bands is None:
            msi_bands = hrmsi.shape[2]
        h, l, m = extract_patches(hrhsi, hrmsi, lrhsi_up, hrsize=hrsize, stride=stride)
        hr_list.append(h)
        lr_list.append(l)
        mr_list.append(m)

    hrdata = np.concatenate(hr_list, axis=0)
    lrdata = np.concatenate(lr_list, axis=0)
    mrdata = np.concatenate(mr_list, axis=0)

    model = build_basfe_model(hrsize=hrsize, hsi_bands=hsi_bands, msi_bands=msi_bands, num_filter=args.num_filter)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=keras.losses.MeanSquaredError())

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        model.save(os.path.join(args.save_dir, "model_untrained.keras"))

    model.fit({"msi_input": mrdata, "lr_input": lrdata}, {"fuse_output": hrdata}, epochs=args.epochs, batch_size=args.batch_size, verbose=1 if not args.quiet else 0)

    if args.save_dir:
        model.save(os.path.join(args.save_dir, "model_trained.keras"))


def reconstruct(args):
    enable_quiet_logs()
    configure_gpu()
    model = keras.models.load_model(args.model_path)
    scenes = discover_scene_paths(args.root_dir, "Test")
    if not scenes:
        raise RuntimeError("No test scenes found under Test/HSI and Test/RGB")

    hrsize = args.hrsize
    edge = args.edge
    scale = args.scale

    out_dir = args.out_dir or os.path.join(args.root_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    for idx, s in enumerate(scenes[: args.max_scenes or len(scenes)]):
        hrhsi, hrmsi, lrhsi_up = load_scene(s["hsi"], s["rgb"], scale=scale)
        H, W, L = hrhsi.shape
        ii, jj = tile_indices(H, W, hrsize, edge)
        num = ii.size * jj.size
        mrdatainput = np.zeros((num, hrsize, hrsize, hrmsi.shape[2]), dtype=np.float32)
        lrdatainput = np.zeros((num, hrsize, hrsize, hrhsi.shape[2]), dtype=np.float32)
        c = 0
        for i in ii:
            for j in jj:
                mrdatainput[c] = hrmsi[i : i + hrsize, j : j + hrsize]
                lrdatainput[c] = lrhsi_up[i : i + hrsize, j : j + hrsize]
                c += 1
        pred = model.predict((mrdatainput, lrdatainput), verbose=0)
        reconst = np.zeros_like(hrhsi)
        c = 0
        for i in ii:
            for j in jj:
                reconst[i : i + hrsize, j : j + hrsize] = pred[c]
                c += 1
        c = 0
        for i in ii:
            for j in jj:
                reconst[i + edge : i + hrsize - edge, j + edge : j + hrsize - edge] = pred[c, edge:-edge, edge:-edge]
                c += 1
        mat_save(os.path.join(out_dir, f"reconst_{idx+1}.mat"), "reconst", reconst)


def compute_metrics(args):
    enable_quiet_logs()
    configure_gpu()
    scenes = discover_scene_paths(args.root_dir, "Test")
    if not scenes:
        raise RuntimeError("No test scenes found under Test/HSI and Test/RGB")

    hrsize = args.hrsize
    edge = args.edge
    scale = args.scale

    model = keras.models.load_model(args.model_path)
    out = {}
    for idx, s in enumerate(scenes[: args.max_scenes or len(scenes)]):
        hrhsi, hrmsi, lrhsi_up = load_scene(s["hsi"], s["rgb"], scale=scale)
        # Reconstruct quickly (same as reconstruct but in-memory)
        H, W, L = hrhsi.shape
        ii, jj = tile_indices(H, W, hrsize, edge)
        num = ii.size * jj.size
        mrdatainput = np.zeros((num, hrsize, hrsize, hrmsi.shape[2]), dtype=np.float32)
        lrdatainput = np.zeros((num, hrsize, hrsize, hrhsi.shape[2]), dtype=np.float32)
        c = 0
        for i in ii:
            for j in jj:
                mrdatainput[c] = hrmsi[i : i + hrsize, j : j + hrsize]
                lrdatainput[c] = lrhsi_up[i : i + hrsize, j : j + hrsize]
                c += 1
        pred = model.predict((mrdatainput, lrdatainput), verbose=0)
        reconst = np.zeros_like(hrhsi)
        c = 0
        for i in ii:
            for j in jj:
                reconst[i : i + hrsize, j : j + hrsize] = pred[c]
                c += 1
        c = 0
        for i in ii:
            for j in jj:
                reconst[i + edge : i + hrsize - edge, j + edge : j + hrsize - edge] = pred[c, edge:-edge, edge:-edge]
                c += 1

        # Metrics: if GT missing or user requests pseudo, compare to lrhsi_up
        gt = hrhsi if not args.pseudo_gt else lrhsi_up
        rmse_total, psnr, _ = rmse_psnr(reconst, gt)
        sam_val = sam(reconst, gt)
        ergas_val = ergas(reconst, gt, scale=scale)
        mssim_val, cc_val = mssim_cc(reconst, gt)
        out[f"scene_{idx+1}"] = {
            "rmse": float(rmse_total),
            "psnr": float(psnr),
            "sam_deg": float(sam_val),
            "ergas": float(ergas_val),
            "mssim": float(mssim_val),
            "cc": float(cc_val),
            "pseudo_gt": bool(args.pseudo_gt),
        }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    if not args.quiet:
        print(json.dumps(out, indent=2))


def build_arg_parser():
    p = argparse.ArgumentParser(description="BASFE Fusion CLI")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root-dir", type=str, required=True)
    common.add_argument("--hrsize", type=int, default=20)
    common.add_argument("--stride", type=int, default=7)
    common.add_argument("--edge", type=int, default=2)
    common.add_argument("--scale", type=int, default=4)
    common.add_argument("--quiet", action="store_true")
    common.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes")

    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--num-filter", type=int, default=32)
    pt.add_argument("--save-dir", type=str, default="./_saved_models")

    pr = sub.add_parser("reconstruct", parents=[common])
    pr.add_argument("--model-path", type=str, required=True)
    pr.add_argument("--out-dir", type=str, default="./results")

    pm = sub.add_parser("metrics", parents=[common])
    pm.add_argument("--model-path", type=str, required=True)
    pm.add_argument("--out-dir", type=str, default="./results")
    pm.add_argument("--pseudo-gt", action="store_true")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "reconstruct":
        reconstruct(args)
    elif args.mode == "metrics":
        compute_metrics(args)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
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
    # Optional noise suppression
    if getattr(args, 'quiet', False):
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')  # suppress INFO and WARNING
    conf = cfg.load_config(force_fast=args.fast_test, root_dir=args.root_dir, use_gt=args.use_gt)
    if args.gt_dir:
        # Accept absolute or relative; if relative, assume under ROOT_DIR
        conf['TEST_GT_HR_HSI_DIR'] = args.gt_dir
    if args.pseudo_gt_test_hsi:
        conf['PSEUDO_GT_TEST_HSI'] = True
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
    p.add_argument('--pseudo-gt-test-hsi', action='store_true', help='Use upsampled test LR-HSI as pseudo GT when real GT missing')
    # Note: disabling XLA via env flags is brittle in some runners; omitted.
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
