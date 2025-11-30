import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Support running as a script (no package parent) by fixing sys.path
if __package__ is None or __package__ == "":
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from basfe_fusion.dataset import discover_scene_paths, load_scene, extract_patches, tile_indices
    from basfe_fusion.model import build_basfe_model
    from basfe_fusion.io_utils import mat_save
    from basfe_fusion.metrics import rmse_psnr, sam, ergas, mssim_cc
else:
    from .dataset import discover_scene_paths, load_scene, extract_patches, tile_indices
    from .model import build_basfe_model
    from .io_utils import mat_save
    from .metrics import rmse_psnr, sam, ergas, mssim_cc


def enable_quiet_logs():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def configure_gpu(num_gpus: int):
    try:
        all_gpus = tf.config.list_physical_devices('GPU')
        if all_gpus:
            use = all_gpus[: max(1, min(num_gpus, len(all_gpus)))]
            tf.config.set_visible_devices(use, 'GPU')
            for d in use:
                try:
                    tf.config.experimental.set_memory_growth(d, True)
                except Exception:
                    pass
    except Exception:
        pass


def train(args):
    enable_quiet_logs()
    configure_gpu(args.gpus)
    scenes = discover_scene_paths(args.root_dir, "Train")
    if not scenes:
        raise RuntimeError("No training scenes found under Train/HSI and Train/RGB")

    # Ensure patch size matches the trained model
    # Try to infer patch size from the saved model input layers
    def _infer_hrsize_from_model(m):
        try:
            li = m.get_layer('msi_input')
            val = getattr(li, 'batch_input_shape', None)
            if val and val[1]:
                return int(val[1])
        except Exception:
            pass
        try:
            li = m.get_layer('lr_input')
            val = getattr(li, 'batch_input_shape', None)
            if val and val[1]:
                return int(val[1])
        except Exception:
            pass
        try:
            s = m.inputs[0].shape[1]
            if s:
                return int(s)
        except Exception:
            pass
        return None

    hrsize = _infer_hrsize_from_model(model) or args.hrsize
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

    # Newer Keras prefers positional inputs/targets over dicts for performance and clarity
    model.fit([mrdata, lrdata], hrdata, epochs=args.epochs, batch_size=args.batch_size, verbose=1 if not args.quiet else 0)

    if args.save_dir:
        model.save(os.path.join(args.save_dir, "model_trained.keras"))


def reconstruct(args):
    enable_quiet_logs()
    configure_gpu(args.gpus)
    model = keras.models.load_model(args.model_path)
    scenes = discover_scene_paths(args.root_dir, "Test")
    if not scenes:
        raise RuntimeError("No test scenes found under Test/HSI and Test/RGB")

    hrsize = _infer_hrsize_from_model(model) or args.hrsize
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
        pred = model.predict([mrdatainput, lrdatainput], verbose=0)
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
    configure_gpu(args.gpus)
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
        pred = model.predict([mrdatainput, lrdatainput], verbose=0)
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
    common.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=1)")
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
