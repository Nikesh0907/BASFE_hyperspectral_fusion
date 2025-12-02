import os
import sys
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Progress disabled if tqdm missing

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


def infer_hrsize_from_model(model):
    """Infer patch size from a loaded model (used in reconstruct/metrics)."""
    try:
        li = model.get_layer('msi_input')
        val = getattr(li, 'batch_input_shape', None)
        if val and val[1]:
            return int(val[1])
    except Exception:
        pass
    try:
        li = model.get_layer('lr_input')
        val = getattr(li, 'batch_input_shape', None)
        if val and val[1]:
            return int(val[1])
    except Exception:
        pass
    try:
        s = model.inputs[0].shape[1]
        if s:
            return int(s)
    except Exception:
        pass
    return None


def train(args):
    enable_quiet_logs()
    configure_gpu(args.gpus)
    scenes = discover_scene_paths(args.root_dir, "Train")
    if not scenes:
        raise RuntimeError("No training scenes found under Train/HSI and Train/RGB")
    if getattr(args, 'list_scenes', False):
        print(f"Discovered {len(scenes)} training scene pairs:")
        for i, s in enumerate(scenes, 1):
            print(f"{i:02d}: HSI={os.path.basename(s['hsi'])} RGB={os.path.basename(s['rgb'])}")
        return

    # Determine patch size; allow resume to dictate hrsize
    model = None
    hrsize = args.hrsize
    if getattr(args, 'resume_from', None):
        if os.path.isfile(args.resume_from):
            model = keras.models.load_model(args.resume_from)
            inferred = infer_hrsize_from_model(model)
            if inferred:
                hrsize = inferred
            print(f"Resuming from: {args.resume_from} (hrsize inferred={hrsize})")
        else:
            raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")

    stride = args.stride
    scale = args.scale

    hr_list = []
    lr_list = []
    mr_list = []
    hsi_bands = None
    msi_bands = None

    scene_subset = scenes[: args.max_scenes or len(scenes)]
    if (not args.quiet and tqdm) or (args.show_progress and tqdm):
        print(f"Building training patches (hrsize={hrsize}, stride={stride}) from {len(scene_subset)} scene(s)...")
    loop = tqdm(scene_subset, desc="Train scenes", unit="scene") if ((not args.quiet and tqdm) or (args.show_progress and tqdm)) else scene_subset
    patch_counts = []
    rng = np.random.default_rng(seed=getattr(args, 'seed', None))

    def augment_triplet(hr_patch, lr_patch, mr_patch):
        if not args.augment:
            return hr_patch, lr_patch, mr_patch
        # Random horizontal flip
        if rng.random() < 0.5:
            hr_patch = hr_patch[:, ::-1, :]
            lr_patch = lr_patch[:, ::-1, :]
            mr_patch = mr_patch[:, ::-1, :]
        # Random vertical flip
        if rng.random() < 0.5:
            hr_patch = hr_patch[::-1, :, :]
            lr_patch = lr_patch[::-1, :, :]
            mr_patch = mr_patch[::-1, :, :]
        # Random 0/90/180/270 rotation
        k = rng.integers(0, 4)
        if k:
            hr_patch = np.rot90(hr_patch, k, axes=(0, 1))
            lr_patch = np.rot90(lr_patch, k, axes=(0, 1))
            mr_patch = np.rot90(mr_patch, k, axes=(0, 1))
        return hr_patch, lr_patch, mr_patch

    for s in loop:
        hrhsi, hrmsi, lrhsi_up = load_scene(s["hsi"], s["rgb"], scale=scale)
        if hsi_bands is None:
            hsi_bands = hrhsi.shape[2]
        if msi_bands is None:
            msi_bands = hrmsi.shape[2]
        scene_patches_hr = []
        scene_patches_lr = []
        scene_patches_mr = []
        if args.sampling == "grid":
            # Full grid extraction then optional subsample
            h, l, m = extract_patches(hrhsi, hrmsi, lrhsi_up, hrsize=hrsize, stride=stride)
            if args.subsample_factor > 1:
                idx = np.arange(h.shape[0])
                keep = idx[idx % args.subsample_factor == 0]
                h = h[keep]; l = l[keep]; m = m[keep]
            for pi in range(h.shape[0]):
                ah, al, am = augment_triplet(h[pi], l[pi], m[pi])
                scene_patches_hr.append(ah)
                scene_patches_lr.append(al)
                scene_patches_mr.append(am)
        else:  # random sampling
            H, W, _ = hrhsi.shape
            max_i = H - hrsize
            max_j = W - hrsize
            # Auto random count heuristic if not specified: approximate moderate overlap grid
            rand_count = args.random_per_scene or int(((H - hrsize) / max(stride, hrsize//2) + 1) * ((W - hrsize) / max(stride, hrsize//2) + 1))
            for _ in range(rand_count):
                i = rng.integers(0, max_i + 1)
                j = rng.integers(0, max_j + 1)
                hr_patch = hrhsi[i:i+hrsize, j:j+hrsize, :]
                lr_patch = lrhsi_up[i:i+hrsize, j:j+hrsize, :]
                mr_patch = hrmsi[i:i+hrsize, j:j+hrsize, :]
                ah, al, am = augment_triplet(hr_patch, lr_patch, mr_patch)
                scene_patches_hr.append(ah)
                scene_patches_lr.append(al)
                scene_patches_mr.append(am)
        # Collate scene
        if scene_patches_hr:
            hr_scene_arr = np.stack(scene_patches_hr, axis=0)
            lr_scene_arr = np.stack(scene_patches_lr, axis=0)
            mr_scene_arr = np.stack(scene_patches_mr, axis=0)
            hr_list.append(hr_scene_arr)
            lr_list.append(lr_scene_arr)
            mr_list.append(mr_scene_arr)
            patch_counts.append(hr_scene_arr.shape[0])

    if not hr_list:
        raise RuntimeError("No patches collected; adjust sampling parameters.")

    hrdata = np.concatenate(hr_list, axis=0)
    lrdata = np.concatenate(lr_list, axis=0)
    mrdata = np.concatenate(mr_list, axis=0)

    # Optional cap total patches
    if args.max_total_patches and hrdata.shape[0] > args.max_total_patches:
        sel = rng.choice(hrdata.shape[0], size=args.max_total_patches, replace=False)
        hrdata = hrdata[sel]; lrdata = lrdata[sel]; mrdata = mrdata[sel]
        # Recompute patch counts approximation
        # (We cannot map back to scenes easily after shuffle; just show total)
        patch_counts = ["capped"]

    # Memory warning
    triplet_bytes = hrdata[0].nbytes + lrdata[0].nbytes + mrdata[0].nbytes
    est_bytes = triplet_bytes * hrdata.shape[0]
    if est_bytes > 2_000_000_000:  # > ~2GB
        print(f"Warning: patch set uses ~{est_bytes/1_000_000_000:.2f} GB; consider increasing stride, subsampling, or random sampling with fewer patches.")

    # Build or reuse model
    if model is None:
        model = build_basfe_model(hrsize=hrsize, hsi_bands=hsi_bands, msi_bands=msi_bands, num_filter=args.num_filter)

        # Learning rate scheduling
        lr_value = args.lr
        lr_schedule = None
        if args.lr_schedule == "cosine":
            decay_steps = max(1, args.lr_decay_steps)
            lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=lr_value, decay_steps=decay_steps, alpha=args.lr_min / max(lr_value, 1e-12))
        elif args.lr_schedule == "step":
            # Use ExponentialDecay with staircase to emulate step drops
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_value, decay_steps=max(1, args.lr_decay_steps), decay_rate=args.lr_decay_rate, staircase=True)

        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule or lr_value)
        model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
    else:
        # Resumed model keeps its optimizer and schedule
        pass

    # Ensure save directory exists (needed for checkpoints even when resuming)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        if not getattr(args, 'resume_from', None):
            model.save(os.path.join(args.save_dir, "model_untrained.keras"))

    class EpochTiming(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self._start = time.time()
        def on_epoch_end(self, epoch, logs=None):
            dur = time.time() - getattr(self, '_start', time.time())
            print(f"Epoch {epoch+1}/{args.epochs} time: {dur:.2f}s")

    callbacks = [EpochTiming()]
    if args.reduce_on_plateau:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=max(1, args.plateau_patience), min_lr=args.lr_min))
    if args.early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(monitor='loss', patience=max(1, args.es_patience), restore_best_weights=True))
    if args.save_dir and args.checkpoint_every > 0:
        class EpochCheckpoint(keras.callbacks.Callback):
            def __init__(self, save_dir, every):
                super().__init__()
                self.save_dir = save_dir
                self.every = every
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.every == 0:
                    fname = f"model_epoch_{epoch+1}.keras"
                    self.model.save(os.path.join(self.save_dir, fname))
        callbacks.append(EpochCheckpoint(args.save_dir, args.checkpoint_every))
    verbose = 1 if (not args.quiet or args.show_progress) else 0
    print(f"Training summary: scenes_used={len(scene_subset)} sampling={args.sampling} patches_per_scene={patch_counts} total_patches={hrdata.shape[0]}")
    if args.epochs > 0:
        model.fit([mrdata, lrdata], hrdata, epochs=args.epochs, batch_size=args.batch_size, verbose=verbose, callbacks=callbacks)

    if args.save_dir:
        model.save(os.path.join(args.save_dir, "model_trained.keras"))


def reconstruct(args):
    enable_quiet_logs()
    configure_gpu(args.gpus)
    model = keras.models.load_model(args.model_path)
    scenes = discover_scene_paths(args.root_dir, "Test")
    if not scenes:
        raise RuntimeError("No test scenes found under Test/HSI and Test/RGB")

    inferred = infer_hrsize_from_model(model)
    hrsize = inferred or args.hrsize
    edge = args.edge
    scale = args.scale

    out_dir = args.out_dir or os.path.join(args.root_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    test_subset = scenes[: args.max_scenes or len(scenes)]
    outer = tqdm(test_subset, desc="Reconstruct scenes", unit="scene") if ((not args.quiet and tqdm) or (args.show_progress and tqdm)) else test_subset
    for idx, s in enumerate(outer):
        scene_start = time.time()
        hrhsi, hrmsi, lrhsi_up = load_scene(s["hsi"], s["rgb"], scale=scale)
        H, W, L = hrhsi.shape
        ii, jj = tile_indices(H, W, hrsize, edge)
        num = ii.size * jj.size
        mrdatainput = np.zeros((num, hrsize, hrsize, hrmsi.shape[2]), dtype=np.float32)
        lrdatainput = np.zeros((num, hrsize, hrsize, hrhsi.shape[2]), dtype=np.float32)
        patch_coords = [(i, j) for i in ii for j in jj]
        inner = tqdm(patch_coords, desc=f"Scene {idx+1} patches", unit="patch") if ((not args.quiet and tqdm) or (args.show_progress and tqdm)) else patch_coords
        c = 0
        for (i, j) in inner:
            mrdatainput[c] = hrmsi[i : i + hrsize, j : j + hrsize]
            lrdatainput[c] = lrhsi_up[i : i + hrsize, j : j + hrsize]
            c += 1
        pred = model.predict([mrdatainput, lrdatainput], verbose=0)
        reconst = np.zeros_like(hrhsi)
        c = 0
        for (i, j) in patch_coords:
            reconst[i : i + hrsize, j : j + hrsize] = pred[c]
            c += 1
        c = 0
        for (i, j) in patch_coords:
            reconst[i + edge : i + hrsize - edge, j + edge : j + hrsize - edge] = pred[c, edge:-edge, edge:-edge]
            c += 1
        mat_save(os.path.join(out_dir, f"reconst_{idx+1}.mat"), "reconst", reconst)
        if not args.quiet:
            print(f"Scene {idx+1} time: {time.time()-scene_start:.2f}s")


def compute_metrics(args):
    enable_quiet_logs()
    configure_gpu(args.gpus)
    scenes = discover_scene_paths(args.root_dir, "Test")
    if not scenes:
        raise RuntimeError("No test scenes found under Test/HSI and Test/RGB")

    # Use model input size if available for consistency, else CLI value
    model = keras.models.load_model(args.model_path)
    hrsize = infer_hrsize_from_model(model) or args.hrsize
    edge = args.edge
    scale = args.scale

    # model already loaded above
    out = {}
    acc = {"rmse": [], "psnr": [], "sam_deg": [], "ergas": [], "mssim": [], "cc": []}
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
        scene_metrics = {
            "rmse": float(rmse_total),
            "psnr": float(psnr),
            "sam_deg": float(sam_val),
            "ergas": float(ergas_val),
            "mssim": float(mssim_val),
            "cc": float(cc_val),
            "pseudo_gt": bool(args.pseudo_gt),
        }
        out[f"scene_{idx+1}"] = scene_metrics
        acc["rmse"].append(scene_metrics["rmse"])
        acc["psnr"].append(scene_metrics["psnr"])
        acc["sam_deg"].append(scene_metrics["sam_deg"])
        acc["ergas"].append(scene_metrics["ergas"])
        acc["mssim"].append(scene_metrics["mssim"])
        acc["cc"].append(scene_metrics["cc"])

    # Add averages across all processed scenes
    if acc["rmse"]:
        out["average"] = {
            "rmse": float(np.mean(acc["rmse"])),
            "psnr": float(np.mean(acc["psnr"])),
            "sam_deg": float(np.mean(acc["sam_deg"])),
            "ergas": float(np.mean(acc["ergas"])),
            "mssim": float(np.mean(acc["mssim"])),
            "cc": float(np.mean(acc["cc"])),
            "pseudo_gt": bool(args.pseudo_gt),
            "num_scenes": int(len(acc["rmse"]))
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
    common.add_argument("--show-progress", action="store_true", help="Show tqdm progress bars & epoch timings even if quiet")
    common.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use (>=1)")
    common.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes")

    pt = sub.add_parser("train", parents=[common])
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--batch-size", type=int, default=16)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--num-filter", type=int, default=32)
    pt.add_argument("--save-dir", type=str, default="./_saved_models")
    pt.add_argument("--resume-from", type=str, default="", help="Path to .keras model to resume training from")
    pt.add_argument("--list-scenes", action="store_true", help="List discovered training scenes and exit")
    pt.add_argument("--sampling", choices=["grid","random"], default="grid", help="Patch sampling strategy")
    pt.add_argument("--random-per-scene", type=int, default=0, help="Random patches per scene (sampling=random, 0=auto heuristic)")
    pt.add_argument("--subsample-factor", type=int, default=1, help="Keep every K-th grid patch (sampling=grid)")
    pt.add_argument("--augment", action="store_true", help="Enable random flips/rotations augmentation")
    pt.add_argument("--max-total-patches", type=int, default=0, help="Cap total patches after collection (0=no cap)")
    pt.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    # LR scheduling & callbacks
    pt.add_argument("--lr-schedule", choices=["none","cosine","step"], default="none", help="Learning rate schedule")
    pt.add_argument("--lr-decay-steps", type=int, default=10000, help="Decay steps for schedules (approx batches)")
    pt.add_argument("--lr-decay-rate", type=float, default=0.5, help="Decay rate for step schedule")
    pt.add_argument("--lr-min", type=float, default=1e-6, help="Minimum lr for cosine/plateau")
    pt.add_argument("--reduce-on-plateau", action="store_true", help="Enable ReduceLROnPlateau on training loss")
    pt.add_argument("--plateau-patience", type=int, default=5, help="Patience epochs for ReduceLROnPlateau")
    pt.add_argument("--early-stopping", action="store_true", help="Enable EarlyStopping on training loss")
    pt.add_argument("--es-patience", type=int, default=15, help="Patience epochs for EarlyStopping")
    pt.add_argument("--checkpoint-every", type=int, default=0, help="Save model every N epochs (0=disabled)")

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
