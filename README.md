# BASFE Hyperspectral Fusion

Modular implementation of BASFE-based hyperspectral–multispectral fusion supporting CAVE-style datasets (Train/Test folders with text index lists) and legacy directory layouts. The original notebook `_BASFE_PaviaU.ipynb` has been refactored into reusable Python modules under `basfe_fusion/`.

## Package Structure

```
basfe_fusion/
	__init__.py
	config.py          # Configuration + fast preset + adaptive patch sizing
	utils.py           # I/O helpers (load .mat cube, list mats, RGB loader)
	dataset.py         # File discovery + patch dataset construction
	model.py           # BASFE model builder
	train.py           # Training loop + callbacks + summary export
	reconstruct.py     # Patch-based reconstruction for test scenes
	metrics.py         # RMSE / PSNR / SAM / ERGAS / MSSIM / CC metrics
	main.py            # CLI entrypoint (train / reconstruct / metrics / full)
requirements.txt     # Dependencies
```

## Dataset Layout (TXT_INDEXED mode)

```
ROOT_DIR/
	Train/
		HSI/          # HR-HSI .mat cubes (used also to derive LR if no explicit LR)
		RGB/          # HR-MSI / RGB (mat or image formats)
		Train.txt     # One basename per line (without extension)
	Test/
		HSI/          # LR-HSI .mat cubes (or HR-HSI if deriving LR internally)
		RGB/          # HR-MSI / RGB
		Test.txt      # One basename per line
		GT_HR/        # (optional) HR-HSI ground truth for metrics
```

## Quick Install

```bash
pip install -r requirements.txt
```

## Running the Pipeline

`main.py` supports four modes:

- `train`        : build patches and train model
- `reconstruct`  : reconstruct test scenes using existing (or freshly trained) model
- `metrics`      : run reconstruction then metrics
- `full`         : train + reconstruct + metrics end-to-end

### Example Commands

Smoke test (fast preset):
```bash
python -m basfe_fusion.main --mode full --fast-test 1 --root-dir /path/to/Data
```

Full run with adaptive patch sizing and metrics (ensure GT_HR present):
```bash
python -m basfe_fusion.main --mode full --fast-test 0 --use-gt 1 --root-dir /path/to/Data
```

Reconstruction only (assumes trained weights saved to `/kaggle/working/BASFE_CAVE_trained.keras` or current working dir override):
```bash
python -m basfe_fusion.main --mode reconstruct --root-dir /path/to/Data
```

Metrics only (will reconstruct first then compute):
```bash
python -m basfe_fusion.main --mode metrics --use-gt 1 --root-dir /path/to/Data
```

## Adaptive Patch Sizing

When `--fast-test 0` and `AUTOTUNE_PATCH=True` in config, the system selects a patch size among `[48, 40, 32, 24, 20, 16, 12]` based on scene dimensions, adjusts overlap and stride to keep estimated reconstruction grid below `TARGET_MAX_GRID_PATCHES` (default 500). This prevents excessive memory/time for large scenes while ensuring full coverage (no reconstruction cap) for metrics.

## Ground Truth Metrics

Enable via `--use-gt 1` and place matching `.mat` cubes named `<basename>.mat` in `Test/GT_HR/`. Metrics saved to `results/metrics.json` with per-scene values: RMSE, PSNR, SAM (deg), ERGAS, MSSIM, CC.

## Mixed Precision

Set `MIXED_PRECISION` in config (default `fp16`) to enable Keras mixed precision policy for speed/memory benefits. Falls back gracefully if unsupported.

## Outputs

- Checkpoints: `checkpoints/epoch_###_loss_*.keras`
- Final trained model: `BASFE_CAVE_trained.keras`
- Reconstruction `.mat` files: `results/<scene>_reconst.mat`
- Training summary: `results/training_summary.json`
- Metrics: `results/metrics.json`

## Customization

Edit `config.py` or pass `--root-dir`, `--fast-test`, `--use-gt`. For further tuning (e.g., different `TARGET_MAX_GRID_PATCHES`), modify config then rerun.

## Notebook

Original exploratory notebook `_BASFE_PaviaU.ipynb` remains for interactive experimentation; the modular code offers reproducible CLI execution.

## Disclaimer

Ensure dataset preparation matches expected layout. Large-scale training may require adjusting memory caps, batch size, or disabling fast preset.
