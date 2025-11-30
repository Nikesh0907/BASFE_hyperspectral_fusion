import os, math, random, numpy as np

DEFAULT_ROOT = os.environ.get('CAVE_ROOT', '/kaggle/input/cave-dataset-2/Data')

FAST_TEST_PRESET_ENV = os.environ.get('FAST_TEST_PRESET')
FAST_TEST_PRESET = False
if FAST_TEST_PRESET_ENV is not None:
    FAST_TEST_PRESET = str(FAST_TEST_PRESET_ENV).strip().lower() in ('1','true','yes','y')

# Public keys shown in summary
SUMMARY_KEYS = [
    'ROOT_DIR','PATCH_HR_SIZE','PATCH_STRIDE','EDGE_OVERLAP','EPOCHS','BATCH_SIZE',
    'LEARNING_RATE','TOTAL_PATCHES_LIMIT','RANDOM_PATCHES_PER_SCENE','SUBSAMPLE_PATCH_RATE',
    'BUILD_PATCHES_MODE','MAX_TRAIN_SCENES','MAX_TEST_SCENES','MIXED_PRECISION','PRED_PIXEL_BUDGET','RECON_STRIDE_MULT','RECON_MAX_PATCHES'
]

def base_config():
    return {
        'ROOT_DIR': DEFAULT_ROOT,
        # Dataset mode discovery
        'DATASET_MODE': 'TXT_INDEXED',
        'TRAIN_HR_HSI_DIR_CAND': ['Z/train/X', 'train/X', 'Train/X'],
        'TRAIN_LR_HSI_DIR_CAND': ['Z/train/X_blur', 'train/X_blur', 'Train/X_blur'],
        'TRAIN_HR_MSI_DIR_CAND': ['Z/train/Y', 'train/Y', 'Train/Y'],
        'TEST_LR_HSI_DIR_CAND':  ['Z/test/X', 'test/X', 'Test/X'],
        'TEST_HR_MSI_DIR_CAND':  ['Z/test/Y', 'test/Y', 'Test/Y'],
        'TEST_GT_HR_HSI_DIR_CAND': ['Z/test/GT_HR', 'test/GT_HR', 'Test/GT_HR'],
        'TRAIN_DIR': 'Train', 'TEST_DIR': 'Test', 'TRAIN_TXT': 'Train.txt', 'TEST_TXT': 'Test.txt',
        'HSI_SUBDIR': 'HSI', 'RGB_SUBDIR': 'RGB', 'GT_SUBDIR': 'GT_HR',
        'TEST_GT_HR_HSI_DIR': None, 'USE_GT': False,
        'MSI_BANDS_SELECT': None,
        # Patch / adaptive sizing
        'PATCH_HR_SIZE': 20, 'PATCH_STRIDE': 7, 'EDGE_OVERLAP': 2,
        'AUTOTUNE_PATCH': True, 'TARGET_MAX_GRID_PATCHES': 500,
        # Limits / sampling
        'MAX_TRAIN_SCENES': None, 'MAX_TEST_SCENES': None,
        'EPOCHS': 50, 'BATCH_SIZE': 64, 'LEARNING_RATE': 1e-4,
        # Output paths
        'SAVE_MODEL_PATH': os.path.join('/kaggle/working', 'BASFE_CAVE_init.keras'),
        'SAVE_MODEL_TRAINED_PATH': os.path.join('/kaggle/working', 'BASFE_CAVE_trained.keras'),
        'LOG_DIR': os.path.join('/kaggle/working', 'logs'),
        'RESULTS_DIR': os.path.join('/kaggle/working', 'results'),
        'CHECKPOINT_DIR': os.path.join('/kaggle/working', 'checkpoints'),
        'CSV_LOG': os.path.join('/kaggle/working', 'training_log.csv'),
        'EARLY_STOP_PATIENCE': 10, 'SEED': 42,
        'PATCH_MEMORY_CAP_MB': 4000, 'SUBSAMPLE_PATCH_RATE': 1, 'TRUNCATE_TO_CAP': True,
        'MIXED_PRECISION': 'fp16', 'SHOW_CONFIG_SUMMARY': True,
        'TRAIN_PROGRESS_EVERY_BATCHES': 10, 'PRED_PATCH_PROGRESS': True, 'SKIP_PLOT': False,
        'FAST_DEBUG_MODE': False, 'RANDOM_PATCHES_PER_SCENE': None, 'TOTAL_PATCHES_LIMIT': None,
        'BUILD_PATCHES_MODE': 'grid', 'ESTIMATE_ONLY': False,
        'MICRO_DEBUG_MODE': False,'MICRO_MAX_PATCHES': 64,'MICRO_EPOCHS': 1,'MICRO_MAX_TRAIN_SCENES': 1,
        'PRED_PIXEL_BUDGET': 2048,'RECON_STRIDE_MULT': 1,'RECON_MAX_PATCHES': None,
        'DERIVE_LR_FROM_HR': False,
        'PSEUDO_GT_TEST_HSI': False,
    }

def apply_fast_test(config):
    config.update({
        'PATCH_HR_SIZE': 12,'PATCH_STRIDE': 10,'MAX_TRAIN_SCENES': 1,'MAX_TEST_SCENES': 1,
        'BUILD_PATCHES_MODE': 'random','RANDOM_PATCHES_PER_SCENE': 40,'TOTAL_PATCHES_LIMIT': 32,
        'SUBSAMPLE_PATCH_RATE': 3,'SKIP_PLOT': True,'EPOCHS': 1,'BATCH_SIZE': 16,
        'FAST_DEBUG_MODE': True,'PRED_PIXEL_BUDGET': 65536,'RECON_STRIDE_MULT': 1,'RECON_MAX_PATCHES': 1500,
    })


def autotune_patch(config, hr_sample):
    if config.get('AUTOTUNE_PATCH') and not FAST_TEST_PRESET:
        h_dim, w_dim = hr_sample.shape[:2]
        base_min = min(h_dim, w_dim)
        candidates = [48, 40, 32, 24, 20, 16, 12]
        chosen = 12
        for c in candidates:
            if base_min >= c:
                chosen = c
                break
        provisional_edge = max(2, chosen//6)
        stride_est = max(4, chosen - 2*provisional_edge)
        grid_rows = math.ceil(max(1, h_dim - chosen + 1) / stride_est)
        grid_cols = math.ceil(max(1, w_dim - chosen + 1) / stride_est)
        est_patches = grid_rows * grid_cols
        if est_patches > config['TARGET_MAX_GRID_PATCHES']:
            for c2 in candidates:
                if c2 > chosen and base_min >= c2:
                    chosen = c2
                    provisional_edge = max(2, chosen//6)
                    stride_est = max(4, chosen - 2*provisional_edge)
                    grid_rows = math.ceil(max(1, h_dim - chosen + 1) / stride_est)
                    grid_cols = math.ceil(max(1, w_dim - chosen + 1) / stride_est)
                    est_patches = grid_rows * grid_cols
                    if est_patches <= config['TARGET_MAX_GRID_PATCHES']:
                        break
        config['PATCH_HR_SIZE'] = chosen
        config['EDGE_OVERLAP'] = provisional_edge
        config['PATCH_STRIDE'] = stride_est
        config['RECON_MAX_PATCHES'] = None
        return {
            'PATCH_HR_SIZE': chosen,
            'EDGE_OVERLAP': provisional_edge,
            'PATCH_STRIDE': stride_est,
            'EST_GRID': est_patches,
        }
    return None


def summarize(config):
    if not config.get('SHOW_CONFIG_SUMMARY'): return
    print('\n--- CONFIG SUMMARY ---')
    for k in SUMMARY_KEYS:
        print(f'{k}: {config.get(k)}')
    if config.get('AUTOTUNE_PATCH') and not FAST_TEST_PRESET:
        print('AUTOTUNE_PATCH used for adaptive sizing.')
    if config.get('USE_GT'):
        print('GT enabled. TEST_GT_HR_HSI_DIR:', config.get('TEST_GT_HR_HSI_DIR'))
    if config.get('PSEUDO_GT_TEST_HSI'):
        print('Pseudo-GT fallback (upsampled test LR-HSI) enabled.')
    print('----------------------\n')


def load_config(force_fast: bool | None = None, root_dir: str | None = None, use_gt: bool | None = None):
    config = base_config()
    if root_dir:
        config['ROOT_DIR'] = root_dir
    if use_gt is not None:
        config['USE_GT'] = bool(use_gt)
    fast = FAST_TEST_PRESET if force_fast is None else force_fast
    if fast:
        apply_fast_test(config)
    return config
