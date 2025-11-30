import os, time, math, json
import numpy as np
from tensorflow import keras
import tensorflow as tf

class BatchProgress(keras.callbacks.Callback):
    def __init__(self, every_n=50, total_batches=None):
        super().__init__()
        self.every_n = every_n
        self.total_batches = total_batches
        self.start_time = None
        self.epoch_start = None
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()
        print(f"\n[Epoch {epoch+1}] starting ...")
    def on_batch_end(self, batch, logs=None):
        if (batch+1) % self.every_n == 0:
            elapsed_epoch = time.time() - self.epoch_start if self.epoch_start else 0
            remaining = 0
            if self.total_batches:
                rate = (batch+1)/max(elapsed_epoch,1e-6)
                remaining_batches = self.total_batches - (batch+1)
                remaining = remaining_batches / max(rate,1e-6)
            print(f"  Batch {batch+1}/{self.total_batches} loss={logs.get('loss'):.5f} epoch_elapsed={int(elapsed_epoch)}s epoch_eta={int(remaining)}s")


def train_model(model, hrdata, lrdata, mrdata, config):
    steps_per_epoch = math.ceil(hrdata.shape[0] / config['BATCH_SIZE'])
    callbacks = [
        keras.callbacks.CSVLogger(config['CSV_LOG'], append=False),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['CHECKPOINT_DIR'], 'epoch_{epoch:03d}_loss_{loss:.5f}.keras'),
            monitor='loss', save_best_only=True, save_weights_only=False, verbose=1),
        keras.callbacks.EarlyStopping(monitor='loss', patience=config['EARLY_STOP_PATIENCE'], restore_best_weights=True, verbose=1),
        keras.callbacks.TensorBoard(log_dir=config['LOG_DIR'], write_graph=False, update_freq='epoch'),
        BatchProgress(every_n=config['TRAIN_PROGRESS_EVERY_BATCHES'], total_batches=steps_per_epoch)
    ]
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=config['LEARNING_RATE']), loss=keras.losses.MeanSquaredError())
    print('Starting training: epochs', config['EPOCHS'], 'batch size', config['BATCH_SIZE'], 'steps/epoch', steps_per_epoch)
    start_train = time.time()
    history = model.fit({'msi_input': mrdata, 'lr_input': lrdata}, hrdata,
                        epochs=config['EPOCHS'], batch_size=config['BATCH_SIZE'], shuffle=True, verbose=0, callbacks=callbacks)
    train_time = time.time() - start_train
    print(f'Training completed in {train_time/60:.2f} min ({int(train_time)}s)')
    try:
        model.save(config['SAVE_MODEL_TRAINED_PATH'])
        print('Trained model saved to', config['SAVE_MODEL_TRAINED_PATH'])
    except Exception as e:
        print('WARNING: trained model save failed:', e)
    summary_info = {
        'epochs_ran': len(history.history['loss']),
        'final_loss': float(history.history['loss'][-1]),
        'train_time_sec': train_time,
        'num_patches': int(hrdata.shape[0]),
        'hr_patch_size': config['PATCH_HR_SIZE'],
        'bands_hsi': hrdata.shape[-1],
        'bands_msi': mrdata.shape[-1],
        'mixed_precision': config['MIXED_PRECISION'],
    }
    os.makedirs(config['RESULTS_DIR'], exist_ok=True)
    with open(os.path.join(config['RESULTS_DIR'], 'training_summary.json'), 'w') as f:
        json.dump(summary_info, f, indent=2)
    return history
