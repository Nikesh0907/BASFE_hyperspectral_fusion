from tensorflow import keras
from keras import layers

# BASFE architecture replication

def spec(inputs, nf):
    x = layers.Conv2D(nf, 1, padding='same', use_bias=True)(inputs)
    x = layers.PReLU(shared_axes=[1,2])(x)
    x = layers.Conv2D(nf, 1, padding='same', use_bias=True)(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    return layers.Add()([inputs, x])

def spat(inputs, nf):
    x = layers.Conv2D(nf, 3, padding='same', use_bias=True)(inputs)
    x = layers.PReLU(shared_axes=[1,2])(x)
    x = layers.Conv2D(nf, 3, padding='same', use_bias=True)(x)
    x = layers.PReLU(shared_axes=[1,2])(x)
    return layers.Add()([inputs, x])


def build_model(patch_size: int, msi_bands: int, hsi_bands: int, num_filter: int = 32):
    msi_input = keras.Input(shape=(patch_size, patch_size, msi_bands), name='msi_input')
    x01 = layers.Conv2D(num_filter, 3, padding='same', use_bias=True)(msi_input)
    x01 = layers.PReLU(shared_axes=[1,2])(x01)
    x02 = spec(x01, num_filter)
    x02 = spat(x02, num_filter)
    x03 = layers.Concatenate()([x01, x02])

    x04 = layers.Conv2D(num_filter, 3, padding='same', use_bias=True)(x03)
    x04 = layers.PReLU(shared_axes=[1,2])(x04)
    x05 = spec(x04, num_filter)
    x05 = spat(x05, num_filter)
    x06 = layers.Concatenate()([x01, x04, x05])

    x07 = layers.Conv2D(num_filter, 5, padding='same', use_bias=True)(x06)
    x07 = layers.PReLU(shared_axes=[1,2])(x07)
    x07 = spec(x07, num_filter)
    x08 = spat(x07, num_filter)

    lr_input = keras.Input(shape=(patch_size, patch_size, hsi_bands), name='lr_input')
    x11 = layers.Conv2D(num_filter, 3, padding='same', use_bias=True)(lr_input)
    x11 = layers.PReLU(shared_axes=[1,2])(x11)
    x12 = spec(x11, num_filter)
    x12 = spat(x12, num_filter)
    x13 = layers.Concatenate()([x11, x12])

    x14 = layers.Conv2D(num_filter, 3, padding='same', use_bias=True)(x13)
    x14 = layers.PReLU(shared_axes=[1,2])(x14)
    x15 = spec(x14, num_filter)
    x15 = spat(x15, num_filter)
    x16 = layers.Concatenate()([x11, x14, x15])

    x17 = layers.Conv2D(num_filter, 5, padding='same', use_bias=True)(x16)
    x17 = layers.PReLU(shared_axes=[1,2])(x17)
    x17 = spec(x17, num_filter)
    x18 = spat(x17, num_filter)

    x21 = layers.Concatenate()([x01, x04, x07, x08, x11, x14, x17, x18])

    x22 = layers.Conv2D(hsi_bands, 3, padding='same', use_bias=True)(x21)
    x22 = layers.PReLU(shared_axes=[1,2])(x22)
    fuse_output = layers.Conv2D(hsi_bands, 3, padding='same', use_bias=True, name='fuse_output')(x22)
    fuse_output = layers.PReLU(shared_axes=[1,2], name='fuse_output_prelu')(fuse_output)

    model = keras.Model(inputs={'msi_input': msi_input, 'lr_input': lr_input}, outputs=fuse_output, name='BASFE_Fusion')
    return model
