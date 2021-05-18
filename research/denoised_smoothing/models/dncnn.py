# Referenced from: https://github.com/microsoft/denoised-smoothing/blob/master/code/archs/dncnn.py
# Original: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_train.py

from tensorflow.keras import layers
import tensorflow as tf

WEIGHT_DECAY = 1e-4

def conv_block(x, channels=64, ksize=3, padding="same", bn=False):
    x = layers.Conv2D(
        channels,
        kernel_size=ksize,
        padding=padding,
        use_bias=True if bn else False,
        kernel_initializer="orthogonal"
    )(x)
    if bn:
        x = layers.BatchNormalization(momentum=0.0, epsilon=0.0001)(x)
    x = layers.Activation("relu")(x)
    return x


def run_dncnn(x, image_chnls=3, depth=17, n_channels=64):
    x = conv_block(x, channels=n_channels)
    for _ in range(depth - 2):
        x = conv_block(x, channels=n_channels, bn=True)

    outputs = layers.Conv2D(
        image_chnls,
        kernel_size=3,
        padding="same",
        use_bias=False,
        kernel_initializer="orthogonal",
    )(x)
    return outputs


def get_dncnn(image_size=32, image_chnls=3,
              depth=17, n_channels=64):
    inputs = layers.Input((image_size, image_size, image_chnls))
    outputs = run_dncnn(inputs, depth=depth, n_channels=n_channels)
    outputs = layers.Subtract()([inputs, outputs])
    return tf.keras.Model(inputs, outputs)
