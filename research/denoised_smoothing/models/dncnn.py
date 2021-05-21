"""
# Referenced from:
https://github.com/microsoft/denoised-smoothing/blob/master/code/archs/dncnn.py
# Original:
https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_train.py
"""
from tensorflow.keras import layers
import tensorflow as tf

WEIGHT_DECAY = 1e-4

def conv_block(x: tf.Tensor, channels: int = 64, ksize: int = 3,
  padding: str = "same", bn: bool = False) -> tf.Tensor:
  """Constructs a convolution block.

  :param x: inputs [batch_size, h, w, nb_channels]
  :param channels: number of channels (int)
  :param ksize: kernel size (int or a tuple of ints)
  :param padding: padding type (str)
  :param bn: whether to use Batch Normalization
  :return: output of the convolution block
  """
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


def run_dncnn(x: tf.Tensor, image_chnls: int = 3, depth: int = 17,
  n_channels: int = 64) -> tf.Tensor:
  """Runs a DNCNN block.

  :param x: inputs [batch_size, h, w, nb_channels]
  :param image_chnls: number of channels in the output images (int)
  :param depth: depth of the network (int)
  :param n_channels: number of channels in the convolutional layers (int)
  :return: batch of images
  """
  x = conv_block(x, channels=n_channels)
  for _ in range(depth - 2):
    x = conv_block(x, channels=n_channels, bn=True)

  outputs = layers.Conv2D(
    image_chnls,
    kernel_size=3,
    padding="same",
    use_bias=False,
    kernel_initializer="orthogonal"
  )(x)
  return outputs


def get_dncnn(image_size: int = 32, image_chnls: int = 3,
  depth: int = 17, n_channels: int = 64) -> tf.keras.Model:
  """Constructs a DNCNN model.

  :param image_size: size of the images (int)
  :param image_chnls: number of channels in the inputs images (int)
  :param depth: depth of the network (int)
  :param n_channels: number of channels in the convolutional layers (int)
  :return: DNCNN model
  """
  inputs = layers.Input((image_size, image_size, image_chnls))
  outputs = run_dncnn(inputs, depth=depth, n_channels=n_channels)
  outputs = layers.Subtract()([inputs, outputs])
  return tf.keras.Model(inputs, outputs)
