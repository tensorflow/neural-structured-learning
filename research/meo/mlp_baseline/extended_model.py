# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class definitions for the feature extraction mapping.

This file contains class definitions for the feature extraction and clean to
obfuscated mappings used by our proposed method. The base class EmbeddingMapper
is used as an inferface for the embedding maps that we shall use. The classes
FeatureExtractor and FeatureExtractorWithClassifier make use of these embedding
mappers to create embeddings for the obfuscated images and use them for
classification.
"""

import abc

from typing import Union, Tuple, Sequence, Optional

import tensorflow as tf
import tensorflow_hub as hub


class EmbeddingMapper(tf.keras.Model, metaclass=abc.ABCMeta):
  """Model that maps input embeddings to output embeddings in the same dimensionality.

  This is used as a template for classes implementing a mapping between
  embeddings. Classes that inherit this class must implement an appropriate call
  function.
  """

  def __init__(self):
    super().__init__()
    pass

  @abc.abstractmethod
  def call(self, inputs: tf.Tensor) -> Union[tf.Tensor, Sequence[tf.Tensor]]:  # pytype: disable=signature-mismatch
    """Abstract call function to be implemented by subclasses.

    Args:
      inputs: the input embeddings that the class must operate on.

    Returns:
      Either a single tensor which corresponds to the output embeddings, or a
      tuple of tensors, one of which is the output embedding and the rest are
      necessary components of the optimization process for each particular
      subclass.
    """


class IdentityEmbeddingMapper(EmbeddingMapper):
  """Placeholder class for an identity mapping.

  This class simply returns the mappings provided to the input. It exists for
  compatibility with the rest of the codebase.
  """

  def __init__(self):
    super().__init__()
    pass

  def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pytype: disable=signature-mismatch
    return inputs


class MLPEmbeddingMapper(EmbeddingMapper):
  """Mapping from input to output embeddings using an MLP.

  This functions wraps a model that maps an embedding vector to another one in
  the same space (with the intent being to map obfuscated embeddings to clean
  ones). The mapping is performed using multi-layer perceptron.

  Attributes:
    mapping: Mapping MLP, taking obfuscated embeddings as input and
      returning clean ones.
  """

  def __init__(self,
               embed_dim: int,
               mlp_sizes: Sequence[int],
               weight_decay: float = 1e-4,
               final_activation: Optional[str] = 'relu'):
    super().__init__()
    self.mapping = tf.keras.Sequential()
    self.mapping.add(tf.keras.layers.Flatten())
    for i in range(len(mlp_sizes)):
      self.mapping.add(tf.keras.layers.Dense(
          mlp_sizes[i],
          activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
          bias_regularizer=tf.keras.regularizers.l2(weight_decay),
          name='mapping_{0:}'.format(i)))

    self.mapping.add(tf.keras.layers.Dense(
        embed_dim,
        activation=final_activation,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='mapping_final'))

  def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pytype: disable=signature-mismatch
    return self.mapping(inputs)


class AutoEncoderEmbeddingMapper(EmbeddingMapper):
  """AutoEncoder style architecture to map between embeddings.

  This class implements an autoencoder style architecture between two
  embedding spaces. This consists of a single encoder and one or more decoder
  heads from the latent dimension to the output. Both the encoder and the
  decoders are implemented as MLPs. Optionally, a skip connection may be added
  from the input embedding space to each of the output embedding spaces.

  When calling this layer, the input is assumed to be a 2-dimensional tensor, of
  shape (batch_size, embed_dim). The output is a 3-dimensional tensor, of
  shape (batch_size, num_decoders, embed_dim) - one extra dimension for the
  varying number of decoder heads.

  Attributes:
    encoder: The MLP mapping from input embedding to latent dimension.
    decoders: A list of decoder MLPs, mapping from latent dimension to the
      various output spaces.
    skip_connection: Whether to add a skip connection from the input embedding
      space to each of the output embedding spaces.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               num_decoders: int = 1,
               weight_decay: float = 1e-4,
               skip_connection: bool = False):
    super().__init__()
    if len(mlp_sizes) % 2 == 0:
      raise ValueError('In this, mlp_sizes must contain an odd number of'
                       'elements. The middle one corresponds to the latent'
                       'dimension of the autoencoder, and the rest to the sizes'
                       'of the encoder and the decoder (first half and second'
                       'half, respectively).')

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(
        latent_dim,
        encoder_mlp_sizes,
        weight_decay
    )

    self.decoders = []
    for _ in range(num_decoders):
      decoder = MLPEmbeddingMapper(embed_dim, decoder_mlp_sizes, weight_decay)
      self.decoders.append(decoder)

    self.skip_connection = skip_connection

  def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pytype: disable=signature-mismatch
    """Method to apply the autoencoder based mapping.

    Args:
      inputs: A 2-dimensional tensor, of shape (batch_size, embed_dim).

    Returns:
      result: A 3-dimensional tensor, of shape (batch_size, num_decoders,
        embed_dim)
    """
    x = self.encoder(inputs)
    decoder_outputs = []
    for i in range(len(self.decoders)):
      decoder = self.decoders[i]
      out = decoder(x)
      if self.skip_connection:
        out = tf.keras.layers.Add()([out, inputs])
      out = tf.expand_dims(out, axis=1)
      decoder_outputs.append(out)

    result = tf.keras.layers.Concatenate(axis=1)(decoder_outputs)
    return result


class GANEmbeddingMapper(EmbeddingMapper):
  """Embedding mapper relying on an adversarially trained network.

  This class makes use of a generator which generates clean embeddings provided
  obfuscated embeddings as input, as well as a discriminator which decides
  whether the provided embeddings come from clean images or from the generator.
  Both of these models are defined using MLPs, mapping from the space of
  obfuscated embedddings to the space of clean embeddings.

  Attributes:
    generator: Generator MLP, mapping obfuscated embeddings to clean ones.
    discriminator: Discriminator MLP, identifying whether its input embedding is
      real or created by the generator.
  """

  def __init__(self,
               embed_dim: int,
               generator_mlp_sizes: Sequence[int],
               discriminator_mlp_sizes: Sequence[int],
               weight_decay: float = 1e-4):
    super().__init__()
    self.generator = MLPEmbeddingMapper(
        embed_dim=embed_dim,
        mlp_sizes=generator_mlp_sizes,
        weight_decay=weight_decay)

    self.discriminator = MLPEmbeddingMapper(
        embed_dim=1,
        mlp_sizes=discriminator_mlp_sizes,
        weight_decay=weight_decay)

    # Start with generator trainable and discriminator frozen.
    self.generator.trainable = True
    self.discriminator.trainable = False

  def flip_training(self):
    """Flip between training the generator and the discriminator.
    """
    self.generator.trainable = not self.generator.trainable
    self.discriminator.trainable = not self.discriminator.trainable

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:  # pytype: disable=signature-mismatch
    embeddings = self.generator(inputs)
    domain = self.discriminator(embeddings)
    return embeddings, domain


class VAEEmbeddingMapper(EmbeddingMapper):
  """Embeddings mapping based on a Variational AutoEncoder (VAE).

  This class generates clean embeddings from the obfuscated ones using a VAE.
  More specifically, the encoder architecture generates a mean and a log
  variance for the latent normal distribution, the components of which are
  uncorrelated. These are then used to generate samples for the decoder.

  Both the encoder and decoder architectures are based on MLPs.

  Attributes:
    encoder: VAE encoder, defined as an MLP.
    decoder: VAE decoder, defined as an MLP.
    encoder_mean: Layer encoding the mean of the latent normal distribution.
    encoder_logvar: Layer encoding the log variance of the latent normal
      distribution.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               weight_decay: float = 1e-4):
    super().__init__()

    if len(mlp_sizes) % 2 == 0:
      raise ValueError('In the case of VAE, mlp_sizes must contain an odd'
                       'number of elements. The middle one corresponds to the'
                       'latent dimension of the VAE, and the rest to the sizes'
                       'of the encoder and the decoder (first half and second'
                       'half, respectively).')

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(latent_dim, encoder_mlp_sizes,
                                      weight_decay)
    self.decoder = MLPEmbeddingMapper(embed_dim, decoder_mlp_sizes,
                                      weight_decay)

    self.encoder_mean = tf.keras.layers.Dense(
        latent_dim,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='encoder_mean')

    self.encoder_logvar = tf.keras.layers.Dense(
        latent_dim,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='encoder_logvar')

  def call(self, inputs: tf.Tensor,  # pytype: disable=signature-mismatch
           training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = self.encoder(inputs)
    z_mean = self.encoder_mean(x)
    z_log_var = self.encoder_logvar(x)

    # During training, generate samples normally.
    if training:
      sample = tf.keras.backend.random_normal(shape=z_mean.shape)
    # During testing, use the means of the distribution to generate
    # representations.
    else:
      sample = 0
    y = z_mean + tf.exp(z_log_var) * sample
    y = self.decoder(y)
    return y, z_mean, z_log_var


class ParameterGenerationEmbeddingMapper(EmbeddingMapper):
  """Class that implements an autoencoder that uses a context dependent decoder.

  This class implements an autoencoder architecture that makes use of a set of
  parameter generators, that generate the parameters of the decoder to be used.
  These generators are context dependent - they are provided with context which
  is learned by another part of the model. This class makes use of a separate
  generator for each layer of the decoder.

  All separate parts of this architecture are implemented as MLPs. The decoder
  architecture is assumed to be symmetric to the encoder. The parameter
  generators all have the same architecture.

  Attributes:
    encoder: The common encoder model mapping from embedding space to latent
      space.
    context: The context oracle that derives the context vector from each of the
      provided embeddings. This means that the model predicts (is an oracle of)
      the context of the input, which correspons to the obfuscation type.
    param_dims: The dimensions of the intermediate vectors of the decoder, to
      which the parameter generators must adhere. This corresponds to the
      architecture of the decoders as MLPs.
    param_generator_list: The list of parameter generators for this model.
  """

  def __init__(
      self,
      encoder_decoder_mlp_sizes: Sequence[int],
      param_generator_mlp_sizes: Sequence[int],
      context_mlp_sizes: Sequence[int],
      embed_dim: int,
      latent_dim: int,
      context_dim: int,
      num_contexts: int = 0,
      weight_decay: float = 1e-4,
  ):
    """Init function.

    Args:
      encoder_decoder_mlp_sizes: The layer sizes of the encoder and the decoder
        architectures of the model.
      param_generator_mlp_sizes: The layer sizes of the parameter generator
        architecture.
      context_mlp_sizes: The layer sizes of the context oracle.
      embed_dim: The dimension of the embedding vectors.
      latent_dim: The latent dimension of the diffusion model.
      context_dim: The dimension of the context vectors.
      num_contexts: How many different domains the model should generate. If
        greater than 0, makes the model generate embeddings.
      weight_decay: L2 weight decay to add to the parameters of the model.
        Defaults to L2.
    """
    super().__init__()

    self.encoder = MLPEmbeddingMapper(
        latent_dim,
        encoder_decoder_mlp_sizes,
        weight_decay
    )

    self.context = MLPEmbeddingMapper(
        context_dim,
        context_mlp_sizes,
        weight_decay
    )

    decoder_mlp_sizes = encoder_decoder_mlp_sizes[::-1]
    self.num_contexts = num_contexts
    self.generation = num_contexts > 0
    self.param_dims = [latent_dim] + list(decoder_mlp_sizes) + [embed_dim]
    self.param_generator_list = []
    for i in range(len(self.param_dims)-1):
      # The generated parameters are param_dims[i] * param_dims[i+1] for the
      # weight matrix, plus param_dims[i+1] for the bias.
      param_generator_output_dim = (self.param_dims[i]+1)*self.param_dims[i+1]

      # TODO(smyrnisg): Make this a single Dense layer.
      param_generator = MLPEmbeddingMapper(param_generator_output_dim,
                                           param_generator_mlp_sizes,
                                           weight_decay)
      self.param_generator_list.append(param_generator)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:  # pytype: disable=signature-mismatch
    """Derive the obfuscation context and apply the generated decoder.

    This method returns both the generated embeddings and the learned context,
    in order to optimize the context oracle. The context is trained so as to be
    representative of the obfuscation type of the image, in order to be given as
    input to the parameter generator afterwards, for the latter to give us the
    parameters of the correct decoder.

    Args:
      inputs: The input embeddings given to the model.

    Returns:
      A tuple containing the generated embeddings and the derived context
        vector.
    """
    latent_vec = self.encoder(inputs)
    batch_size = tf.shape(latent_vec)[0]
    if self.generation:
      context_vec = tf.expand_dims(tf.range(self.num_contexts), axis=0)
      context_vec = tf.repeat(context_vec, batch_size, axis=0)
      context_vec = tf.reshape(context_vec, [-1])
      context_vec = self.context(context_vec)
      result = tf.expand_dims(latent_vec, axis=1)
      result = tf.repeat(result, self.num_contexts, axis=0)
    else:
      context_vec = self.context(inputs)
      result = tf.expand_dims(latent_vec, axis=1)  # For proper batch matmul.
    for i in range(len(self.param_generator_list)):
      params = self.param_generator_list[i](context_vec)
      params = tf.reshape(
          params, [-1, self.param_dims[i]+1, self.param_dims[i+1]]
      )
      result = tf.matmul(result, params[:, :-1, :]) + params[:, -1:, :]

    result = result[:, 0, :]  # Remove the extra axis.
    if self.generation:
      result = tf.reshape(result, [-1, self.num_contexts, result.shape[-1]])

    return result, context_vec


class DiffusionEmbeddingMapper(EmbeddingMapper):
  """Diffusion model mapping embeddings from one domain to the other.

  This class implements a generator according to the techniques proposed in
  https://arxiv.org/pdf/2006.11239.pdf. In particular, this class implements a
  diffusion process in order to generate samples which attempt to mimic the
  images of the domain it was trained once

  During training, this class outputs a prediction of the noise added to the
  image, as well as the noise itself.

  In order to generate samples, a point from the normal latent space is sampled,
  and the reverse diffusion process is iteratively solved, in order to arrive at
  the generated image (without extra noise).

  In this class, betas, alphas and alphas_bar are defined as in the paper (see
  https://arxiv.org/pdf/2006.11239.pdf for more details).

  Attributes:
    encoder: The encoder part of the architecture.
    decoder: The decoder part of the architecture. Note that this also receives
      a timestep as input, in order to predict the noise at a particular
      timestep.
    concat_layer: Concatenation layer between the encoder and decoder.
    total_time: The total number of timesteps to run the diffusion process.
    betas: The values of beta used in the diffusion.
    alphas: The values of alpha used in the diffusion.
    alphas_bar: The values of alpha_bar used in the diffusion.
    num_points: Number of points in time to pick during training.
  """

  def __init__(self,
               mlp_sizes: Sequence[int],
               embed_dim: int,
               total_time: int = 1000,
               weight_decay: float = 1e-4,
               num_points: int = 1):
    super().__init__()

    if len(mlp_sizes) % 2 == 0:
      raise ValueError(
          'In this case, mlp_sizes must be a list of odd length. The first half'
          'of the list corresponds to the encoder part of the diffusion process'
          'while the second part corresponds to the decoder part. The middle'
          'element corresponds to the latent dimension.'
      )

    num_layers_encoder = len(mlp_sizes) // 2
    encoder_mlp_sizes = mlp_sizes[:num_layers_encoder]
    latent_dim = mlp_sizes[num_layers_encoder]
    decoder_mlp_sizes = mlp_sizes[num_layers_encoder+1:]

    self.encoder = MLPEmbeddingMapper(
        latent_dim,
        encoder_mlp_sizes,
        weight_decay
    )

    self.decoder = MLPEmbeddingMapper(
        embed_dim,
        decoder_mlp_sizes,
        weight_decay
    )

    self.concat_layer = tf.keras.layers.Concatenate(axis=-1)

    self.total_time = total_time

    # Below definitions are used as in https://arxiv.org/pdf/2006.11239.pdf.
    self.betas = tf.linspace(1e-4, 2e-2, self.total_time)
    self.alphas = 1 - self.betas
    self.alphas_bar = tf.math.cumprod(self.alphas)

    self.num_points = num_points

  def _noise_prediction(self, inputs: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """Predict the noise to add to the input at a given timestep.

    Args:
      inputs: The input to add noise to.
      t: The tensor containing the timesteps for the predictions of this batch.

    Returns:
      The noise prediction for the input at the given timestep.
    """
    x = self.encoder(inputs)
    x = self.concat_layer([x, tf.cast(t, tf.float32)])
    x = self.decoder(x)
    return x

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:  # pytype: disable=signature-mismatch

    multiple_inputs = tf.repeat(inputs, self.num_points, axis=0)

    t = tf.random.uniform([tf.shape(multiple_inputs)[0], 1],
                          minval=0,
                          maxval=self.total_time,
                          dtype=tf.int32)
    noise = tf.random.normal(tf.shape(multiple_inputs))

    chosen_alphas_bar = tf.gather(self.alphas_bar, tf.reshape(t, [-1]))

    noisy_image_1 = multiple_inputs * tf.expand_dims(
        tf.sqrt(chosen_alphas_bar), axis=1
    )
    noisy_image_2 = noise * tf.expand_dims(
        tf.sqrt(1 - chosen_alphas_bar), axis=1
    )
    noisy_image = noisy_image_1 + noisy_image_2

    return self._noise_prediction(noisy_image, t), noise

  def sample(
      self,
      embedding_prior: tf.Tensor,
      stop_gradient: bool = True
  ) -> tf.Tensor:
    """Return a batch of samples from the diffusion process.

    Args:
      embedding_prior: Embeddings on which to condition generation.
      stop_gradient: Whether to stop gradients to the model during sampling.
        Defaults to True.

    Returns:
      A batch of samples from the diffusion process.
    """
    result_shape = tf.shape(embedding_prior)
    result = tf.random.normal(result_shape) + embedding_prior
    for i in range(self.total_time-1, -1, -1):
      z = tf.random.normal(result_shape) if i > 1 else tf.zeros(result_shape)
      sigma = tf.sqrt(self.betas[i])
      model_factor = (1 - self.alphas[i])/tf.sqrt(1-self.alphas_bar[i])
      result = result - model_factor * self._noise_prediction(
          result, i * tf.ones([tf.shape(result)[0], 1], dtype=tf.int32)
      )
      result = result / tf.sqrt(self.alphas[i]) + sigma * z

    if stop_gradient:
      result = tf.stop_gradient(result)
    return result


class FeatureExtractor(tf.keras.Model):
  """Feature extractor wrapper.

  This class wraps a feature extraction model and optionally adds a mapping
  layer on top. The intent of this mapping is to transfer embeddings of clean
  images to obfuscated ones.

  The base model used in this architecture is defined using TensorFlow Hub. The
  model_link constructor parameter is one of the available models (see
  https://tfhub.dev/s?module-type=image-feature-vector for a list of available
  models).

  Attributes:
    base_model: Base feature extractor, originally trained on clean mappings.
      This model is kept frozen during training.
    encoder: EmbeddingMapper object, mapping the embeddings of the obfuscated
      images to the clean ones.
    base_model_trainable: Whether the base model is trainable or not. The
      default is False (so the base model is frozen).
    bypass_base_model: Optionally bypass the base model when using the call()
      method. This is meant to be used in the case where the input is already
      given in embedding space rather than in pixel space. The default is False.
  """

  def __init__(self,
               model_link: str,
               encoder: EmbeddingMapper,
               base_model_trainable: bool = False,
               bypass_base_model: bool = False):
    super().__init__()
    try:
      self.base_model_trainable = base_model_trainable
      self.base_model = hub.KerasLayer(model_link,
                                       trainable=self.base_model_trainable)
    except RuntimeError as e:
      raise ValueError('Implementation corresponding to model_link not found in'
                       'TensorFlow Hub. See https://tfhub.dev/'
                       's?module-type=image-feature-vector for a list of'
                       'available models') from e

    self.encoder = encoder
    self.bypass_base_model = bypass_base_model

  def _call_base_model(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    if self.bypass_base_model:
      x = inputs
    else:
      # If base model is not trainable, override training argument.
      x = self.base_model(
          inputs, training=(self.base_model_trainable and training)
      )
    return x

  def call(self, inputs: tf.Tensor,  # pytype: disable=signature-mismatch
           training: bool) -> Union[tf.Tensor, Sequence[tf.Tensor]]:
    x = self._call_base_model(inputs, training=training)
    x = self.encoder(x)
    return x

  def encode_clean(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    return self._call_base_model(inputs, training=training)

  def encode_obfuscated(self, inputs: tf.Tensor) -> tf.Tensor:
    return self(inputs)


class FeatureExtractorWithClassifier(tf.keras.Model):
  """Wrapper of FeatureExtractor with the linear classification head.

  This class implements the linear classification layer on top of the trained
  mapping from the obfuscated embedddings to the clean ones. This class uses a
  FeatureExtractor model as a part of its pipeline. No assumption is made on the
  trainability of the FeatureExtractor model internally.

  Attributes:
    feature_extractor: FeatureExtractor object, which extracts embeddings from
      the obfuscated images and maps them to embeddings of clean images.
    clf_layer: Linear classifier on top of the trained mapping from obfuscated
      to clean embedddings.
  """

  def __init__(self,
               num_classes: int,
               feature_extractor: FeatureExtractor,
               weight_decay: float = 1e-4):
    super().__init__()
    self.feature_extractor = feature_extractor

    self.clf_layer = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        name='classifier')

  def call(self, inputs: tf.Tensor) -> tf.Tensor:  # pytype: disable=signature-mismatch
    x = self.feature_extractor(inputs)
    # Obtain only the first item, the encoded embeddings, from feature_extractor
    # that returns tuple.
    if isinstance(x, (list, tuple)):
      x = x[0]
    x = self.clf_layer(x)
    return x
