import numpy as np
import tensorflow as tf


def _add_diagonal_jitter(matrix, jitter=1e-2):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + jitter)


def build_MLP_inference_graph(vid_batch, full_cholesky=False, layers=[500], tftype=tf.float32):
    """
    Takes a placeholder for batches of videos to be fed in, returns
    a mean and var of 2d latent space that are tf variables.

    args:
        vid_batch: tf placeholder (batch, tmax, width, height)
        full_cholesky: whether or not model learns entire cholesky matrix or just single param (var) per time step
        layers: list of widths of fully connected layers
        tftype: data type to use in graph

    returns:
        means:  tf variable, (batch, tmax, 2) x,y points.
        vars:  tf variable, (batch, tmax, output_dim) x,y uncertainties. output_dim equals 60 if full_cholesky else 2
    """

    batch, tmax, px, py = vid_batch.get_shape()

    # first layer, flatten images to vectors
    h0 = tf.reshape(vid_batch, (batch*tmax, px*py))

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))), name="encW")
        B = tf.Variable(tf.zeros([1, l]), name="encB")
        h0 = tf.matmul(h0, W) + B
        h0 = tf.nn.tanh(h0)

    if full_cholesky:  # learn full cholesky matrix, note that model is over-parametrized here
        output_dim = 2*(tf.cast(tmax, dtype=tf.int32) + 1)
    else: # final layer just outputs x,y mean and log(var) of q network
        output_dim = 2*(1 + 1)

    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, output_dim],
            stddev=1.0 / np.sqrt(float(i_dims))), name="encW")
    B = tf.Variable(tf.zeros([1, output_dim]), name="encB")
    h0 = tf.matmul(h0, W) + B

    h0 = tf.reshape(h0, (batch, tmax, output_dim))

    q_means = h0[:, :, :2]
    q_vars = tf.exp(h0[:, :, 2:])

    return q_means, q_vars


def build_MLP_decoder_graph(latent_samples, px, py, layers=[500]):
    """
    Constructs a TF graph that goes from latent points in 2D time series
    to a bernoulli probabilty for each pixel in output video time series.
    Args:
        latent_samples: (batch, tmax, 2), tf variable
        px: image width (int)
        py: image height (int)
        layers: list of num. of nodes (list of ints)

    Returns:
        pred_batch_vid_logits: (batch, tmax, px, py) tf variable
    """

    batch, tmax, _ = latent_samples.get_shape()

    # flatten all latents into one matrix (decoded in i.i.d fashion)
    h0 = tf.reshape(latent_samples, (batch*tmax, 2))

    # loop over layers in given list
    for l in layers:
        i_dims = int(h0.get_shape()[-1])
        W = tf.Variable(tf.truncated_normal([i_dims, l],
                stddev=1.0 / np.sqrt(float(i_dims))), name="decW")
        B = tf.Variable(tf.zeros([1, l]), name="decB")
        h0 = tf.matmul(h0, W) + B
        h0 = tf.nn.tanh(h0)

    # final layer just outputs full video batch
    l = px*py
    i_dims = int(h0.get_shape()[-1])
    W = tf.Variable(tf.truncated_normal([i_dims, l],
            stddev=1.0 / np.sqrt(float(i_dims))), name="decW")
    B = tf.Variable(tf.zeros([1, l]), name="decB")
    h0 = tf.matmul(h0, W) + B

    pred_vid_batch_logits = tf.reshape(h0, (batch, tmax, px, py))

    return pred_vid_batch_logits


class mnistVAE:

    dtype = tf.float64

    def __init__(self, im_width=28, im_height=28, L=16):
        """
        VAE for rotated MNIST data. Architecture (almost) same as in Casale (see Figure 4 in Supplementary material).

        :param im_width:
        :param im_height:
        :param L:
        """

        self.L = L

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(im_width, im_height, 1), dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(2 * self.L)
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, dtype=self.dtype),
                tf.keras.layers.Reshape(target_shape=(4, 4, 8)),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
            ])

    def encode(self, images):
        """

        :param images:
        :return:
        """

        encodings = self.encoder(images)
        means, vars = encodings[:, :self.L], tf.exp(encodings[:, self.L:])  # encoder outputs \mu and log(\sigma^2)
        return means, vars

    def decode(self, latent_samples):
        """

        :param latent_samples:
        :return:
        """

        recon_images = self.decoder(latent_samples)
        return recon_images


class mnistCVAE:

    dtype = tf.float64

    def __init__(self, im_width=28, im_height=28, L=16):
        """
        CVAE for rotated MNIST data. Architecture (almost) same as in Casale (see Figure 4 in Supplementary material).

        :param im_width:
        :param im_height:
        :param L:
        """

        self.L = L

        dim_last_layer = 2 * self.L

        self.encoder_first_part = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(im_width, im_height, 3), dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Flatten()
            ])

        self.encoder_last_part = tf.keras.Sequential([
                # No activation
                tf.keras.layers.Dense(dim_last_layer)
            ])

        self.decoder_first_part = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, dtype=self.dtype),
                tf.keras.layers.Reshape(target_shape=(4, 4, 8))
            ])

        self.decoder_last_part = tf.keras.Sequential(
            [
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
            ])

    def encode(self, images, angles):
        """

        :param images:
        :param angles:
        :return:
        """

        encodings = self.encoder_first_part(images)
        encodings = tf.concat([encodings, tf.expand_dims(tf.math.sin(angles), axis=1),
                               tf.expand_dims(tf.math.cos(angles), axis=1)], axis=1)
        encodings = self.encoder_last_part(encodings)

        means, vars = encodings[:, :self.L], tf.exp(encodings[:, self.L:])  # encoder outputs \mu and log(\sigma^2)

        return means, vars

    def decode(self, latent_samples, angles):
        """

        :param latent_samples:
        :param angles:
        :return:
        """
        b = tf.shape(latent_samples)[0]

        # add angle info to latent samples
        sin_ = tf.math.sin(angles)
        cos_ = tf.math.cos(angles)
        latent_samples = tf.concat([latent_samples, tf.expand_dims(sin_, axis=1),
                             tf.expand_dims(cos_, axis=1)], axis=1)

        # add angle info before first conv layer
        recon_images = self.decoder_first_part(latent_samples)
        sin_ = tf.reshape(tf.repeat(sin_, tf.repeat(4*4, b)), shape=(b, 4, 4, 1))
        cos_ = tf.reshape(tf.repeat(cos_, tf.repeat(4*4, b)), shape=(b, 4, 4, 1))
        recon_images = tf.concat([recon_images, sin_, cos_], axis=3)

        recon_images = self.decoder_last_part(recon_images)

        return recon_images


def KL_term_standard_normal_prior(mean_vector, var_vector, dtype):
    """
    Computes KL divergence between standard normal prior and variational distribution from encoder.

    :param mean_vector: (batch_size, L)
    :param var_vector:  (batch_size, L)
    :return: (batch_size, 1)
    """
    return 0.5 * (- tf.cast(tf.reduce_prod(tf.shape(mean_vector)), dtype=dtype)
                  - 2.0*tf.reduce_sum(tf.log(tf.sqrt(var_vector)))
                  + tf.reduce_sum(var_vector)
                  + tf.reduce_sum(mean_vector**2))


class spritesVAE:

    dtype = tf.float32

    def __init__(self, L, im_width=64, im_height=64, n_channels=3):
        """
        VAE for SPRITES data. Architecture (almost) same as in Casale for face dataset
        (see Figure 5 in Supplementary material).

        :param im_width:
        :param im_height:
        :param n_channels:
        :param L:
        """

        self.L = L

        # [14.8.] Currently, there are 3 pairs of conv layers and then we use linear layer to go from 1024 to 256.
        # Alternatively, could use 4 pairs of conv layers and omit linear layer at the end.
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(im_width, im_height, n_channels), dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(2, 2), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(2, 2), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(2, 2), activation='elu', padding='same', dtype=self.dtype),
                # tf.keras.layers.Conv2D(
                #     filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                # tf.keras.layers.Conv2D(
                #     filters=16, kernel_size=3, strides=(2, 2), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(2*L)
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(1024, dtype=self.dtype),
                tf.keras.layers.Reshape(target_shape=(8, 8, 16)),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=3, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype)
            ])

    def encode(self, images):
        """

        :param images:
        :return:
        """

        encodings = self.encoder(images)
        means, vars = encodings[:, :self.L], tf.exp(encodings[:, self.L:])  # encoder outputs \mu and log(\sigma^2)

        return means, vars

    def decode(self, latent_samples):
        """

        :param latent_samples:
        :return:
        """

        recon_images = self.decoder(latent_samples)
        return recon_images


class sprites_representation_network:

    dtype = tf.float32

    def __init__(self, L, im_width=64, im_height=64, n_channels=3):
        """
        Representation network for sprites data, to learn character style vectors for each frame.
        Architecture somewhat follows "Pool" architecture from GQN paper (see Figure S1).

        :param L:
        """

        self.repr_nn = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(im_width, im_height, n_channels),
                                           dtype=self.dtype, name="GP_repr_NN_1"),
                tf.keras.layers.Conv2D(
                    filters=L, kernel_size=2, strides=(2, 2), activation='elu', padding='same',
                    dtype=self.dtype, name="GP_repr_NN_2"),
                tf.keras.layers.Conv2D(
                    filters=L, kernel_size=2, strides=(2, 2), activation='elu', padding='same',
                    dtype=self.dtype, name="GP_repr_NN_3"),
                tf.keras.layers.Conv2D(
                    filters=L, kernel_size=2, strides=(2, 2), activation='elu', padding='same',
                    dtype=self.dtype, name="GP_repr_NN_4"),
                tf.keras.layers.AveragePooling2D(pool_size=(8, 8), padding='same',
                                                 name="GP_repr_NN_5"),
                tf.keras.layers.Flatten(name="GP_repr_NN_6")
            ])
