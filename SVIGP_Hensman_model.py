import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def _add_diagonal_jitter(matrix, jitter=1e-8):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + jitter)


class SVIGP_Hensman:

    def __init__(self, fixed_inducing_points, initial_inducing_points,
                 name, jitter, N_train, dtype, L, fixed_gp_params, object_vectors_init, K_obj_normalize=False):
        """
        Class for SVGP model from Hensman (2013).

        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param name: name (or index) of the latent channel
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L: number of latent channels used in SVGPVAE
        :param fixed_gp_params:
        :param object_vectors_init: initial value for object vectors (PCA embeddings).
                        If None, object vectors are fixed throughout training. GPLVM
        :param K_obj_normalize: whether or not to normalize object linear kernel
        """

        self.dtype = dtype
        self.jitter = jitter
        self.nr_inducing = len(initial_inducing_points)
        self.N_train = N_train
        self.L = L
        self.K_obj_normalize = K_obj_normalize

        # u (inducing points)
        if fixed_inducing_points:
            self.inducing_index_points = tf.constant(initial_inducing_points, dtype=self.dtype)
        else:
            self.inducing_index_points = tf.Variable(initial_inducing_points, dtype=self.dtype,
                                                     name='Sparse_GP_inducing_points_{}'.format(name))

        # GP hyperparams
        if fixed_gp_params:
            self.l_GP = tf.constant(1.0, dtype=self.dtype)
            self.amplitude = tf.constant(1.0, dtype=self.dtype)
        else:
            self.l_GP = tf.Variable(initial_value=1.0, name="GP_length_scale_{}".format(name), dtype=self.dtype)
            self.amplitude = tf.Variable(initial_value=1.0, name="GP_amplitude_{}".format(name), dtype=self.dtype)

        # kernels
        self.kernel_view = tfk.ExpSinSquared(amplitude=self.amplitude, length_scale=self.l_GP, period=2 * np.pi)
        self.kernel_object = tfk.Linear()

        # object vectors (GPLVM)
        if object_vectors_init is not None:
            self.object_vectors = tf.Variable(initial_value=object_vectors_init,
                                              name="GP_object_vectors_{}".format(name),
                                              dtype=self.dtype)
        else:
            self.object_vectors = None

        # (inner) variational parameters
        self.variational_inducing_observations_loc = [tf.Variable(np.zeros([self.nr_inducing], dtype=self.dtype),
                                    name='GP_var_params_mu_{}'.format(l + 1)) for l in range(self.L)]

        self.variational_inducing_observations_scale = [tf.Variable(np.eye(self.nr_inducing, dtype=self.dtype),
                                            name='GP_var_params_A_{}'.format(l + 1)) for l in range(self.L)]

        self.variational_inducing_observations_cov_mat = [tf.matmul(x, tf.transpose(x)) for x in
                                                          self.variational_inducing_observations_scale]

        self.noise = tf.Variable(initial_value=0.1, name="Hensman_likelihood_noise", dtype=self.dtype)

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y). Kernel from Casale's paper is used for rotated MNIST data.

        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param y_inducing: whether y is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        # unpack auxiliary data
        if self.object_vectors is None:
            x_view, x_object, y_view, y_object = x[:, 1], x[:, 2:], y[:, 1], y[:, 2:]
        else:
            x_view, y_view = x[:, 1], y[:, 1]
            if x_inducing:
                x_object = x[:, 2:]
            else:
                x_object = tf.gather(self.object_vectors, tf.cast(x[:, 0], dtype=tf.int64))
            if y_inducing:
                y_object = y[:, 2:]
            else:
                y_object = tf.gather(self.object_vectors, tf.cast(y[:, 0], dtype=tf.int64))

        # compute kernel matrix
        if diag_only:
            view_matrix = self.kernel_view.apply(tf.expand_dims(x_view, axis=1), tf.expand_dims(y_view, axis=1))
        else:
            view_matrix = self.kernel_view.matrix(tf.expand_dims(x_view, axis=1), tf.expand_dims(y_view, axis=1))

        if diag_only:
            object_matrix = self.kernel_object.apply(x_object, y_object)
            if self.K_obj_normalize:
                obj_norm = tf.math.reduce_euclidean_norm(x_object, axis=1) * tf.math.reduce_euclidean_norm(y_object,
                                                                                                           axis=1)
                object_matrix = object_matrix / obj_norm
        else:
            object_matrix = self.kernel_object.matrix(x_object, y_object)
            if self.K_obj_normalize:  # normalize object matrix
                obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_object, axis=1, keepdims=True),
                                         tf.transpose(tf.math.reduce_euclidean_norm(y_object, axis=1, keepdims=True),
                                                      perm=[1, 0]))
                object_matrix = object_matrix * obj_norm

        return view_matrix * object_matrix

    def variable_summary(self):
        """
        Returns values of parameters of sparse GP object. For debugging purposes.
        :return:
        """

        return self.l_GP, self.amplitude, self.object_vectors, self.inducing_index_points

    def variational_loss(self, x, z, lat_channel):
        """
        Computes L_H for the data in the current batch.

        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param z: latent variables for current latent channel (batch, 1)
        :param lat_channel: latent channel index

        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (1,)
        """
        b = tf.shape(x)[0]
        m = self.inducing_index_points.get_shape()[0]
        b = tf.cast(b, dtype=self.dtype)
        m = tf.cast(m, dtype=self.dtype)

        # kernel matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)

        K_nn = self.kernel_matrix(x, x, x_inducing=False, y_inducing=False, diag_only=True)  # (b)

        K_nm = self.kernel_matrix(x, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mn = tf.transpose(K_nm, perm=[1, 0])  # (m, b)

        variational_inducing_observations_loc = self.variational_inducing_observations_loc[lat_channel]
        variational_inducing_observations_cov_mat = self.variational_inducing_observations_cov_mat[lat_channel]

        # K_nm \cdot K_mm_inv \cdot m,  (b,)
        mean_vector = tf.linalg.matvec(K_nm,
                                       tf.linalg.matvec(K_mm_inv, variational_inducing_observations_loc))

        S = variational_inducing_observations_cov_mat

        # KL term
        K_mm_chol = tf.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
        S_chol = tf.linalg.cholesky(
            _add_diagonal_jitter(variational_inducing_observations_cov_mat, self.jitter))
        K_mm_log_det = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(K_mm_chol)))
        S_log_det = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(S_chol)))

        KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                         tf.trace(tf.matmul(K_mm_inv, variational_inducing_observations_cov_mat)) +
                         tf.reduce_sum(variational_inducing_observations_loc *
                                       tf.linalg.matvec(K_mm_inv, variational_inducing_observations_loc)))

        # diag(K_tilde), (b, )
        precision = 1 / self.noise

        K_tilde_terms = precision * (K_nn - tf.linalg.diag_part(tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))))

        # k_i \cdot k_i^T, (b, m, m)
        lambda_mat = tf.matmul(tf.expand_dims(K_nm, axis=2),
                               tf.transpose(tf.expand_dims(K_nm, axis=2), perm=[0, 2, 1]))

        # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
        lambda_mat = tf.matmul(K_mm_inv, tf.matmul(lambda_mat, K_mm_inv))

        # Trace terms, (b,)
        trace_terms = precision * tf.trace(tf.matmul(S, lambda_mat))

        # L_3 sum part, (1,)
        L_3_sum_term = -0.5 * (tf.reduce_sum(K_tilde_terms) + tf.reduce_sum(trace_terms))

        return L_3_sum_term, KL_term, mean_vector

    def approximate_posterior_params(self, index_points_test, lat_channel):
        """
        Computes parameters of q_S.

        :param index_points_test: X_*
        :param lat_channel:

        :return: posterior mean at index points,
                 (diagonal of) posterior covariance matrix at index points
        """

        variational_inducing_observations_loc = self.variational_inducing_observations_loc[lat_channel]
        variational_inducing_observations_cov_mat = self.variational_inducing_observations_cov_mat[lat_channel]

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False,
                                  y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)

        A = tf.matmul(K_xm, K_mm_inv)

        mean_vector = tf.linalg.matvec(A, variational_inducing_observations_loc)

        mid_mat = K_mm - variational_inducing_observations_cov_mat
        B = K_xx - tf.matmul(A, tf.matmul(mid_mat, tf.transpose(A, perm=[1, 0])))

        return mean_vector, B


def forward_pass_deep_SVIGP_Hensman(data_batch, vae, svgp):
    """
    Forward pass for deep SVIGP_Hensman on rotated MNIST data (based on discussions in Feb 2021).

    :param data_batch: (images, aux_data). images dimension: (batch_size, 28, 28, 1).
        aux_data dimension: (batch_size, 10)
    :param beta:
    :param vae: VAE object
    :param svgp: SVGP object
    :param C_ma: average constraint from t-1 step (GECO)
    :param lagrange_mult: lambda from t-1 step (GECO)
    :param kappa: reconstruction level parameter for GECO
    :param alpha: moving average parameter for GECO
    :param GECO: whether or not to use GECO algorithm for training
    :param use_qS: If True, use qS (sparse GP posterior) to obtain latent vectors z. Else, use latent vectors directly.

    :return:
    """

    images, aux_data = data_batch
    _, w, h, c = images.get_shape()
    K = tf.cast(w, dtype=vae.dtype) * tf.cast(h, dtype=vae.dtype) * tf.cast(c, dtype=vae.dtype)
    b = tf.cast(tf.shape(images)[0], dtype=vae.dtype)  # batch_size
    L = svgp.L

    inside_elbo_recon, inside_elbo_kl, mean_vectors = [], [], []
    for l in range(L):  # iterate over latent dimensions
        inside_elbo_recon_l,  inside_elbo_kl_l, mean_l = svgp.variational_loss(x=aux_data[:, 1:], z=None, lat_channel=l)

        inside_elbo_recon.append(inside_elbo_recon_l)
        inside_elbo_kl.append(inside_elbo_kl_l)
        mean_vectors.append(mean_l)

    inside_elbo_recon = tf.reduce_sum(inside_elbo_recon)
    inside_elbo_kl = tf.reduce_sum(inside_elbo_kl)

    inside_elbo = inside_elbo_recon - (b / svgp.N_train) * inside_elbo_kl
    KL_term = inside_elbo

    mean_vectors = tf.stack(mean_vectors, axis=1)

    # DECODER NETWORK
    recon_images_logits = vae.decode(mean_vectors)
    recon_images = recon_images_logits


    recon_loss = tf.reduce_sum((images - recon_images_logits) ** 2)

    # ELBO
    # beta plays role of sigma_gaussian_decoder here (\lambda(\sigma_y) in Casale paper)
    # K and L are not part of ELBO. They are used in loss objective to account for the fact that magnitudes of
    # reconstruction and KL terms depend on number of pixels (K) and number of latent GPs used (L), respectively
    # recon_loss = recon_loss / K
    # elbo = - recon_loss + (beta / L) * KL_term

    elbo = -b*K*tf.log(svgp.noise) - 0.5*b*K*tf.cast(tf.log(2 * np.pi), dtype=svgp.dtype) \
           -(0.5*svgp.noise**(-2))*recon_loss + inside_elbo
    recon_loss = recon_loss / K

    return elbo, recon_loss, KL_term, inside_elbo,  recon_images, inside_elbo_recon, inside_elbo_kl, mean_vectors


def predict_deep_SVIGP_Hensman(test_data_batch, vae, svgp):
    """

    :param test_data_batch: batch of test data
    :param vae: fitted (!) VAE object
    :param svgp: fitted (!) SVGP object
    :return:
    """

    images_test_batch, aux_data_test_batch = test_data_batch
    L = svgp.L

    _, w, h, _ = images_test_batch.get_shape()

    # get mean vectors for test data from GP posterior
    p_m = []
    for l in range(L):  # iterate over latent dimensions
        p_m_l, _ = svgp.approximate_posterior_params(index_points_test=aux_data_test_batch[:, 1:], lat_channel=l)
        p_m.append(p_m_l)

    p_m = tf.stack(p_m, axis=1)

    latent_samples = p_m

    # predict (decode) latent images.
    # ===============================================
    # Since this is generation (testing pipeline), could add \sigma_y to images
    recon_images_test_logits = vae.decode(latent_samples)

    # Gaussian observational likelihood, no variance
    recon_images_test = recon_images_test_logits

    # Bernoulli observational likelihood
    # recon_images_test = tf.nn.sigmoid(recon_images_test_logits)

    # Gaussian observational likelihood, fixed variance \sigma_y
    # recon_images_test = recon_images_test_logits + tf.random.normal(shape=tf.shape(recon_images_test_logits),
    #                                                                 mean=0.0, stddev=0.04, dtype=tf.float64)

    # MSE loss for CGEN (here we do not consider MSE loss, ince )
    recon_loss = tf.reduce_sum((images_test_batch - recon_images_test_logits) ** 2)

    # report per pixel loss
    K = tf.cast(w, dtype=tf.float64) * tf.cast(h, dtype=tf.float64)
    recon_loss = recon_loss / K
    # ===============================================

    return recon_images_test, recon_loss
