import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp
from VAE_utils import build_MLP_inference_graph, build_MLP_decoder_graph, \
                      KL_term_standard_normal_prior
from utils import gauss_cross_entropy

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def _add_diagonal_jitter(matrix, jitter=1e-8):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + jitter)


class SVGP:

    dtype = np.float32

    def __init__(self, titsias, num_inducing_points, fixed_inducing_points, tmin, tmax, vidlt,
                 fixed_gp_params, name, jitter, ip_min, ip_max, GP_init):
        """
        SVGP class for the moving ball data.

        :param titsias: if true we use L_T (Titsias elbo). Else we use L_H (Hensman elbo).
        :param num_inducing_points:
        :param fixed_inducing_points:
        :param tmin: param specific to Pearce setting, first time index
        :param tmax: param specific to Pearce setting, last time index
        :param vidlt: param specific to Pearce setting, length-scale value with which data was generated.
        :param fixed_gp_params:
        :param name: name (or index) of the latent channel
        """

        self.titsias = titsias
        self.num_inducing_points = num_inducing_points
        self.tmin = tmin
        self.tmax = tmax
        self.ip_min = ip_min
        self.ip_max = ip_max
        self.jitter = jitter

        # inducing points
        if fixed_inducing_points:
            initial_inducing_points_ = np.linspace(self.tmin, self.tmax, num_inducing_points, dtype=self.dtype)
            self.inducing_index_points = tf.constant(initial_inducing_points_, dtype=self.dtype)
        else:
            initial_inducing_points_ = np.linspace(self.ip_min, self.ip_max, num_inducing_points, dtype=self.dtype)
            self.inducing_index_points = tf.Variable(initial_inducing_points_, dtype=self.dtype,
                                                 name='inducing_index_points_{}'.format(name))

        # length scale of Gaussian kernel
        if fixed_gp_params:
            self.l_GP = tf.constant(vidlt, dtype=self.dtype)
        else:
            self.l_GP = tf.Variable(initial_value=GP_init,  # vidlt
                                    name="GP_length_scale_{}".format(name), dtype=self.dtype)

        self.kernel = tfk.ExponentiatedQuadratic(amplitude=None, length_scale=self.l_GP)

    def variational_loss(self, x, y, noise, mu_hat, A_hat):
        """

        :param x: time index points (batch, tmax)
        :param y: mean vector for current latent channel, output of the encoder network (batch, tmax)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, tmax)
        :param mu_hat:
        :param A_hat:

        :return: sum_term, KL_term (variational loss = sum_term + KL_term)  (batch,)
        """
        _, T = x.get_shape()
        m = self.inducing_index_points.get_shape()
        T = tf.cast(T, dtype=tf.float32)
        m = tf.cast(m, dtype=tf.float32)

        precision = tf.math.reciprocal_no_nan(noise)

        # kernel matrices
        K_mm = self.kernel.matrix(tf.expand_dims(self.inducing_index_points, axis=1),
                                  tf.expand_dims(self.inducing_index_points, axis=1))  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_nn = self.kernel.matrix(tf.expand_dims(x, axis=2), tf.expand_dims(x, axis=2))  # (batch, tmax, tmax)

        K_nm = self.kernel.matrix(tf.expand_dims(x, axis=2), tf.expand_dims(self.inducing_index_points, axis=1))  # (batch, tmax, m)
        K_mn = tf.transpose(K_nm, perm=[0, 2, 1])  # (batch, m, tmax)

        if self.titsias:

            cov_mat = tf.linalg.diag(noise) + tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))
            cov_mat_inv = tf.linalg.inv(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_chol = tf.linalg.cholesky(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_log_det = 2*tf.reduce_sum(tf.log(tf.linalg.diag_part(cov_mat_chol)), axis=1)  # (batch)
            trace_term = precision * tf.linalg.diag_part(K_nn - tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))) # (batch, tmax)

            L_2_term = -0.5 * (T * tf.log(2*np.pi) + cov_mat_log_det +
                               tf.reduce_sum(y * tf.linalg.matvec(cov_mat_inv, y), axis=1) +
                               tf.reduce_sum(trace_term, axis=1))

            return L_2_term, 0.0

        else:  # Hensman

            # K_nm \cdot K_mm_inv \cdot m, (batch, tmax)
            mean_vector = tf.linalg.matvec(K_nm, tf.linalg.matvec(K_mm_inv, mu_hat))

            # diag(K_tilde), (batch, tmax)
            K_tilde_terms = precision * tf.linalg.diag_part(K_nn - tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn)))

            # k_i \cdot k_i^T, (batch, tmax, m, m)
            lambda_mat = tf.matmul(tf.expand_dims(K_nm, axis=3),
                                   tf.transpose(tf.expand_dims(K_nm, axis=3), perm=[0, 1, 3, 2]))

            # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (batch, tmax, m, m)
            lambda_mat = tf.matmul(K_mm_inv, tf.matmul(lambda_mat, K_mm_inv))

            # Trace terms, (batch, tmax)
            # trace_terms = precision * tf.trace(tf.matmul(A_hat, lambda_mat))
            A_hat_ = tf.repeat(tf.expand_dims(A_hat, axis=1), repeats=[T], axis=1)
            trace_terms = precision * tf.trace(tf.matmul(A_hat_, lambda_mat))

            # L_3 sum part, (batch)
            L_3_sum_term = -0.5*(tf.reduce_sum(K_tilde_terms, axis=1) + tf.reduce_sum(trace_terms, axis=1) +
                                 tf.reduce_sum(tf.log(noise), axis=1) + T*tf.log(2*np.pi) +
                                 tf.reduce_sum(precision * (y - mean_vector)**2, axis=1))

            # KL term
            K_mm_chol = tf.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
            S_chol = tf.linalg.cholesky(_add_diagonal_jitter(A_hat, self.jitter))
            K_mm_log_det = 2*tf.reduce_sum(tf.log(tf.linalg.diag_part(K_mm_chol)))
            S_log_det = 2*tf.reduce_sum(tf.log(tf.linalg.diag_part(S_chol)))

            KL_term = 0.5*(K_mm_log_det - S_log_det - m +
                           tf.trace(tf.matmul(K_mm_inv, A_hat)) +
                           tf.reduce_sum(A_hat *
                                         tf.linalg.matvec(K_mm_inv, A_hat)))

            return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points, y=None, noise=None):
        """

        :param index_points: points at which we want to evaluate posterior mean at
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return: posterior mean at index points (batch, tmax),
                 posterior covariance matrix at index points (batch, tmax, tmax)
        """

        # kernel matrices
        K_mm = self.kernel.matrix(tf.expand_dims(self.inducing_index_points, axis=1),
                                  tf.expand_dims(self.inducing_index_points, axis=1))  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_nn = self.kernel.matrix(tf.expand_dims(index_points, axis=2),
                                  tf.expand_dims(index_points, axis=2))  # (batch, tmax, tmax)
        K_nm = self.kernel.matrix(tf.expand_dims(index_points, axis=2), tf.expand_dims(self.inducing_index_points, axis=1))  # (batch, tmax, m)
        K_mn = tf.transpose(K_nm, perm=[0, 2, 1])  # (batch, m, tmax)

        sigma_l = K_mm + tf.matmul(K_mn, tf.matmul(tf.linalg.diag(tf.math.reciprocal_no_nan(noise)), K_nm))
        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        K_nm_Sigma_l_K_mn = tf.matmul(K_nm, tf.matmul(sigma_l_inv, K_mn))

        mean_vector = tf.linalg.matvec(K_nm_Sigma_l_K_mn, tf.math.reciprocal_no_nan(noise) * y)
        B = K_nn - tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn)) + K_nm_Sigma_l_K_mn

        mu_hat = tf.linalg.matvec(tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mn)),
                                                       tf.math.reciprocal_no_nan(noise) * y)
        A_hat = tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat


class mainSVGP:

    def __init__(self, titsias, fixed_inducing_points, initial_inducing_points,
                 name, jitter, N_train, dtype, L, K_obj_normalize=False):
        """
        SVGP main class.

        :param titsias: if true we use L_T (Titsias elbo). Else we use L_H (Hensman elbo).
        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param name: name (or index) of the latent channel
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L: number of latent channels used in SVGPVAE
        :param K_obj_normalize: whether or not to normalize object linear kernel
        """

        self.dtype = dtype
        self.jitter = jitter
        self.titsias = titsias
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

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y).

        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points
        :param y_inducing: whether y is a set of inducing points
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        raise NotImplementedError()

    def variational_loss(self, x, y, mu_hat, A_hat, noise=None):
        """
        Computes L_H for the data in the current batch.

        :param x: auxiliary data for current batch (batch, 1 + 1 + M)
        :param y: mean vector for current latent channel, output of the encoder network (batch, 1)
        :param noise: variance vector for current latent channel, output of the encoder network (batch, 1)
        :param mu_hat:
        :param A_hat:

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

        if self.titsias:

            cov_mat = tf.linalg.diag(noise) + tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))
            trace_term = tf.math.reciprocal_no_nan(noise) * (
                        K_nn - tf.linalg.diag_part(tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))))  # (b)
            cov_mat_inv = tf.linalg.inv(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_chol = tf.linalg.cholesky(_add_diagonal_jitter(cov_mat, self.jitter))
            cov_mat_log_det = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(cov_mat_chol)))

            L_2_term = -0.5 * (b * tf.cast(tf.log(2 * np.pi), dtype=self.dtype) + cov_mat_log_det +
                               tf.reduce_sum(y * tf.linalg.matvec(cov_mat_inv, y)) +
                               tf.reduce_sum(trace_term))

            return L_2_term, tf.constant(0.0, dtype=self.dtype)

        else:  # Hensman

            # K_nm \cdot K_mm_inv \cdot m,  (b,)
            mean_vector = tf.linalg.matvec(K_nm,
                                           tf.linalg.matvec(K_mm_inv, mu_hat))

            S = A_hat

            # KL term
            K_mm_chol = tf.linalg.cholesky(_add_diagonal_jitter(K_mm, self.jitter))
            S_chol = tf.linalg.cholesky(
                _add_diagonal_jitter(A_hat, self.jitter))
            K_mm_log_det = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(K_mm_chol)))
            S_log_det = 2 * tf.reduce_sum(tf.log(tf.linalg.diag_part(S_chol)))

            KL_term = 0.5 * (K_mm_log_det - S_log_det - m +
                             tf.trace(tf.matmul(K_mm_inv, A_hat)) +
                             tf.reduce_sum(mu_hat *
                                           tf.linalg.matvec(K_mm_inv, mu_hat)))

            # diag(K_tilde), (b, )
            precision = tf.math.reciprocal_no_nan(noise)

            K_tilde_terms = precision * (K_nn - tf.linalg.diag_part(tf.matmul(K_nm, tf.matmul(K_mm_inv, K_mn))))

            # k_i \cdot k_i^T, (b, m, m)
            lambda_mat = tf.matmul(tf.expand_dims(K_nm, axis=2),
                                   tf.transpose(tf.expand_dims(K_nm, axis=2), perm=[0, 2, 1]))

            # K_mm_inv \cdot k_i \cdot k_i^T \cdot K_mm_inv, (b, m, m)
            lambda_mat = tf.matmul(K_mm_inv, tf.matmul(lambda_mat, K_mm_inv))

            # Trace terms, (b,)
            trace_terms = precision * tf.trace(tf.matmul(S, lambda_mat))

            # L_3 sum part, (1,)
            L_3_sum_term = -0.5 * (tf.reduce_sum(K_tilde_terms) + tf.reduce_sum(trace_terms) +
                                   tf.reduce_sum(tf.log(noise)) + b * tf.cast(tf.log(2 * np.pi), dtype=self.dtype) +
                                   tf.reduce_sum(precision * (y - mean_vector) ** 2))

            return L_3_sum_term, KL_term

    def approximate_posterior_params(self, index_points, y=None, noise=None):
        """
        Computes parameters of q_S.

        :param index_points: points at which we want to evaluate posterior mean at
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP

        :return: posterior mean at index points (b, 1),
                 diagonal of posterior covariance matrix at index points (b, 1)
        """

        b = tf.cast(tf.shape(index_points)[0], dtype=self.dtype)

        # kernel matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_bb = self.kernel_matrix(index_points, index_points, x_inducing=False, y_inducing=False, diag_only=True)  # (b)
        K_bm = self.kernel_matrix(index_points, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mb = tf.transpose(K_bm, perm=[1, 0])  # (m, b)

        sigma_l = K_mm + (self.N_train / b) * tf.matmul(K_mb,
                                                        tf.matmul(
                                                            tf.linalg.diag(tf.math.reciprocal_no_nan(noise)),
                                                            K_bm))

        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        K_bm_Sigma_l_K_mb = tf.matmul(K_bm, tf.matmul(sigma_l_inv, K_mb))
        B = K_bb + tf.linalg.diag_part(-tf.matmul(K_bm, tf.matmul(K_mm_inv, K_mb)) + K_bm_Sigma_l_K_mb)

        mean_vector = (self.N_train / b) * tf.linalg.matvec(K_bm_Sigma_l_K_mb,
                                                            tf.math.reciprocal_no_nan(noise) * y)

        mu_hat = (self.N_train / b) * tf.linalg.matvec(tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mb)),
                                                       tf.math.reciprocal_no_nan(noise) * y)
        A_hat = tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mm))

        return mean_vector, B, mu_hat, A_hat

    def mean_vector_bias_analysis(self, index_points, y=None, noise=None):
        """
        Bias analysis (see C.4 in the Supplementary material).

        :param index_points: auxiliary data
        :param y: y vector of latent GP
        :param noise: noise vector of latent GP
        :return:
        """

        b = tf.cast(tf.shape(index_points)[0], dtype=self.dtype)

        # kernel matrices
        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_bm = self.kernel_matrix(index_points, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mb = tf.transpose(K_bm, perm=[1, 0])  # (m, b)

        # compute mean vector
        sigma_l = K_mm + (self.N_train / b) * tf.matmul(K_mb,
                                                        tf.matmul(
                                                            tf.linalg.diag(tf.math.reciprocal_no_nan(noise)),
                                                            K_bm))
        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = (self.N_train / b) * tf.linalg.matvec(tf.matmul(K_mm, tf.matmul(sigma_l_inv, K_mb)),
                                                            tf.math.reciprocal_no_nan(noise) * y)
        return mean_vector

    def approximate_posterior_params_precomputed_GP_posterior_params(self, index_points, mean_term, sigma_term,
                                                                     K_mm_inv=None):
        """
        Parameters of GP predictive posterior based on the entire train dataset.

        :param index_points:
        :param mean_term:
        :param sigma_term:
        :param K_mm_inv: precomputed inverse of inducing points kernel matrix.
                If None, it is computed inside this function.
        :return:
        """

        if K_mm_inv is None:
            K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
            K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)

        K_bb = self.kernel_matrix(index_points, index_points, x_inducing=False, y_inducing=False, diag_only=True)  # (b)
        K_bm = self.kernel_matrix(index_points, self.inducing_index_points, x_inducing=False)  # (b, m)
        K_mb = tf.transpose(K_bm, perm=[1, 0])  # (m, b)

        mean_vector = tf.linalg.matvec(K_bm, mean_term)
        B = K_bb + tf.linalg.diag_part(- tf.matmul(K_bm, tf.matmul(K_mm_inv, K_mb)) +
                                       tf.matmul(K_bm, tf.matmul(sigma_term, K_mb)))

        return mean_vector, B

    def approximate_posterior_predict(self, index_points_test, index_points_train=None, y=None, noise=None):
        """
        Use full train data to construct the GP predictive posterior.
        Then use the posterior to predict for test points.

        :param index_points_test: x
        :param index_points_train: N (in case Hensman posterior is used, we do not have to pass training points)
        :param y: y vector of latent GP, note that it is only needed in case of \mathcal{L}_2 (titsias)
        :param noise: noise vector of latent GP, note that it is only needed in case of \mathcal{L}_2 (titsias)

        :return: posterior mean at index points (b, 1),
                 posterior covariance matrix at index points (b, b)
        """

        K_mm = self.kernel_matrix(self.inducing_index_points, self.inducing_index_points)  # (m,m)
        K_mm_inv = tf.linalg.inv(_add_diagonal_jitter(K_mm, self.jitter))  # (m,m)
        K_xx = self.kernel_matrix(index_points_test, index_points_test, x_inducing=False,
                                  y_inducing=False, diag_only=True)  # (x)
        K_xm = self.kernel_matrix(index_points_test, self.inducing_index_points, x_inducing=False)  # (x, m)
        K_mx = tf.transpose(K_xm, perm=[1, 0])  # (m, x)

        K_nm = self.kernel_matrix(index_points_train, self.inducing_index_points, x_inducing=False)  # (N, m)
        K_mn = tf.transpose(K_nm, perm=[1, 0])  # (m, N)

        sigma_l = K_mm + tf.matmul(K_mn, tf.multiply(K_nm, tf.math.reciprocal_no_nan(noise)[:, tf.newaxis]))
        sigma_l_inv = tf.linalg.inv(_add_diagonal_jitter(sigma_l, self.jitter))
        mean_vector = tf.linalg.matvec(K_xm, tf.linalg.matvec(sigma_l_inv,
                                                              tf.linalg.matvec(K_mn, tf.math.reciprocal_no_nan(
                                                                  noise) * y)))

        K_xm_Sigma_l_K_mx = tf.matmul(K_xm, tf.matmul(sigma_l_inv, K_mx))
        B = K_xx + tf.linalg.diag_part(-tf.matmul(K_xm, tf.matmul(K_mm_inv, K_mx)) + K_xm_Sigma_l_K_mx)

        return mean_vector, B

    def variable_summary(self):
        """
        Returns values of parameters of sparse GP object. For debugging purposes.
        :return:
        """

        raise NotImplementedError()


class mnistSVGP(mainSVGP):

    def __init__(self, titsias, fixed_inducing_points, initial_inducing_points, fixed_gp_params,
                 object_vectors_init, name, jitter, N_train, L, K_obj_normalize):
        """
        SVGP class for rotated MNIST data.

        :param titsias: if true we use \mathcal{L}_2 (Titsias elbo). Else we use \mathcal{L}_3 (Hensman elbo).
        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param fixed_gp_params:
        :param object_vectors_init: initial value for object vectors (PCA embeddings).
                        If None, object vectors are fixed throughout training. GPLVM
        :param name: name (or index) of the latent channel
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L: number of latent channels used in SVGPVAE
        :param K_obj_normalize: whether or not to normalize object linear kernel
        """

        super(mnistSVGP, self).__init__(titsias=titsias, fixed_inducing_points=fixed_inducing_points,
                                        initial_inducing_points=initial_inducing_points,
                                        name=name, jitter=jitter,
                                        N_train=N_train, dtype=np.float64, L=L,
                                        K_obj_normalize=K_obj_normalize)

        # GP hyperparams
        if fixed_gp_params:
            self.l_GP = tf.constant(1.0, dtype=self.dtype)
            self.amplitude = tf.constant(1.0, dtype=self.dtype)
        else:
            self.l_GP = tf.Variable(initial_value=1.0, name="GP_length_scale_{}".format(name), dtype=self.dtype)
            self.amplitude = tf.Variable(initial_value=1.0, name="GP_amplitude_{}".format(name), dtype=self.dtype)

        # kernels
        self.kernel_view = tfk.ExpSinSquared(amplitude=self.amplitude, length_scale=self.l_GP, period=2*np.pi)
        self.kernel_object = tfk.Linear()

        # object vectors (GPLVM)
        if object_vectors_init is not None:
            self.object_vectors = tf.Variable(initial_value=object_vectors_init,
                                              name="GP_object_vectors_{}".format(name),
                                              dtype=self.dtype)
        else:
            self.object_vectors = None

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

        # this stays here as a reminder of a nasty, nasty bug...
        # x_inducing = tf.shape(x)[0] == self.nr_inducing
        # y_inducing = tf.shape(y)[0] == self.nr_inducing

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
                obj_norm = tf.math.reduce_euclidean_norm(x_object, axis=1) * tf.math.reduce_euclidean_norm(y_object, axis=1)
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


class spritesSVGP(mainSVGP):

    def __init__(self, titsias, fixed_inducing_points, initial_inducing_points,
                 name, jitter, N_train, L_action, initial_GPLVM_action, L, fixed_GP_params,
                 fixed_GPLVM_action=False, K_obj_normalize=False, K_tanh=False, K_SE=False):
        """
        SVGP class for rotated MNIST data.

        :param titsias: if true we use L_T (Titsias elbo). Else we use L_H (Hensman elbo).
        :param fixed_inducing_points:
        :param initial_inducing_points:
        :param name:
        :param jitter: jitter/noise for numerical stability
        :param N_train: number of training datapoints
        :param L_action: dimension of GPLVM action vectors
        :param initial_GPLVM_action: GPLVM action vectors
        :param fixed_GPLVM_action: if False GPLVM action vectors are jointly
            optimized along other SVGPVAE parameters. Else, they are fixed throughout training
        :param K_obj_normalize: whether or not to normalize object linear kernel
        :param L: number of latent channels used in SVGPVAE
        :param K_tanh: normalize a linear GP kernel using a tanh function
        :param K_SE: use the squared-exponential kernel instead of the linear kernel
        :param fixed_GP_params:
        """

        super(spritesSVGP, self).__init__(titsias=titsias, fixed_inducing_points=fixed_inducing_points,
                                          initial_inducing_points=initial_inducing_points,
                                          name=name, jitter=jitter,
                                          N_train=N_train,
                                          dtype=np.float32, K_obj_normalize=K_obj_normalize, L=L)

        self.L_action = L_action
        self.K_tanh = K_tanh
        self.K_SE = K_SE

        # action vectors (GPLVM)
        # u (inducing points)
        if fixed_GPLVM_action:
            self.GPLVM_action = tf.constant(initial_GPLVM_action, dtype=self.dtype)
        else:
            self.GPLVM_action = tf.Variable(initial_GPLVM_action, dtype=self.dtype,
                                            name='GP_GPLVM_action_vectors_'.format(name))

        # kernels
        if self.K_SE:
            if fixed_GP_params:
                self.l_action = tf.constant(1.0, dtype=self.dtype)
                self.sigma_action = tf.constant(0.1, dtype=self.dtype)
                self.l_character = tf.constant(1.0, dtype=self.dtype)
                self.sigma_character = tf.constant(0.1, dtype=self.dtype)
            else:
                self.l_action = tf.Variable(initial_value=1.0, name="GP_length_scale_action", dtype=np.float32)
                self.sigma_action = tf.Variable(initial_value=0.1, name="GP_amplitude_action", dtype=np.float32)
                self.l_character = tf.Variable(initial_value=1.0, name="GP_length_scale_character", dtype=np.float32)
                self.sigma_character = tf.Variable(initial_value=0.1, name="GP_amplitude_character", dtype=np.float32)

            self.kernel_action = tfk.ExponentiatedQuadratic(amplitude=self.sigma_action, length_scale=self.l_action)
            self.kernel_character = tfk.ExponentiatedQuadratic(amplitude=self.sigma_character,
                                                               length_scale=self.l_character)

        else:
            self.kernel_action = tfk.Linear()
            self.kernel_character = tfk.Linear()

    def kernel_matrix(self, x, y, x_inducing=True, y_inducing=True, diag_only=False):
        """
        Computes GP kernel matrix K(x,y). Product kernel between two linear kernels is used here.

        :param x:
        :param y:
        :param x_inducing: whether x is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param y_inducing: whether y is a set of inducing points (ugly but solution using tf.shape did not work...)
        :param diag_only: whether or not to only compute diagonal terms of the kernel matrix
        :return:
        """

        if x_inducing:
            x_action, x_character = x[:, :self.L_action], x[:, self.L_action:]
        else:
            x_action, x_character = tf.gather(self.GPLVM_action, tf.cast(x[:, 0], dtype=tf.int64)), x[:, 1:]

        if y_inducing:
            y_action, y_character = y[:, :self.L_action], y[:, self.L_action:]
        else:
            y_action, y_character = tf.gather(self.GPLVM_action, tf.cast(y[:, 0], dtype=tf.int64)), y[:, 1:]

        if diag_only:
            action_matrix = self.kernel_action.apply(x_action, y_action)
            character_matrix = self.kernel_character.apply(x_character, y_character)

            if not self.K_SE:  # normalize when linear kernels are used
                if self.K_obj_normalize:
                    action_norm = tf.math.reduce_euclidean_norm(x_action, axis=1) * \
                                  tf.math.reduce_euclidean_norm(y_action, axis=1)
                    action_matrix = action_matrix / action_norm

                    character_norm = tf.math.reduce_euclidean_norm(x_character, axis=1) * \
                                     tf.math.reduce_euclidean_norm(y_character, axis=1)
                    character_matrix = character_matrix / character_norm
                elif self.K_tanh:
                    action_matrix = tf.math.tanh(action_matrix)
                    character_matrix = tf.math.tanh(character_matrix)

        else:
            action_matrix = self.kernel_action.matrix(x_action, y_action)
            character_matrix = self.kernel_character.matrix(x_character, y_character)

            if not self.K_SE:
                if self.K_obj_normalize:  # normalize when linear kernels are used
                    action_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_action, axis=1, keepdims=True),
                                             tf.transpose(tf.math.reduce_euclidean_norm(y_action, axis=1, keepdims=True),
                                                          perm=[1, 0]))
                    action_matrix = action_matrix * action_norm

                    character_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_character, axis=1, keepdims=True),
                                         tf.transpose(tf.math.reduce_euclidean_norm(y_character, axis=1, keepdims=True),
                                                      perm=[1, 0]))
                    character_matrix = character_matrix * character_norm
                elif self.K_tanh:
                    action_matrix = tf.math.tanh(action_matrix)
                    character_matrix = tf.math.tanh(character_matrix)

        return action_matrix * character_matrix

    def variable_summary(self):
        """
        Returns values of parameters of sparse GP object. For debugging purposes.
        :return:
        """

        return self.GPLVM_action, self.inducing_index_points


def build_SVGPVAE_elbo_graph(vid_batch, beta, svgp_x, svgp_y, clipping_qs=False):
    """
        Builds SVGPVAE elbo for Pearce data.
        Returns pretty much everything!
        Args:
            vid_batch: tf variable (batch, tmax, px, py) binay arrays or images
            beta: scalar, tf variable, annealing term for prior KL
            svgp: SVGP object


        Returns:
            elbo: CPH elbo
            recon_err: reconstruction term
            KL_term: prior KL term
            full_p_mu: approx posterior mean
            full_p_var: approx post var
            qnet_mu: recognition network mean
            qnet_var: recog. net var
            pred_vid: reconstructed video
            globals(): aaaalll variables in local scope
        """

    batch, tmax, px, py = [int(s) for s in vid_batch.get_shape()]

    dt = vid_batch.dtype
    T = tf.range(tmax, dtype=dt) + 1.0  # to have range between 1-30 instead of 0-29
    batch_T = tf.concat([tf.reshape(T, (1, tmax)) for i in range(batch)], 0)

    # ENCODER NETWORK
    qnet_mu, qnet_var = build_MLP_inference_graph(vid_batch)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-6, 1e3)

    # approx posterior distribution
    p_m_x, p_v_x, mu_hat_x, A_hat_x = svgp_x.approximate_posterior_params(batch_T, y=qnet_mu[:, :, 0],
                                                                          noise=qnet_var[:, :, 0])
    p_m_y, p_v_y, mu_hat_y, A_hat_y = svgp_y.approximate_posterior_params(batch_T, y=qnet_mu[:, :, 1],
                                                                          noise=qnet_var[:, :, 1])

    # Inside-ELBO term (L_2 or L_3)
    inside_elbo_recon_x, inside_elbo_kl_x = svgp_x.variational_loss(batch_T, qnet_mu[:, :, 0],  qnet_var[:, :, 0],
                                                                    mu_hat=mu_hat_x, A_hat=A_hat_x)
    inside_elbo_recon_y, inside_elbo_kl_y = svgp_y.variational_loss(batch_T, qnet_mu[:, :, 1], qnet_var[:, :, 1],
                                                                    mu_hat=mu_hat_y, A_hat=A_hat_y)
    inside_elbo_recon = inside_elbo_recon_x + inside_elbo_recon_y
    inside_elbo_kl = inside_elbo_kl_x + inside_elbo_kl_y
    inside_elbo = inside_elbo_recon - inside_elbo_kl

    # added on 20.4., to investigate Cholesky vs diag conundrum
    gp_covariance_posterior_elemwise_mean_x = tf.reduce_mean(p_v_x, 0)
    gp_covariance_posterior_elemwise_mean_y = tf.reduce_mean(p_v_y, 0)

    full_p_mu = tf.stack([p_m_x, p_m_y], axis=2)
    full_p_var = tf.stack([tf.linalg.diag_part(p_v_x), tf.linalg.diag_part(p_v_y)], axis=2)

    # cross entropy term
    ce_term = gauss_cross_entropy(full_p_mu, full_p_var, qnet_mu, qnet_var)  # (batch, tmax, 2)
    ce_term = -tf.reduce_sum(ce_term, (1, 2))

    # latent samples
    epsilon = tf.random.normal(shape=(batch, tmax, 2))
    latent_samples = full_p_mu + epsilon * tf.sqrt(tf.clip_by_value(full_p_var, 1e-4, 1000))

    # reconstruction term
    pred_vid_batch_logits = build_MLP_decoder_graph(latent_samples, px, py)  # (batch, tmax, px, py)
    pred_vid = tf.nn.sigmoid(pred_vid_batch_logits)
    recon_term = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch, logits=pred_vid_batch_logits)
    recon_term = tf.reduce_sum(-recon_term, (1, 2, 3))  # (batch)

    KL_term = ce_term + inside_elbo
    CPH_elbo = recon_term + beta * KL_term

    return CPH_elbo, recon_term, KL_term, inside_elbo, ce_term, full_p_mu, full_p_var, qnet_mu, qnet_var, \
           pred_vid, svgp_x.l_GP, svgp_y.l_GP, \
           inside_elbo_recon, inside_elbo_kl, svgp_x.inducing_index_points, svgp_y.inducing_index_points, \
           gp_covariance_posterior_elemwise_mean_x, gp_covariance_posterior_elemwise_mean_y, globals()


def forward_pass_standard_VAE_rotated_mnist(data_batch, vae, sigma_gaussian_decoder=0.01,
                                            clipping_qs=False, CVAE=False):
    """
    Forward pass for SVGPVAE on rotated MNIST data. This is plain VAE forward pass (used in VAE-GP-joint
    training regime).

    :param data_batch:
    :param vae:
    :param sigma_gaussian_decoder: standard deviation of Gaussian decoder
    :param CVAE: run CVAE

    :return:
    """

    images, aux_data = data_batch

    _, w, h, c = images.get_shape()  # for MNIST c==1, for SPRITES c==3
    b = tf.shape(images)[0]

    if CVAE:  # add angles to input images
        sin_ = tf.reshape(tf.repeat(tf.math.sin(aux_data[:, 1]), tf.repeat(w * h, b)), shape=(b, w, h, 1))
        cos_ = tf.reshape(tf.repeat(tf.math.cos(aux_data[:, 1]), tf.repeat(w * h, b)), shape=(b, w, h, 1))
        images_cvae = tf.concat([images, sin_, cos_], axis=3)

    # ENCODER NETWORK
    if CVAE:
        qnet_mu, qnet_var = vae.encode(images_cvae, aux_data[:, 1])
    else:
        qnet_mu, qnet_var = vae.encode(images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=vae.dtype)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    # DECODER NETWORK
    # could consider CE loss as well here (then would have Bernoulli decoder), but for that would then need to adjust
    # range of beta param. Note that sigmoid only makes sense for Bernoulli decoder
    if CVAE:
        recon_images_logits = vae.decode(latent_samples, aux_data[:, 1])
    else:
        recon_images_logits = vae.decode(latent_samples)

    # Gaussian observational likelihood
    recon_images = recon_images_logits
    recon_loss = tf.reduce_sum((images - recon_images_logits) ** 2)

    # Bernoulli observational likelihood, CE
    # recon_images = tf.nn.sigmoid(recon_images_logits)
    # recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=images,
    #                                                                    logits=recon_images_logits))

    # ELBO (plain VAE)
    KL_term = KL_term_standard_normal_prior(qnet_mu, qnet_var, dtype=vae.dtype)

    elbo = -(0.5/sigma_gaussian_decoder**2)*recon_loss - KL_term

    # report MSE per pixel
    K = tf.cast(w, dtype=vae.dtype) * tf.cast(h, dtype=vae.dtype) * tf.cast(c, dtype=vae.dtype)
    recon_loss = recon_loss / K

    return recon_loss, KL_term, elbo, recon_images, qnet_mu, qnet_var, latent_samples


def predict_CVAE(images_train, images_test, aux_data_train, aux_data_test, vae, test_indices):
    """
    Prediction step for CVAE.

    :param images_train:
    :param images_test:
    :param aux_data_train:
    :param aux_data_test:
    :param vae:
    :param test_indices: since can not use tensor to range() in for loop we pass it as an argument
    :return:
    """

    _, w, h, _ = images_train.get_shape()
    N_train = tf.shape(images_train)[0]

    # encode train data
    sin_ = tf.reshape(tf.repeat(tf.math.sin(aux_data_train[:, 1]), tf.repeat(w * h, N_train)), shape=(N_train, w, h, 1))
    cos_ = tf.reshape(tf.repeat(tf.math.cos(aux_data_train[:, 1]), tf.repeat(w * h, N_train)), shape=(N_train, w, h, 1))
    images_train = tf.concat([images_train, sin_, cos_], axis=3)

    qnet_mu, qnet_var = vae.encode(images_train, aux_data_train[:, 1])
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=tf.float64)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    # average train latent samples for each test digit
    mean_latent_samples = []
    for test_id in test_indices:
        mask = tf.math.equal(aux_data_train[:, 0], test_id)
        mean_latent_samples.append(tf.expand_dims(tf.reduce_mean(tf.boolean_mask(latent_samples, mask=mask), axis=0), axis=0))
    mean_latent_samples = tf.concat(mean_latent_samples, axis=0)

    recon_images_test = vae.decode(mean_latent_samples, aux_data_test[:, 1])
    recon_loss = tf.reduce_mean((images_test - recon_images_test) ** 2)

    return recon_images_test, recon_loss


def forward_pass_SVGPVAE(data_batch, beta, vae, svgp, C_ma, lagrange_mult, alpha,
                         kappa, clipping_qs=False, GECO=False,
                         repr_NN=None, segment_ids=None, repeats=None, bias_analysis=False):
    """
    Forward pass for SVGPVAE on rotated MNIST data.

    :param data_batch: (images, aux_data). images dimension: (batch_size, 28, 28, 1).
        aux_data dimension: (batch_size, 10)
    :param beta:
    :param vae: VAE object
    :param svgp: SVGP object
    :param C_ma: average constraint from t-1 step (GECO)
    :param lagrange_mult: lambda from t-1 step (GECO)
    :param kappa: reconstruction level parameter for GECO
    :param alpha: moving average parameter for GECO
    :param clipping_qs: clipping of VAE posterior distribution (for numerical stability)
    :param GECO: whether or not to use GECO algorithm for training
    :param repr_NN: representation network (used only in case of SPRITES data)
    :param segment_ids: Used only in case of SPRITES data.
    :param repeats: Used only in case of SPRITES data.
    :param bias_analysis:

    :return:
    """

    images, aux_data = data_batch
    _, w, h, c = images.get_shape()
    K = tf.cast(w, dtype=vae.dtype) * tf.cast(h, dtype=vae.dtype) * tf.cast(c, dtype=vae.dtype)
    b = tf.cast(tf.shape(images)[0], dtype=vae.dtype)  # batch_size

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(images)
    L = tf.cast(qnet_mu.get_shape()[1], dtype=vae.dtype)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    if repr_NN:
        aux_data = aux_data_SVGPVAE_sprites(data_batch=data_batch, repr_nn=repr_NN,
                                            segment_ids=segment_ids, repeats=repeats)

    # SVGP: inside-ELBO term (L_2 or L_3), approx posterior distribution
    inside_elbo_recon, inside_elbo_kl = [], []
    p_m, p_v = [], []
    for l in range(qnet_mu.get_shape()[1]):  # iterate over latent dimensions
        p_m_l, p_v_l, mu_hat_l, A_hat_l = svgp.approximate_posterior_params(aux_data, qnet_mu[:, l], qnet_var[:, l])
        inside_elbo_recon_l,  inside_elbo_kl_l = svgp.variational_loss(x=aux_data, y=qnet_mu[:, l],
                                                                       noise=qnet_var[:, l], mu_hat=mu_hat_l,
                                                                       A_hat=A_hat_l)

        inside_elbo_recon.append(inside_elbo_recon_l)
        inside_elbo_kl.append(inside_elbo_kl_l)
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    inside_elbo_recon = tf.reduce_sum(inside_elbo_recon)
    inside_elbo_kl = tf.reduce_sum(inside_elbo_kl)

    if svgp.titsias:
        inside_elbo = inside_elbo_recon - inside_elbo_kl
    else:
        inside_elbo = inside_elbo_recon - (b / svgp.N_train) * inside_elbo_kl

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    if repr_NN:  # for numerical stability in SPRITES experiment
        p_v = tf.clip_by_value(p_v, 1e-4, 100)

    # cross entropy term
    ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
    ce_term = tf.reduce_sum(ce_term)

    KL_term = -ce_term + inside_elbo

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=vae.dtype)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # DECODER NETWORK
    recon_images_logits = vae.decode(latent_samples)
    recon_images = recon_images_logits

    if GECO:
        recon_loss = tf.reduce_mean((images - recon_images_logits) ** 2, axis=(1, 2, 3))
        recon_loss = tf.reduce_sum(recon_loss - kappa**2)
        C_ma = alpha * C_ma + (1 - alpha) * recon_loss / b

        elbo = - KL_term + lagrange_mult * (recon_loss/b + tf.stop_gradient(C_ma - recon_loss/b))

        lagrange_mult = lagrange_mult * tf.exp(C_ma)

    else:
        recon_loss = tf.reduce_sum((images - recon_images_logits) ** 2)

        # ELBO
        # beta plays role of sigma_gaussian_decoder here (\lambda(\sigma_y) in Casale paper)
        # K and L are not part of ELBO. They are used in loss objective to account for the fact that magnitudes of
        # reconstruction and KL terms depend on number of pixels (K) and number of latent GPs used (L), respectively
        recon_loss = recon_loss / K
        elbo = - recon_loss + (beta / L) * KL_term

    # bias analysis
    if bias_analysis:
        mean_vectors = []
        for l in range(qnet_mu.get_shape()[1]):
            mean_vectors.append(svgp.mean_vector_bias_analysis(aux_data, qnet_mu[:, l], qnet_var[:, l]))
    else:
        mean_vectors = tf.constant(1.0)  # dummy placeholder

    return elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, \
           recon_images, inside_elbo_recon, inside_elbo_kl, latent_samples, C_ma, lagrange_mult, mean_vectors


def batching_encode_SVGPVAE(data_batch, vae, clipping_qs=False, repr_nn=None,
                            segment_ids=None, repeats=None):
    """
    This function encodes images to latent representations in batches for SVGPVAE model.
    
    :param data_batch:
    :param vae:
    :param clipping_qs:
    :param repr_nn: representation network. used only in case of SPRITES data
    :param segment_ids: used only in case of SPRITES data
    :param repeats: used only in case of SPRITES data
    :return: 
    """

    images, aux_data = data_batch

    b = tf.shape(images)[0]

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    if repr_nn:
        aux_data_train = aux_data_SVGPVAE_sprites(data_batch=data_batch, repr_nn=repr_nn,
                                                  segment_ids=segment_ids, repeats=repeats)

        return qnet_mu, qnet_var, aux_data_train
    else:
        return qnet_mu, qnet_var


def batching_encode_SVGPVAE_full(train_images, vae, clipping_qs=False):
    """
    This function encodes images to latent representations in batches for SVGPVAE model at once.

    :param data_batch:
    :return:
    """

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(train_images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    return qnet_mu, qnet_var


def precompute_GP_params_SVGPVAE(means, vars, aux_data, svgp):
    """
    This function computes mean vector and inverse of \Sigma_l for GP posterior (for each latent dim).
    Used in test pipelines for SVGPVAE for SPRITES.

    :param means: matrix of encoded means of data (N, L)
    :param vars: matrix of encoded vars of data (N, L)
    :param aux_data: auxiliary data (N, 10) or (N, 1 + L_character)
    :param svgp:

    :return: mean term (L, m), inverse of Sigma_l (L, m, m)

    """
    L = means.get_shape()[1]

    K_mm = svgp.kernel_matrix(svgp.inducing_index_points, svgp.inducing_index_points)  # (m,m)
    K_nm = svgp.kernel_matrix(aux_data, svgp.inducing_index_points, x_inducing=False)  # (n, m)
    K_mn = tf.transpose(K_nm, perm=[1, 0])  # (m, b)

    mean_terms, inv_Sigma_l_mats = [], []
    for l in range(L):
        # [17.8.2020, update] not converting noise vector to diagonal matrix anymore. In case of SPRITES data,
        #   it lead to memory issues (50000 x 50000 matrix...)
        # Sigma_l = K_mm + tf.matmul(K_mn, tf.matmul(tf.linalg.diag(tf.math.reciprocal_no_nan(vars[:, l])), K_nm))
        Sigma_l = K_mm + tf.matmul(K_mn, tf.multiply(K_nm, tf.math.reciprocal_no_nan(vars[:, l])[:, tf.newaxis]))
        Sigma_l_inv = tf.linalg.inv(Sigma_l)
        mean_term_l = tf.linalg.matvec(Sigma_l_inv, tf.linalg.matvec(K_mn,
                                       tf.math.multiply(tf.math.reciprocal_no_nan(vars[:, l]), means[:, l])))
        mean_terms.append(mean_term_l)
        inv_Sigma_l_mats.append(Sigma_l_inv)

    mean_terms = tf.stack(mean_terms, axis=0)
    inv_Sigma_l_mats = tf.stack(inv_Sigma_l_mats, axis=0)

    return mean_terms, inv_Sigma_l_mats


def bacthing_predict_SVGPVAE_rotated_mnist(test_data_batch, vae, svgp,
                                           qnet_mu, qnet_var, aux_data_train):
    """
    Get predictions for test data. See chapter 3.3 in Casale's paper.
    This version supports batching in prediction pipeline (contrary to function predict_SVGPVAE_rotated_mnist) .

    :param test_data_batch: batch of test data
    :param vae: fitted (!) VAE object
    :param svgp: fitted (!) SVGP object
    :param qnet_mu: precomputed encodings (means) of train dataset (N_train, L)
    :param qnet_var: precomputed encodings (vars) of train dataset (N_train, L)
    :param aux_data_train: train aux data (N_train, 10)
    :return:
    """

    images_test_batch, aux_data_test_batch = test_data_batch

    _, w, h, _ = images_test_batch.get_shape()

    # get latent samples for test data from GP posterior
    p_m, p_v = [], []
    for l in range(qnet_mu.get_shape()[1]):  # iterate over latent dimensions
        p_m_l, p_v_l = svgp.approximate_posterior_predict(index_points_test=aux_data_test_batch,
                                                          index_points_train=aux_data_train,
                                                          y=qnet_mu[:, l], noise=qnet_var[:, l])
        # p_v_l = tf.linalg.diag_part(p_v_l)  # no need for Cholesky, remember email correspondence with Michael
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=tf.float64)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

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


def aux_data_SVGPVAE_sprites(data_batch, repr_nn, segment_ids, repeats):
    """
    Generates auxiliary data for images in the current batch for SPRITES data:
    - pass all frames through repr NN and sum representation for each character
    - for actions store IDs for each frame (look-up to GPLVM table is done inside kernel_matrix function)

    :param data_batch:
    :param repr_nn:
    :param segment_ids: array, which frame belongs to which character
    :param repeats: array, how many times is each summed character vector repeated

    :return: auxiliary data (b, 1 + L_character)
    """

    images, action_IDs = data_batch

    # representation network forward pass
    character_vectors = repr_nn.repr_nn(images)

    # sum representations for each character
    # character_vectors = tf.segment_sum(character_vectors, segment_ids=segment_ids)
    character_vectors = tf.segment_mean(character_vectors, segment_ids=segment_ids)

    # copy summed representations
    character_vectors = tf.repeat(character_vectors, repeats=repeats, axis=0)

    # add action IDs
    aux_data = tf.concat([tf.expand_dims(tf.cast(action_IDs, dtype=tf.float32), axis=1), character_vectors], axis=1)

    return aux_data


def predict_SVGPVAE_sprites_test_character(data_batch, vae, svgp, repr_NN, mean_terms,
                                           var_terms, N_context, N_actions, batch_size_test,
                                           segment_ids, repeats, K_mm_inv):
    """
    Conditional generation prediction function for test_character SPRITES data.

    :param data_batch:
    :param vae:
    :param svgp:
    :param repr_NN:
    :param mean_terms: mean terms precomputed using precompute_GP_params_SVGPVAE function (L, m)
    :param var_terms: var terms precomputed using precompute_GP_params_SVGPVAE function (L, m, m)
    :param N_context: number of context frames per character
    :param N_actions:
    :param batch_size_test:
    :param segment_ids:
    :param repeats:
    :param K_mm_inv: precomputed inverse of inducing points kernel matrix
    :return:
    """
    
    images, aux_data = data_batch
    _, w, h, c = images.get_shape()

    # split into context and target frames
    # (here we use the fact that test_char batch size is 576 and hence N_test_character % 576 == 0)
    context = np.sort(np.array([list(i*N_actions+np.random.choice(range(N_actions), N_context, replace=False)) for i
                                in range(int(batch_size_test/N_actions))]).reshape(-1))
    target = np.array([x for x in range(batch_size_test) if x not in list(context)])

    images_context, images_target = tf.gather(images, context), tf.gather(images, target)
    _, aux_data_target = tf.gather(aux_data, context), tf.gather(aux_data, target)

    # generate aux_data for target frames
    aux_data_target = aux_data_SVGPVAE_sprites(data_batch=(images_context, aux_data_target), repr_nn=repr_NN,
                                               segment_ids=segment_ids, repeats=repeats)

    # get latent samples for test data from GP posterior
    p_m, p_v = [], []
    for l in range(vae.L):  # iterate over latent dimensions
        p_m_l, p_v_l = svgp.approximate_posterior_params_precomputed_GP_posterior_params(index_points=aux_data_target,
                                                                                         mean_term=mean_terms[l, :],
                                                                                         sigma_term=var_terms[l, :, :],
                                                                                         K_mm_inv=K_mm_inv)
        p_m.append(p_m_l)
        p_v.append(p_v_l)

    p_m = tf.stack(p_m, axis=1)
    p_v = tf.stack(p_v, axis=1)

    # TODO: in case of SPRITES data p_v has negative values, which necessitates clipping below.
    #  p_v should in theory attain only non-negative values, understand why this is not the case for SPRITES data!
    p_v = tf.clip_by_value(p_v, 1e-4, 100)

    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=vae.dtype)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # predict (decode) latent images.
    recon_images_test_logits = vae.decode(latent_samples)
    recon_images_test = recon_images_test_logits

    # MSE loss for CGEN
    recon_loss = tf.reduce_sum((images_target - recon_images_test_logits) ** 2)

    # report per pixel loss
    # (note that we do not divide it by the number of target frames - that is done
    # later in the session at the end of test epoch)
    K = tf.cast(w, dtype=vae.dtype) * tf.cast(h, dtype=vae.dtype) * tf.cast(c, dtype=vae.dtype)
    recon_loss = recon_loss / K

    return recon_images_test, images_target, recon_loss, tf.gather(aux_data, target), aux_data_target


def predict_SVGPVAE_sprites_test_action(data_batch, vae, svgp, qnet_mu, qnet_var, aux_data_train):
    """
    Conditional generation prediction function for test_action SPRITES data.

    :param data_batch:
    :param vae:
    :param svgp:
    :param qnet_mu:
    :param qnet_var:
    :param aux_data_train:
    :return:
    """
    raise NotImplementedError()
