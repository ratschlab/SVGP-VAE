import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import pickle

tfk = tfp.math.psd_kernels


def tf_kron(a, b):
    """
    Computes Kronecker product between two matrices.

    :param a:
    :param b:
    :return:
    """
    a_shape = tf.shape(a)  # [a.shape[0].value, a.shape[1].value]
    b_shape = tf.shape(b)  # [b.shape[0].value, b.shape[1].value]
    return tf.reshape(tf.reshape(a, [a_shape[0], 1, a_shape[1], 1]) * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
                      [a_shape[0]*b_shape[0], a_shape[1]*b_shape[1]])


def train_angles_mask(data_path, save_path):
    """
    Computes mask for subsampling matrix V from "Efficient GP computations chapter" in Casale's paper.

    :param data_path:
    :param save_path:
    :return:
    """

    train_data = pickle.load(open(data_path, 'rb'))
    train_angles = [np.sort(train_data['aux_data'][np.where((train_data['aux_data'][:, 0] == x))][:, 1]) for x in
                    np.sort(np.unique(train_data['aux_data'][:, 0]))]
    train_angles_unique = np.sort(np.unique(train_data['aux_data'][:, 1]))
    train_ids_mask = np.array([x in y for y in train_angles for x in train_angles_unique])
    pickle.dump(train_ids_mask, open(save_path, "wb"))

    return


def sort_train_data(train_data_dict, dataset='3'):
    """
    Sorts train data so that indexing is compatible with indexing in V_matrix function in casaleGP class.
    Also adds global id column to aux_data array.

    :param train_data_dict:

    :return: sorted train data
    """

    images, aux_data = train_data_dict['images'], train_data_dict['aux_data']
    N = len(aux_data)

    sorted_idx = sorted(list(zip(aux_data[:, 0], aux_data[:, 1], range(N))), key=lambda x: (x[0], x[1]))
    sorted_idx = [x[2] for x in sorted_idx]

    # sort aux_data and add id column
    aux_data = aux_data[sorted_idx]
    train_data_dict['aux_data'] = np.hstack((np.expand_dims(np.array(range(4050 * len(dataset))), axis=1), aux_data))

    # sort images
    train_data_dict['images'] = images[sorted_idx]

    return train_data_dict


def encode(train_images, vae, clipping_qs=False, batch=False):
    """
    Encode entire train dataset: pass through inference network and sample to obtain latent vectors z.

    :param train_images:
    :param vae:
    :param clipping_qs:
    :return:
    """

    # ENCODER NETWORK
    if batch:
        qnet_mu, qnet_var = vae.encode(train_images[0])
    else:
        qnet_mu, qnet_var = vae.encode(train_images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # sample
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=tf.float64)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    return latent_samples


def forward_pass_Casale(data_batch, vae, a, B, c, V, beta, GP, clipping_qs=False):
    """
    Forward pass for Casale's GP-VAE model.

    :param data_batch:
    :param vae:
    :param a:
    :param B:
    :param c:
    :param V:
    :param beta:
    :param clipping_qs:

    :return:
    """

    images, aux_data = data_batch
    batch_idx = aux_data[:, 0]

    _, w, h, _ = images.get_shape()
    K = tf.cast(w, dtype=tf.float64) * tf.cast(h, dtype=tf.float64)

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(images)
    L = tf.cast(qnet_mu.get_shape()[1], dtype=tf.float64)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 100)

    # log variance term
    log_var = tf.reduce_sum(tf.log(qnet_var))

    # sample z
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=tf.float64)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    # GP prior term
    a_batch = tf.gather(tf.transpose(a, perm=[1, 0]), tf.cast(batch_idx, dtype=tf.int64))
    B_batch = tf.gather(tf.transpose(B, perm=[1, 2, 0]), tf.cast(batch_idx, dtype=tf.int64))
    V_batch = tf.gather(V, tf.cast(batch_idx, dtype=tf.int64))

    B_terms = []
    for l in range(qnet_mu.get_shape()[1]):
        B_terms.append(tf.reduce_sum(B_batch[:, :, l] * V_batch))

    GP_prior_term = tf.reduce_sum(latent_samples*a_batch) + tf.reduce_sum(B_terms) + tf.reduce_sum(c)*GP.alpha

    # recon term
    recon_images_logits = vae.decode(latent_samples)
    recon_images = recon_images_logits
    recon_loss = tf.reduce_sum((images - recon_images_logits) ** 2)

    # ELBO/loss objective (see eq. (18) in appendix)
    elbo = recon_loss / K - (beta / L) * (GP_prior_term + 0.5*log_var)

    # report per pixel MSE
    recon_loss = recon_loss / K

    return elbo, recon_loss, GP_prior_term, log_var, qnet_mu, qnet_var, recon_images


def predict_test_set_Casale(test_images, test_aux_data, train_aux_data, vae, GP, V,
                            latent_samples_train, take_mean=False):
    """

    :param test_images:
    :param test_aux_data:
    :param train_aux_data:
    :param vae:
    :param GP:
    :param V:
    :param latent_samples_train:
    :param take_mean: if True, mean of GP predictive posterior is taken for z^* (note that in this case, there is no
                      need to compute variance of GP predictive posterior). If False, we sample from GP predictive
                      posterior to obtain z^*.
    :return:

    """
    N = tf.shape(V)[0]
    H = tf.shape(V)[1]
    L = vae.L

    # K matrices
    K_test_train = GP.kernel_matrix(test_aux_data, train_aux_data[:, 1:])
    K_test_test = GP.kernel_matrix(test_aux_data, test_aux_data)
    inside_inv = tf.linalg.inv(GP.alpha * tf.eye(H, dtype=tf.float64) + tf.matmul(tf.transpose(V, perm=[1, 0]), V))
    K_inv = (1 / GP.alpha) * tf.eye(N, dtype=tf.float64) - (1 / GP.alpha) * \
            tf.matmul(V, tf.matmul(inside_inv, tf.transpose(V, perm=[1, 0])))

    # predictive posterior GP
    # TODO: figure out if I should take mean here for latent_samples_test or do I sample
    mean = tf.linalg.matmul(K_test_train, tf.linalg.matmul(K_inv, latent_samples_train))
    if take_mean:
        latent_samples_test = mean
    else:
        var = K_test_test - tf.linalg.matmul(K_test_train, tf.linalg.matmul(K_inv, tf.transpose(K_test_train, perm=[1, 0])))
        var = tf.diag_part(var)  # note how here variance is the same across all latent channels, contrary to SVGPVAE
        var = tf.reshape(tf.tile(var, [L]), (-1, L))

        epsilon = tf.random.normal(shape=tf.shape(mean), dtype=tf.float64)
        latent_samples_test = mean + epsilon * tf.sqrt(var)

    # decode latent samples to images
    recon_images_test = vae.decode(latent_samples_test)
    recon_loss = tf.reduce_mean((test_images - recon_images_test) ** 2)

    return recon_images_test, recon_loss


class casaleGP:

    dtype = np.float64

    def __init__(self, fixed_gp_params, object_vectors_init, object_kernel_normalize, ov_joint, jitter=1e-6):
        """
        GP class for rotated MNIST data, based on Casale's paper.
        :param fixed_gp_params:
        :param object_vectors_init: initial value for object vectors (PCA embeddings).
                        If None, object vectors are fixed throughout training
        :param object_kernel_normalize: whether or not to normalize object kernel (linear kernel)
        :param ov_joint: whether or not to jointly optimize object vectors (GPLVM)
        :param jitter: jitter/noise for numerical stability
        """

        self.jitter = jitter
        self.object_kernel_normalize = object_kernel_normalize
        self.ov_joint = ov_joint

        # GP hyperparams
        if fixed_gp_params:
            self.l_GP = tf.constant(1.0, dtype=self.dtype)
            self.amplitude = tf.constant(1.0, dtype=self.dtype)
            self.alpha = tf.constant(0.1, dtype=self.dtype)
        else:
            self.l_GP = tf.Variable(initial_value=1.0, name="GP_length_scale", dtype=self.dtype)
            self.amplitude = tf.Variable(initial_value=1.0, name="GP_amplitude", dtype=self.dtype)
            self.alpha = tf.Variable(initial_value=0.1, name="GP_alpha", dtype=self.dtype)

        # object vectors
        if self.ov_joint:
            self.object_vectors = tf.Variable(initial_value=object_vectors_init,
                                              name="GP_object_vectors",
                                              dtype=self.dtype)
        else:
            self.object_vectors = tf.constant(value=object_vectors_init,
                                              name="GP_object_vectors",
                                              dtype=self.dtype)

        # kernels
        self.kernel_view = tfk.ExpSinSquared(amplitude=self.amplitude, length_scale=self.l_GP, period=2 * np.pi)
        self.kernel_object = tfk.Linear()

    def kernel_matrix(self, x, y):
        """
        Computes product kernel matrix. Product kernel from Casale's paper is used.
        :param x:
        :param y:

        :return:
        """

        # unpack auxiliary data
        if not self.ov_joint:
            x_view, x_object, y_view, y_object = x[:, 1], x[:, 2:], y[:, 1], y[:, 2:]
        else:
            x_view, y_view = x[:, 1], y[:, 1]
            x_object = tf.gather(self.object_vectors, tf.cast(x[:, 0], dtype=tf.int64))
            y_object = tf.gather(self.object_vectors, tf.cast(y[:, 0], dtype=tf.int64))

        # compute kernel matrix
        view_matrix = self.kernel_view.matrix(tf.expand_dims(x_view, axis=1), tf.expand_dims(y_view, axis=1))

        object_matrix = self.kernel_object.matrix(x_object, y_object)
        if self.object_kernel_normalize:  # normalize object matrix
            obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_object, axis=1, keepdims=True),
                                     tf.transpose(tf.math.reduce_euclidean_norm(y_object, axis=1, keepdims=True),
                                                  perm=[1, 0]))
            object_matrix = object_matrix * obj_norm

        return view_matrix * object_matrix

    def V_matrix(self, aux_data_train, train_ids_mask):
        """
        Calculates V (N x H) matrix, see chapter Efficient GP computations in Casale's paper.
        H = P \cdot Q

        :param aux_data_train: auxiliary data for train dataset (N x (1 + R + M))
        :param train_ids_mask: mask for which angles we have images in the train data for object vectors,
                               generated using function train_angles_mask. Used after Kronecker product, in
                               subsampling of V.

        :return: V (N X H)
        """

        # drop global id column from aux_data
        aux_data_train = aux_data_train[:, 1:]

        # construct \Tilde{V}
        train_object_ids = tf.sort(tf.unique(aux_data_train[:, 0]).y)
        train_angles_unique = tf.expand_dims(tf.sort(tf.unique(aux_data_train[:, 1]).y), 1)
        object_vectors = tf.gather(self.object_vectors, tf.cast(train_object_ids, dtype=tf.int64))
        if self.object_kernel_normalize:
            object_vectors = object_vectors / tf.math.reduce_euclidean_norm(object_vectors, axis=1, keepdims=True)

        K_W = self.kernel_view.matrix(train_angles_unique, train_angles_unique)
        L_W = tf.linalg.cholesky(K_W)

        V = tf_kron(object_vectors, L_W)

        # subsample only rows belonging to train indices from \Tilde{V}
        V = tf.boolean_mask(V, train_ids_mask)

        return V

    def taylor_coeff(self, Z, V):
        """
        Computes coefficients of first-order Taylor expansion, see
        "Implementation of low-memory stochastic backpropagation" chapter in the appendix of Casale's paper.

        :param Z: latent vectors (N x L)
        :param V: V matrix (N x H)

        :return:
            - a (L x N)
            - B (L x N x H)
            - c (L)

        """
        N = tf.cast(tf.shape(V)[0], dtype=tf.float64)
        H = tf.cast(tf.shape(V)[1], dtype=tf.float64)
        L = Z.get_shape()[1]

        # K inverse
        inside_inv = tf.linalg.inv(self.alpha*tf.eye(H, dtype=tf.float64) + tf.matmul(tf.transpose(V, perm=[1, 0]), V))
        K_inv = (1/self.alpha) * tf.eye(N, dtype=tf.float64) - (1/self.alpha) * \
                    tf.matmul(V, tf.matmul(inside_inv, tf.transpose(V, perm=[1, 0])))

        # a
        a = tf.matmul(tf.transpose(Z, perm=[1, 0]), K_inv)

        # B, c
        K_inv_V = tf.matmul(K_inv, V)
        B = []
        c = []

        for l in range(L):
            z = tf.expand_dims(Z[:, l], 1)
            z_T = tf.transpose(z, perm=[1, 0])
            B.append(- tf.matmul(K_inv, tf.matmul(z, tf.matmul(z_T, K_inv_V))) + K_inv_V)
            c.append(0.5*(- tf.matmul(z_T, tf.matmul(K_inv, tf.matmul(K_inv, z))) + tf.trace(K_inv)))

        B = tf.stack(B, axis=0)
        c = tf.squeeze(c)

        return a, B, c

    def variable_summary(self):
        """
        Returns values of parameters of GP object. For debugging purposes.
        :return:
        """

        return self.l_GP, self.amplitude, self.object_vectors, self.alpha


if __name__ == "__main__":

    # train_angles_mask("MNIST data/train_data13679.p", "MNIST data/train_ids_mask13679.p")
    # train_angles_mask("MNIST data/train_data3_4.p", "MNIST data/train_ids_mask3_4.p")
    # train_angles_mask("MNIST data/train_data3_16.p", "MNIST data/train_ids_mask3_16.p")
    # train_angles_mask("MNIST data/train_data3_32.p", "MNIST data/train_ids_mask3_32.p")
    # train_angles_mask("MNIST data/train_data3_64.p", "MNIST data/train_ids_mask3_64.p")
    train_angles_mask("MNIST data/train_data3_24.p", "MNIST data/train_ids_mask3_24.p")

