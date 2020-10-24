import tensorflow as tf
import numpy as np

from VAE_utils import build_MLP_inference_graph, build_MLP_decoder_graph
from utils import gauss_cross_entropy


def build_1d_gp(X, Y, varY, X_test, lt=5, full_variance=False,
                GP_joint=False, GP_init=2.0):
    """
    Takes input-output dataset and returns post mean, var, marginal lhood.
    This is standard GP regression (in this application X is time, Y is
    recognition network means with noise as recognition network variance).

    Args:
        X: inputs tensor (batch, npoints)
        Y: outputs tensor (batch, npoints)
        varY: noise of outputs tensor (batch, npoints)
        X_test: (batch, ns) input points to compute post mean + var
        lt: length scale of RBF kernel. Added for purposes of joint GP optimization
        GP_joint: whether or not to do joint optimization of GP kernel parameters
        GP_init: init value for GP kernel hyperparameter, used in case of GP_joint=True

    Returns:
        p_m: (batch, ns) post mean at X_test
        p_v: (batch, ns) post var at X_test
        logZ: (batch) marginal lhood of each dataset in batch
    """

    # Prepare all constants
    batch, _ = X.get_shape()
    n = tf.shape(X)[1]
    _, ns = X_test.get_shape()

    # inverse square length scale
    if GP_joint:  # jointly optimize GP params
        # l_GP = tf.Variable(initial_value=tf.random.uniform(shape=(1,), minval=1, maxval=5),
        #                    name="GP_length_scale", trainable=True)
        l_GP = tf.Variable(initial_value=GP_init,
                           name="GP_length_scale", trainable=True)
    else:
        l_GP = tf.constant(lt, dtype=tf.float32)

    ilt = -0.5 * (1 / (l_GP * l_GP))

    # lhood term 1/3
    lhood_pi_term = tf.cast(n, dtype=tf.float32) * np.log(2 * np.pi)

    # data cov matrix K = exp( -1/2 * (X-X)**2/l**2) + noise
    K = tf.reshape(X, (batch, n, 1)) - tf.reshape(X, (batch, 1, n))  # (batch, n n)
    K = tf.exp((K ** 2) * ilt) + tf.matrix_diag(
        varY)
    chol_K = tf.linalg.cholesky(K)  # (batch, n, n)

    # lhood term 2/3
    lhood_logdet_term = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_K)), 1)  # (batch)

    # lhood term 3/3
    Y = tf.reshape(Y, (batch, n, 1))
    iKY = tf.cholesky_solve(chol_K, Y)  # (batch, n, 1)
    lh_quad_term = tf.matmul(tf.reshape(Y, (batch, 1, n)), iKY)  # (batch, 1, 1)
    lh_quad_term = tf.reshape(lh_quad_term, [batch])

    # log P(Y|X) = -1/2 * ( n log(2 pi) + Y inv(K+noise) Y + log det(K+noise))
    gp_lhood = -0.5 * (lhood_pi_term + lh_quad_term + lhood_logdet_term)

    # Compute posterior mean and variances
    Ks = tf.reshape(X, (batch, n, 1)) - tf.reshape(X_test, (batch, 1, ns))  # broadcasts to (batch, n, ns)
    Ks = tf.exp((Ks ** 2) * ilt)  # (batch, n, ns)
    Ks_t = tf.transpose(Ks, (0, 2, 1))  # (batch, ns, n)

    # posterior mean
    p_m = tf.matmul(Ks_t, iKY)
    p_m = tf.reshape(p_m, (batch, ns))

    # posterior variance
    if full_variance:
        # this implementation holds for case X=X_test only!
        p_v = Ks - tf.matmul(Ks, tf.matmul(tf.linalg.inv(K), Ks))
    else:
        iK_Ks = tf.cholesky_solve(chol_K, Ks)  # (batch, n, ns)
        Ks_iK_Ks = tf.reduce_sum(Ks * iK_Ks, axis=1)  # (batch, ns)
        p_v = 1 - Ks_iK_Ks  # (batch, ns)
        p_v = tf.reshape(p_v, (batch, ns))

    return p_m, p_v, gp_lhood, l_GP


def build_pearce_elbo_graphs(vid_batch, beta, type_elbo="GPVAE_Pearce", lt=5, context_ratio=0.5,
                             GP_joint=False, GP_init=2.0):
    """
    Builds standard (GPVAE_Pearce) eblo or neural process (NP) elbo for Pearce data.
    Returns pretty much everything!
    Args:
        vid_batch: tf variable (batch, tmax, px, py) binay arrays or images
        beta: scalar, tf variable, annealing term for prior KL
        type_elbo: standard (GPVAE_Pearce) or neural process (NP)
        lt: length scale of GP
        context_ratio: float in [0,1], for np elbo, random target-context split ratio
        GP_joint: whether or not to do joint optimization of GP kernel parameters
        GP_init: init value for GP kernel hyperparameter, used in case of GP_joint=True

    Returns:
        elbo: "standard" elbo
        elbo_recon: recon struction term
        elbo_prior_kl: prior KL term
        full_p_mu: approx posterior mean
        full_p_var: approx post var
        qnet_mu: recognition network mean
        qnet_var: recog. net var
        pred_vid: reconstructed video
        globals(): aaaalll variables in local scope
    """

    batch, tmax, px, py = [int(s) for s in vid_batch.get_shape()]

    dt = vid_batch.dtype
    T = tf.range(tmax, dtype=dt)
    batch_T = tf.concat([tf.reshape(T, (1, tmax)) for i in range(batch)], 0)  # (batch, tmax)

    # recognition network terms
    qnet_mu, qnet_var = build_MLP_inference_graph(vid_batch)

    if type_elbo == 'NP':
        # Choose a random split of target-context for each batch
        con_tf = tf.random.normal(shape=(),
                                  mean=context_ratio*float(tmax),
                                  stddev=np.sqrt(context_ratio*(1-context_ratio)*float(tmax)))
        con_tf = tf.math.maximum(con_tf, 2)
        con_tf = tf.math.minimum(con_tf, int(tmax)-2)
        con_tf = tf.cast(tf.round(con_tf), tf.int32)



        ##################################################################
        ####################### CONTEXT LIKELIHOOD #######################
        # make random indices
        ran_ind = tf.range(tmax, dtype=tf.int32)
        ran_ind = [tf.random.shuffle(ran_ind) for i in range(batch)] # (batch, tmax)
        ran_ind = [tf.reshape(r_i, (1,tmax)) for r_i in ran_ind] # len batch list( (tmax), ..., (tmax) )
        ran_ind = tf.concat(ran_ind, 0) # ()

        con_ind = ran_ind[:, :con_tf]
        tar_ind = ran_ind[:, con_tf:]

        # time stamps of context points
        con_T = [tf.gather(T, con_ind[i,:]) for i in range(batch)]
        con_T = [tf.reshape(ct, (1,con_tf)) for ct in con_T]
        con_T = tf.concat(con_T, 0)

        # encoded means of contet points
        con_lm = [tf.gather(qnet_mu[i,:,:], con_ind[i,:], axis=0) for i in range(batch)]
        con_lm = [tf.reshape(cm, (1,con_tf,2)) for cm in con_lm]
        con_lm = tf.concat(con_lm, 0)

        # encoded variances of context points
        con_lv = [tf.gather(qnet_var[i,:,:], con_ind[i,:], axis=0) for i in range(batch)]
        con_lv = [tf.reshape(cv, (1,con_tf,2)) for cv in con_lv]
        con_lv = tf.concat(con_lv, 0)

        # conext Lhoods
        _,_, con_lhoodx, _ = build_1d_gp(con_T, con_lm[:,:,0], con_lv[:,:,0], batch_T, lt=lt)
        _,_, con_lhoody, _ = build_1d_gp(con_T, con_lm[:,:,1], con_lv[:,:,1], batch_T, lt=lt)
        con_lhood = con_lhoodx + con_lhoody


    ####################################################################################
    #################### PriorKL 1/3: FULL APPROX POST AND LIKELIHOOD ##################

    # posterior and lhood for full dataset
    p_mx, p_vx, full_lhoodx, l_GP_x = build_1d_gp(batch_T, qnet_mu[:, :, 0], qnet_var[:, :, 0], batch_T, lt=lt,
                                                  GP_joint=GP_joint, GP_init=GP_init)
    p_my, p_vy, full_lhoody, l_GP_y = build_1d_gp(batch_T, qnet_mu[:, :, 1], qnet_var[:, :, 1], batch_T, lt=lt,
                                                  GP_joint=GP_joint, GP_init=GP_init)

    full_p_mu = tf.stack([p_mx, p_my], axis=2)
    full_p_var = tf.stack([p_vx, p_vy], axis=2)

    full_lhood = full_lhoodx + full_lhoody

    ####################################################################################
    ########################### PriorKL 2/3: CROSS ENTROPY TERMS #######################

    # cross entropy term
    sin_elbo_ce = gauss_cross_entropy(full_p_mu, full_p_var, qnet_mu, qnet_var) #(batch, tmax, 2)
    sin_elbo_ce = tf.reduce_sum(sin_elbo_ce, 2) # (batch, tmax)

    if type_elbo == 'NP':
        np_elbo_ce = [tf.gather(sin_elbo_ce[i,:], tar_ind[i,:]) for i in range(batch)] # (batch, con_tf)
        np_elbo_ce = [tf.reduce_sum(np_i) for np_i in np_elbo_ce] # list of scalars, len=batch

        np_elbo_ce = tf.stack(np_elbo_ce) # (batch)

    ####################################################################################
    ################################ Prior KL 3/3 ######################################

    if type_elbo == 'GPVAE_Pearce' or type_elbo == 'VAE':
        sin_elbo_ce = tf.reduce_sum(sin_elbo_ce, 1)  # (batch)
        elbo_prior_kl = full_lhood - sin_elbo_ce
        # omitting GP marginal likelihood from ELBO objective
        # elbo_prior_kl = - sin_elbo_ce
    elif type_elbo == 'NP':
        elbo_prior_kl = full_lhood - np_elbo_ce - con_lhood


    ####################################################################################
    ########################### RECONSTRUCTION TERMS ###################################

    epsilon = tf.random.normal(shape=(batch, tmax, 2))
    latent_samples = full_p_mu + epsilon * tf.sqrt(full_p_var)

    pred_vid_batch_logits = build_MLP_decoder_graph(latent_samples, px, py)

    pred_vid = tf.nn.sigmoid(pred_vid_batch_logits)
    recon_err = tf.nn.sigmoid_cross_entropy_with_logits(labels=vid_batch,
                                                        logits=pred_vid_batch_logits)
    sin_elbo_recon = tf.reduce_sum(-recon_err, (2, 3)) # (batch, tmax)

    if type_elbo == 'NP':
        np_elbo_recon = [tf.gather(sin_elbo_recon[i,:], tar_ind[i,:]) for i in range(batch)] # (batch, con_tf)
        np_elbo_recon = [tf.reduce_sum(np_i) for np_i in np_elbo_recon]

    # finally the reconstruction error for each objective!
    if type_elbo == 'GPVAE_Pearce' or type_elbo == 'VAE':
        elbo_recon = tf.reduce_sum(sin_elbo_recon, 1)  # (batch)
    elif type_elbo == 'NP':
        elbo_recon = tf.stack(np_elbo_recon)  # (batch)

    #####################################################################################
    ####################### PUT IT ALL TOGETHER  ########################################

    elbo = elbo_recon + beta * elbo_prior_kl

    return elbo, elbo_recon, elbo_prior_kl, \
           full_p_mu, full_p_var, \
           qnet_mu, qnet_var, pred_vid, l_GP_x, l_GP_y, globals()

