import argparse
import time
import pickle
import os
import json

import numpy as np
# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from utils import plot_mnist, generate_init_inducing_points, import_rotated_mnist, \
                  print_trainable_vars, parse_opt_regime, compute_bias_variance_mean_estimators, \
                  make_checkpoint_folder, pandas_res_saver, latent_samples_SVGPVAE, latent_samples_VAE_full_train
from VAE_utils import mnistVAE, mnistCVAE, SVIGP_Hensman_decoder
from SVGPVAE_model import forward_pass_SVGPVAE, mnistSVGP, forward_pass_standard_VAE_rotated_mnist, \
                          batching_encode_SVGPVAE, batching_encode_SVGPVAE_full, \
                          bacthing_predict_SVGPVAE_rotated_mnist, predict_CVAE

from GPVAE_Casale_model import encode, casaleGP, forward_pass_Casale, predict_test_set_Casale, sort_train_data
from SVIGP_Hensman_model import SVIGP_Hensman, forward_pass_deep_SVIGP_Hensman, predict_deep_SVIGP_Hensman

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def run_experiment_rotated_mnist_SVGPVAE(args, args_dict):
    """
    Function with tensorflow graph and session for SVGPVAE experiments on rotated MNIST data.
    For description of SVGPVAE see chapter 7 in SVGPVAE.tex

    :param args:
    :return:
    """

    # define some constants
    n = len(args.dataset)
    N_train = n * 4050
    N_eval = n * 640
    N_test = n * 270

    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta)
        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        res_file_GP = chkpnt_dir + "res/ELBO_GP_pandas"
        if "SVGPVAE" in args.elbo:
            res_file_VAE = chkpnt_dir + "res/ELBO_VAE_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        json.dump(args_dict, open(chkpnt_dir + "/args.json", "wt"))

    # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data ======================
        # shuffled data or not
        ending = args.dataset + ".p"

        iterator, training_init_op, eval_init_op, test_init_op, train_data_dict, eval_data_dict, test_data_dict, \
            eval_batch_size_placeholder, test_batch_size_placeholder = import_rotated_mnist(args.mnist_data_path,
                                                                                            ending, args.batch_size)

        # get the batch
        input_batch = iterator.get_next()

        # ====================== 2) build ELBO graph ======================

        # init VAE object
        if args.elbo == "CVAE":
            VAE = mnistCVAE(L=args.L)
        else:
            VAE = mnistVAE(L=args.L)
        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        # placeholders
        train_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 2 + args.M))
        train_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))
        test_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 2 + args.M))
        test_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))

        if "SVGPVAE" in args.elbo:  # SVGPVAE
            inducing_points_init = generate_init_inducing_points(args.mnist_data_path + 'train_data' + ending,
                                                                 n=args.nr_inducing_points,
                                                                 remove_test_angle=None,
                                                                 PCA=args.PCA, M=args.M)
            titsias = 'Titsias' in args.elbo
            ip_joint = not args.ip_joint
            GP_joint = not args.GP_joint
            if args.ov_joint:
                if args.PCA:  # use PCA embeddings for initialization of object vectors
                    object_vectors_init = pickle.load(open(args.mnist_data_path +
                                                           'pca_ov_init{}.p'.format(args.dataset), 'rb'))
                else:  # initialize object vectors randomly
                    object_vectors_init = np.random.normal(0, 1.5,
                                                           len(args.dataset)*400*args.M).reshape(len(args.dataset)*400,
                                                                                                 args.M)
            else:
                object_vectors_init = None

            # init SVGP object
            SVGP_ = mnistSVGP(titsias=titsias, fixed_inducing_points=ip_joint,
                              initial_inducing_points=inducing_points_init,
                              fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init, name='main',
                              jitter=args.jitter, N_train=N_train,
                              L=args.L, K_obj_normalize=args.object_kernel_normalize)

            # forward pass SVGPVAE
            C_ma_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            lagrange_mult_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
            alpha_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

            elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, recon_images, \
            inside_elbo_recon, inside_elbo_kl, latent_samples, \
            C_ma, lagrange_mult, mean_vectors = forward_pass_SVGPVAE(input_batch,
                                                                     beta=beta,
                                                                     vae=VAE,
                                                                     svgp=SVGP_,
                                                                     C_ma=C_ma_placeholder,
                                                                     lagrange_mult=lagrange_mult_placeholder,
                                                                     alpha=alpha_placeholder,
                                                                     kappa=np.sqrt(args.kappa_squared),
                                                                     clipping_qs=args.clip_qs,
                                                                     GECO=args.GECO,
                                                                     bias_analysis=args.bias_analysis)

            # forward pass standard VAE (for training regime from CASALE: VAE-GP-joint)
            recon_loss_VAE, KL_term_VAE, elbo_VAE, \
            recon_images_VAE, qnet_mu_VAE, qnet_var_VAE, \
            latent_samples_VAE = forward_pass_standard_VAE_rotated_mnist(input_batch,
                                                                         vae=VAE)

        elif args.elbo == "VAE" or args.elbo == "CVAE":  # plain VAE or CVAE
            CVAE = args.elbo == "CVAE"

            recon_loss, KL_term, elbo, \
            recon_images, qnet_mu, qnet_var, latent_samples = forward_pass_standard_VAE_rotated_mnist(input_batch,
                                                                                                      vae=VAE,
                                                                                                      CVAE=CVAE)

        else:
            raise ValueError

        # test loss and predictions

        if "SVGPVAE" in args.elbo:
            train_encodings_means_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, args.L))
            train_encodings_vars_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, args.L))

            qnet_mu_train, qnet_var_train, _ = batching_encode_SVGPVAE(input_batch, vae=VAE,
                                                                    clipping_qs=args.clip_qs)
            recon_images_test, \
            recon_loss_test = bacthing_predict_SVGPVAE_rotated_mnist(input_batch,
                                                                     vae=VAE,
                                                                     svgp=SVGP_,
                                                                     qnet_mu=train_encodings_means_placeholder,
                                                                     qnet_var=train_encodings_vars_placeholder,
                                                                     aux_data_train=train_aux_data_placeholder)

            # GP diagnostics
            GP_l, GP_amp, GP_ov, GP_ip = SVGP_.variable_summary()

        # bias analysis
        if args.bias_analysis:
            means, vars = batching_encode_SVGPVAE_full(train_images_placeholder,
                                                       vae=VAE, clipping_qs=args.clip_qs)
            mean_vector_full_data = []
            for l in range(args.L):
                mean_vector_full_data.append(SVGP_.mean_vector_bias_analysis(index_points=train_aux_data_placeholder,
                                                                             y=means[:, l], noise=vars[:, l]))

        if args.save_latents:
            if "SVGPVAE" in args.elbo:
                latent_samples_full = latent_samples_SVGPVAE(train_images_placeholder, train_aux_data_placeholder,
                                                             vae=VAE, svgp=SVGP_, clipping_qs=args.clip_qs)
            else:
                latent_samples_full = latent_samples_VAE_full_train(train_images_placeholder,
                                                                    vae=VAE, clipping_qs=args.clip_qs)
        # conditional generation for CVAE
        if args.elbo == "CVAE":
            recon_images_test, recon_loss_test = predict_CVAE(images_train=train_images_placeholder,
                                                              images_test=test_images_placeholder,
                                                              aux_data_train=train_aux_data_placeholder,
                                                              aux_data_test=test_aux_data_placeholder,
                                                              vae=VAE, test_indices=test_data_dict['aux_data'][:, 0])

        # ====================== 3) optimizer ops ======================
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        if args.GECO:  # minimizing GECO objective
            gradients = tf.gradients(elbo, train_vars)
        else:  # minimizing negative elbo
            gradients = tf.gradients(-elbo, train_vars)

        optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                               global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if args.save:
            res_vars = [global_step,
                        elbo,
                        recon_loss,
                        KL_term,
                        tf.math.reduce_min(qnet_mu),
                        tf.math.reduce_max(qnet_mu),
                        tf.math.reduce_min(qnet_var),
                        tf.math.reduce_max(qnet_var),
                        qnet_var]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "KL term",
                         "min qnet_mu",
                         "max qnet_mu",
                         "min qnet_var",
                         "max qnet_var",
                         "full qnet_var"]

            if 'SVGPVAE' in args.elbo:
                res_vars += [inside_elbo,
                             inside_elbo_recon,
                             inside_elbo_kl,
                             ce_term,
                             tf.math.reduce_min(p_m),
                             tf.math.reduce_max(p_m),
                             tf.math.reduce_min(p_v),
                             tf.math.reduce_max(p_v),
                             latent_samples,
                             C_ma,
                             lagrange_mult]

                res_names += ["inside elbo",
                              "inside elbo recon",
                              "inside elbo KL",
                              "ce_term",
                              "min p_m",
                              "max p_m",
                              "min p_v",
                              "max p_v",
                              "latent_samples",
                              "C_ma",
                              "lagrange_mult"]

                res_vars_VAE = [global_step,
                                elbo_VAE,
                                recon_loss_VAE,
                                KL_term_VAE,
                                tf.math.reduce_min(qnet_mu_VAE),
                                tf.math.reduce_max(qnet_mu_VAE),
                                tf.math.reduce_min(qnet_var_VAE),
                                tf.math.reduce_max(qnet_var_VAE),
                                latent_samples_VAE]

                res_names_VAE = ["step",
                                 "ELBO",
                                 "recon loss",
                                 "KL term",
                                 "min qnet_mu",
                                 "max qnet_mu",
                                 "min qnet_var",
                                 "max qnet_var",
                                 "latent_samples"]

                res_vars_GP = [GP_l,
                               GP_amp,
                               GP_ov,
                               GP_ip]

                res_names_GP = ['length scale', 'amplitude', 'object vectors', 'inducing points']

                res_saver_VAE = pandas_res_saver(res_file_VAE, res_names_VAE)
                res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)

            res_saver = pandas_res_saver(res_file, res_names)

        # ====================== 5) print and init trainable params ======================
        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if args.save_model_weights:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)

        # ====================== 7) tf.session ======================

        if "SVGPVAE" in args.elbo:
            nr_epochs, training_regime = parse_opt_regime(args.opt_regime)
        else:
            nr_epochs = args.nr_epochs

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            # training loop
            first_step = True  # switch for initialization of GECO algorithm
            C_ma_ = 0.0
            lagrange_mult_ = 1.0

            start_time = time.time()
            cgen_test_set_MSE = []
            for epoch in range(nr_epochs):

                # 7.1) train for one epoch
                sess.run(training_init_op)
                elbos, losses = [], []
                start_time_epoch = time.time()
                if args.bias_analysis:
                    mean_vectors_arr = []
                while True:
                    try:
                        if args.GECO and "SVGPVAE" in args.elbo and training_regime[epoch] != 'VAE':
                            if first_step:
                                alpha = 0.0
                            else:
                                alpha = args.alpha
                            _, g_s_, elbo_, C_ma_, lagrange_mult_, recon_loss_, mean_vectors_ = sess.run([optim_step, global_step,
                                                                              elbo, C_ma, lagrange_mult,
                                                                              recon_loss, mean_vectors],
                                                                              {beta: args.beta, lr: args.lr,
                                                                               alpha_placeholder: alpha,
                                                                               C_ma_placeholder: C_ma_,
                                                                               lagrange_mult_placeholder: lagrange_mult_})
                            if args.bias_analysis:
                                mean_vectors_arr.append(mean_vectors_)
                        elif args.elbo == "VAE" or args.elbo == "CVAE":
                            _, g_s_, elbo_, recon_loss_ = sess.run(
                                [optim_step, global_step, elbo, recon_loss],
                                {beta: args.beta, lr: args.lr})
                        else:
                            _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo, recon_loss],
                                                      {beta: args.beta, lr: args.lr,
                                                       alpha_placeholder: args.alpha,
                                                       C_ma_placeholder: C_ma_,
                                                       lagrange_mult_placeholder: lagrange_mult_})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                        first_step = False  # switch for initizalition of GECO algorithm
                    except tf.errors.OutOfRangeError:
                        if args.bias_analysis:
                            mean_vector_full_data_ = sess.run(mean_vector_full_data,
                                                              {train_images_placeholder: train_data_dict['images'],
                                                               train_aux_data_placeholder: train_data_dict['aux_data']})

                            bias = compute_bias_variance_mean_estimators(mean_vectors_arr, mean_vector_full_data_)
                            print("Bias for epoch {}: {}".format(epoch, bias))
                        if (epoch + 1) % 10 == 0:
                            regime = training_regime[epoch] if "SVGPVAE" in args.elbo else "VAE"
                            print('Epoch {}, opt regime {}, mean ELBO per batch: {}'.format(epoch, regime,
                                                                                            np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('MSE loss on train set for epoch {} : {}'.format(epoch, MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}, opt regime {}: {}".format(epoch,
                                                                                        regime,
                                                                                        end_time_epoch - start_time_epoch))
                        break

                # 7.2) calculate loss on eval set
                if args.save and (epoch + 1) % 10 == 0 and "SVGPVAE" in args.elbo:
                    losses = []
                    sess.run(eval_init_op, {eval_batch_size_placeholder: args.batch_size})
                    while True:
                        try:
                            recon_loss_ = sess.run(recon_loss, {beta: args.beta, lr: args.lr,
                                                                     alpha_placeholder: args.alpha,
                                                                     C_ma_placeholder: C_ma_,
                                                                     lagrange_mult_placeholder: lagrange_mult_})
                            losses.append(recon_loss_)
                        except tf.errors.OutOfRangeError:
                            MSE = np.sum(losses) / N_eval
                            print('MSE loss on eval set for epoch {} : {}'.format(epoch, MSE))
                            break

                # 7.3) save metrics to Pandas df for model diagnostics
                if args.save and (epoch + 1) % 10 == 0:
                    if args.test_set_metrics:
                        # sess.run(test_init_op, {test_batch_size_placeholder: N_test})  # see [update, 7.7.] above
                        sess.run(test_init_op, {test_batch_size_placeholder: args.batch_size})
                    else:
                        # sess.run(eval_init_op, {eval_batch_size_placeholder: N_eval})  # see [update, 7.7.] above
                        sess.run(eval_init_op, {eval_batch_size_placeholder: args.batch_size})

                    if "SVGPVAE" in args.elbo:
                        # save elbo metrics depending on the type of forward pass (plain VAE vs SVGPVAE)
                        if training_regime[epoch] == 'VAE':
                            new_res = sess.run(res_vars_VAE, {beta: args.beta})
                            res_saver_VAE(new_res, 1)
                        else:
                            new_res = sess.run(res_vars, {beta: args.beta,
                                                          alpha_placeholder: args.alpha,
                                                          C_ma_placeholder: C_ma_,
                                                          lagrange_mult_placeholder: lagrange_mult_})
                            res_saver(new_res, 1)

                        # save GP params
                        new_res_GP = sess.run(res_vars_GP, {beta: args.beta,
                                                            alpha_placeholder: args.alpha,
                                                            C_ma_placeholder: C_ma_,
                                                            lagrange_mult_placeholder: lagrange_mult_})
                        res_saver_GP(new_res_GP, 1)
                    else:
                        new_res = sess.run(res_vars, {beta: args.beta})
                        res_saver(new_res, 1)

                # 7.4) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % 10 == 0:

                    losses, recon_images_arr = [], []
                    sess.run(test_init_op, {test_batch_size_placeholder: args.batch_size})
                    # test set: reconstruction
                    while True:
                        try:
                            if "SVGPVAE" in args.elbo:
                                recon_loss_, recon_images_ = sess.run([recon_loss, recon_images],
                                                                      {beta: args.beta,
                                                                       alpha_placeholder: args.alpha,
                                                                       C_ma_placeholder: C_ma_,
                                                                       lagrange_mult_placeholder: lagrange_mult_})
                            else:
                                recon_loss_, recon_images_ = sess.run([recon_loss, recon_images],
                                                                      {beta: args.beta})
                            losses.append(recon_loss_)
                            recon_images_arr.append(recon_images_)
                        except tf.errors.OutOfRangeError:
                            MSE = np.sum(losses) / N_test
                            print('MSE loss on test set for epoch {} : {}'.format(epoch, MSE))
                            recon_images_arr = np.concatenate(tuple(recon_images_arr))
                            plot_mnist(test_data_dict['images'],
                                       recon_images_arr,
                                       title="Epoch: {}. Recon MSE test set:{}".format(epoch + 1, round(MSE, 4)))
                            if args.show_pics:
                                plt.show()
                                plt.pause(0.01)
                            if args.save:
                                plt.savefig(pic_folder + str(g_s_) + ".png")
                            break
                    # test set: conditional generation SVGPVAE
                    if "SVGPVAE" in args.elbo:

                        # encode training data (in batches)
                        sess.run(training_init_op)
                        means, vars = [], []
                        while True:
                            try:
                                qnet_mu_train_, qnet_var_train_ = sess.run([qnet_mu_train, qnet_var_train])
                                means.append(qnet_mu_train_)
                                vars.append(qnet_var_train_)
                            except tf.errors.OutOfRangeError:
                                break
                        means = np.concatenate(means, axis=0)
                        vars = np.concatenate(vars, axis=0)

                        # predict test data (in batches)
                        sess.run(test_init_op, {test_batch_size_placeholder: args.batch_size})
                        recon_loss_cgen, recon_images_cgen = [], []
                        while True:
                            try:
                                loss_, pics_ = sess.run([recon_loss_test, recon_images_test],
                                                        {train_aux_data_placeholder: train_data_dict['aux_data'],
                                                         train_encodings_means_placeholder: means,
                                                         train_encodings_vars_placeholder: vars})
                                recon_loss_cgen.append(loss_)
                                recon_images_cgen.append(pics_)
                            except tf.errors.OutOfRangeError:
                                break
                        recon_loss_cgen = np.sum(recon_loss_cgen) / N_test
                        recon_images_cgen = np.concatenate(recon_images_cgen, axis=0)

                    # test set: conditional generation CVAE
                    if args.elbo == "CVAE":
                        recon_loss_cgen, recon_images_cgen = sess.run([recon_loss_test, recon_images_test],
                                            {train_aux_data_placeholder: train_data_dict['aux_data'],
                                             train_images_placeholder: train_data_dict['images'],
                                             test_aux_data_placeholder: test_data_dict['aux_data'],
                                             test_images_placeholder: test_data_dict['images']})

                    # test set: plot generations
                    if args.elbo != "VAE":
                        cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                        print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                                    recon_loss_cgen))
                        plot_mnist(test_data_dict['images'],
                                   recon_images_cgen,
                                   title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                        if args.show_pics:
                            plt.show()
                            plt.pause(0.01)
                        if args.save:
                            plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                            with open(pic_folder + "test_metrics.txt", "a") as f:
                                f.write("{},{},{}\n".format(epoch + 1, round(MSE, 4), round(recon_loss_cgen, 4)))

                    # save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(nr_epochs, round(end_time - start_time, 2)))

            if "SVGPVAE" in args.elbo:
                # report best test set cgen MSE achieved throughout training
                best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
                print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                             best_cgen_MSE[1]))

            # save images from conditional generation
            if args.save and args.elbo != "VAE":
                with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                    pickle.dump(recon_images_cgen, test_pickle)

            # save latents
            if args.save_latents:
                if "SVGPVAE" in args.elbo:
                    latent_samples_full_ = sess.run(latent_samples_full,
                                                    {train_images_placeholder: train_data_dict['images'],
                                                     train_aux_data_placeholder: train_data_dict['aux_data']})
                else:
                    latent_samples_full_ = sess.run(latent_samples_full,
                                                    {train_images_placeholder: train_data_dict['images']})
                with open(chkpnt_dir + '/latents_train_full.p', 'wb') as pickle_latents:
                    pickle.dump(latent_samples_full_, pickle_latents)


def run_experiment_rotated_mnist_SVIGP_Hensman(args, args_dict):
    """
    Function with tensorflow graph and session for SVIGP_Hensman experiments on rotated MNIST data.

    :param args:
    :return:
    """

    # define some constants
    n = len(args.dataset)
    N_train = n * 4050
    N_eval = n * 640
    N_test = n * 270

    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta)
        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        res_file_GP = chkpnt_dir + "res/ELBO_GP_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        json.dump(args_dict, open(chkpnt_dir + "/args.json", "wt"))

    # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data ======================
        # shuffled data or not
        ending = args.dataset + ".p"

        iterator, training_init_op, eval_init_op, test_init_op, train_data_dict, eval_data_dict, test_data_dict, \
            eval_batch_size_placeholder, test_batch_size_placeholder = import_rotated_mnist(args.mnist_data_path,
                                                                                            ending, args.batch_size,
                                                                                            global_index=True)

        # get the batch
        input_batch = iterator.get_next()

        # ====================== 2) build ELBO graph ======================

        # init VAE object
        VAE = SVIGP_Hensman_decoder(L=args.L)

        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        # init inducing points
        inducing_points_init = generate_init_inducing_points(args.mnist_data_path + 'train_data' + ending,
                                                             n=args.nr_inducing_points,
                                                             remove_test_angle=None,
                                                             PCA=args.PCA, M=args.M)
        ip_joint = not args.ip_joint
        GP_joint = not args.GP_joint

        # init GP-LVM vectors
        if args.ov_joint:
            if args.PCA:  # use PCA embeddings for initialization of object vectors
                object_vectors_init = pickle.load(open(args.mnist_data_path +
                                                       'pca_ov_init{}.p'.format(args.dataset), 'rb'))
            else:  # initialize object vectors randomly
                object_vectors_init = np.random.normal(0, 1.5,
                                                       len(args.dataset)*400*args.M).reshape(len(args.dataset)*400,
                                                                                             args.M)
        else:
            object_vectors_init = None

        # init SVGP object
        SVGP_ = SVIGP_Hensman(fixed_inducing_points=ip_joint, initial_inducing_points=inducing_points_init,
                              fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init, name='main',
                              jitter=args.jitter, N_train=N_train, L=args.L,
                              K_obj_normalize=args.object_kernel_normalize, dtype=np.float64)

        # forward pass
        elbo, recon_loss, KL_term, inside_elbo, recon_images, \
        inside_elbo_recon, inside_elbo_kl, latent_samples = forward_pass_deep_SVIGP_Hensman(input_batch, vae=VAE, svgp=SVGP_)

        # test loss and predictions
        recon_images_test, recon_loss_test = predict_deep_SVIGP_Hensman(input_batch, vae=VAE, svgp=SVGP_)

        # GP diagnostics
        GP_l, GP_amp, GP_ov, GP_ip = SVGP_.variable_summary()

        # ====================== 3) optimizer ops ======================
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # minimizing negative elbo
        gradients = tf.gradients(-elbo, train_vars)
        optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars), global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if args.save:
            res_vars = [global_step,
                        elbo,
                        recon_loss,
                        KL_term]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "KL term"]

            res_vars += [inside_elbo,
                         inside_elbo_recon,
                         inside_elbo_kl,
                         latent_samples]

            res_names += ["inside elbo",
                          "inside elbo recon",
                          "inside elbo KL",
                          "latent_samples"]

            res_vars_GP = [GP_l,
                           GP_amp,
                           GP_ov,
                           GP_ip]

            res_names_GP = ['length scale', 'amplitude', 'object vectors', 'inducing points']

            res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)

            res_saver = pandas_res_saver(res_file, res_names)

        # ====================== 5) print and init trainable params ======================
        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if args.save_model_weights:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)

        # ====================== 7) tf.session ======================

        nr_epochs = args.nr_epochs

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            # training loop

            start_time = time.time()
            cgen_test_set_MSE = []
            for epoch in range(nr_epochs):

                # 7.1) train for one epoch
                sess.run(training_init_op)

                elbos, losses = [], []
                start_time_epoch = time.time()
                while True:
                    try:
                        _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo, recon_loss],
                                                               {beta: args.beta, lr: args.lr})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                    except tf.errors.OutOfRangeError:
                        if (epoch + 1) % 10 == 0:
                            print('Epoch {}, mean ELBO per batch: {}'.format(epoch, np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('MSE loss on train set for epoch {} : {}'.format(epoch, MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}: {}".format(epoch, end_time_epoch - start_time_epoch))
                        break

                # 7.2) save metrics to Pandas df for model diagnostics
                if args.save and (epoch + 1) % 10 == 0:
                    if args.test_set_metrics:
                        sess.run(test_init_op, {test_batch_size_placeholder: args.batch_size})
                    else:
                        sess.run(eval_init_op, {eval_batch_size_placeholder: args.batch_size})

                    new_res = sess.run(res_vars, {beta: args.beta})
                    res_saver(new_res, 1)

                    # save GP params
                    new_res_GP = sess.run(res_vars_GP, {beta: args.beta})
                    res_saver_GP(new_res_GP, 1)

                # 7.3) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % 10 == 0:

                    # test set: conditional generation
                    # predict test data (in batches)
                    sess.run(test_init_op, {test_batch_size_placeholder: args.batch_size})
                    recon_loss_cgen, recon_images_cgen = [], []
                    while True:
                        try:
                            loss_, pics_ = sess.run([recon_loss_test, recon_images_test])
                            recon_loss_cgen.append(loss_)
                            recon_images_cgen.append(pics_)
                        except tf.errors.OutOfRangeError:
                            break
                    recon_loss_cgen = np.sum(recon_loss_cgen) / N_test
                    recon_images_cgen = np.concatenate(recon_images_cgen, axis=0)

                    # test set: plot generations
                    cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                    print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                                recon_loss_cgen))
                    plot_mnist(test_data_dict['images'],
                               recon_images_cgen,
                               title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                    if args.show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if args.save:
                        plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                        with open(pic_folder + "test_metrics.txt", "a") as f:
                            f.write("{},{},{}\n".format(epoch + 1, round(MSE, 4), round(recon_loss_cgen, 4)))

                    # save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(nr_epochs, round(end_time - start_time, 2)))

            # report best test set cgen MSE achieved throughout training
            best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                         best_cgen_MSE[1]))

            # save images from conditional generation
            if args.save:
                with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                    pickle.dump(recon_images_cgen, test_pickle)


def run_experiment_rotated_mnist_Casale(args):
    """
    Reimplementation of Casale's GPVAE model.

    :param args:
    :return:
    """

    # define some constants
    n = len(args.dataset)
    N_train = n * 4050
    N_test = n * 270

    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta)
        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        res_file_GP = chkpnt_dir + "res/ELBO_GP_pandas"
        res_file_VAE = chkpnt_dir + "res/ELBO_VAE_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        json.dump(vars(args), open(chkpnt_dir + "/args.json", "wt"))

    # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data and data placeholders ======================
        GPLVM_ending = "" if args.M == 8 else "_{}".format(args.M)
        train_data_dict = pickle.load(open(args.mnist_data_path + 'train_data' + args.dataset +
                                           "{}.p".format(GPLVM_ending), 'rb'))
        train_data_dict = sort_train_data(train_data_dict, dataset=args.dataset)
        train_ids_mask = pickle.load(open(args.mnist_data_path + "train_ids_mask" + args.dataset +
                                          "{}.p".format(GPLVM_ending), 'rb'))

        train_data_images = tf.data.Dataset.from_tensor_slices(train_data_dict['images'])
        train_data_aux_data = tf.data.Dataset.from_tensor_slices(train_data_dict['aux_data'])
        train_data = tf.data.Dataset.zip((train_data_images, train_data_aux_data)).batch(args.batch_size)
        iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        training_init_op = iterator.make_initializer(train_data)
        input_batch = iterator.get_next()

        train_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 3 + args.M))
        train_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))
        test_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 2 + args.M))
        test_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))

        test_data_dict = pickle.load(open(args.mnist_data_path + 'test_data' + args.dataset +
                                          "{}.p".format(GPLVM_ending), 'rb'))

        # ====================== 2) build ELBO graph ======================

        # 2.0) define placeholders and all the objects
        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        VAE = mnistVAE(L=args.L)

        GP_joint = not args.GP_joint
        if args.PCA:  # use PCA embeddings for initialization of object vectors
            object_vectors_init = pickle.load(
                open(args.mnist_data_path + 'pca_ov_init{}{}.p'.format(args.dataset, GPLVM_ending), 'rb'))
        else:  # initialize object vectors randomly
            assert args.ov_joint, "If --ov_joint is not used, at least PCA initialization must be utilized."
            object_vectors_init = np.random.normal(0, 1.5, len(args.dataset) * 400 * args.M).reshape(
                len(args.dataset) * 400, args.M)

        GP = casaleGP(fixed_gp_params=GP_joint, object_vectors_init=object_vectors_init,
                      object_kernel_normalize=args.object_kernel_normalize, ov_joint=args.ov_joint)

        # 2.1) encode full train dataset
        Z = encode(train_images_placeholder, vae=VAE, clipping_qs=args.clip_qs)  # (N x L)

        # 2.2) compute V matrix and GP taylor coefficients
        V = GP.V_matrix(train_aux_data_placeholder, train_ids_mask=train_ids_mask)  # (N x H)
        a, B, c = GP.taylor_coeff(Z=Z, V=V)

        # 2.3) forward passes

        # GPPVAE forward pass
        elbo, recon_loss, GP_prior_term, log_var, \
        qnet_mu, qnet_var, recon_images = forward_pass_Casale(input_batch, vae=VAE, a=a, B=B, c=c, V=V, beta=beta,
                                                              GP=GP, clipping_qs=args.clip_qs)

        # plain VAE forward pass
        recon_loss_VAE, KL_term_VAE, elbo_VAE, \
        recon_images_VAE, qnet_mu_VAE, qnet_var_VAE, _ = forward_pass_standard_VAE_rotated_mnist(input_batch, vae=VAE)

        # 2.5) predict on test set (conditional generation)
        recon_images_test, recon_loss_test = predict_test_set_Casale(test_images=test_images_placeholder,
                                                                     test_aux_data=test_aux_data_placeholder,
                                                                     train_aux_data=train_aux_data_placeholder,
                                                                     V=V, vae=VAE, GP=GP, latent_samples_train=Z)

        # ====================== 3) optimizer ops ======================

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        # 3.1) joint optimization
        gradients_joint = tf.gradients(elbo, train_vars)
        optim_step_joint = optimizer.apply_gradients(grads_and_vars=zip(gradients_joint, train_vars),
                                                     global_step=global_step)

        # 3.2) GP optimization
        GP_vars = [x for x in train_vars if 'GP' in x.name]
        gradients_GP = tf.gradients(elbo, GP_vars)
        optim_step_GP = optimizer.apply_gradients(grads_and_vars=zip(gradients_GP, GP_vars),
                                                  global_step=global_step)

        # 3.3) VAE optimization
        VAE_vars = [x for x in train_vars if not 'GP' in x.name]
        gradients_VAE = tf.gradients(-elbo_VAE, VAE_vars)  # here we optimize standard ELBO objective
        optim_step_VAE = optimizer.apply_gradients(grads_and_vars=zip(gradients_VAE, VAE_vars),
                                                   global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if args.save:
            # GP diagnostics
            GP_l, GP_amp, GP_ov, GP_alpha = GP.variable_summary()
            if GP_ov is None:
                GP_ov = tf.constant(0.0)

            res_vars = [global_step,
                        elbo,
                        recon_loss,
                        GP_prior_term,
                        log_var,
                        tf.math.reduce_min(qnet_mu),
                        tf.math.reduce_max(qnet_mu),
                        tf.math.reduce_min(qnet_var),
                        tf.math.reduce_max(qnet_var)]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "GP prior term",
                         "log var term",
                         "min qnet_mu",
                         "max qnet_mu",
                         "min qnet_var",
                         "max qnet_var"]

            res_vars_GP = [GP_l,
                           GP_amp,
                           GP_ov,
                           GP_alpha]

            res_names_GP = ['length scale', 'amplitude', 'object vectors', 'alpha']

            res_vars_VAE = [global_step,
                            elbo_VAE,
                            recon_loss_VAE,
                            KL_term_VAE,
                            tf.math.reduce_min(qnet_mu_VAE),
                            tf.math.reduce_max(qnet_mu_VAE),
                            tf.math.reduce_min(qnet_var_VAE),
                            tf.math.reduce_max(qnet_var_VAE)]

            res_names_VAE = ["step",
                             "ELBO",
                             "recon loss",
                             "KL term",
                             "min qnet_mu",
                             "max qnet_mu",
                             "min qnet_var",
                             "max qnet_var"]

            res_saver = pandas_res_saver(res_file, res_names)
            res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)
            res_saver_VAE = pandas_res_saver(res_file_VAE, res_names_VAE)

        # ====================== 5) print and init trainable params ======================

        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if args.save_model_weights:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)

        # ====================== 7) tf.session ======================
        nr_epochs, training_regime = parse_opt_regime(args.opt_regime)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            start_time = time.time()
            cgen_test_set_MSE = []
            # training loop
            for epoch in range(nr_epochs):

                # 7.1) set objective functions etc (support for different training regimes, handcrafted schedules etc)

                if training_regime[epoch] == "VAE":
                    optim_step = optim_step_VAE
                    elbo_main = elbo_VAE
                    recon_loss_main = recon_loss_VAE
                    recon_images_main = recon_images_VAE
                    lr_main = 0.001
                    beta_main = 1.0
                elif training_regime[epoch] == "GP":
                    optim_step = optim_step_GP
                    elbo_main = elbo
                    beta_main = args.beta
                    lr_main = 0.01
                    recon_loss_main = recon_loss
                    recon_images_main = recon_images
                elif training_regime[epoch] == "joint":
                    optim_step = optim_step_joint
                    elbo_main = elbo
                    beta_main = args.beta
                    lr_main = 0.001
                    recon_loss_main = recon_loss
                    recon_images_main = recon_images

                # 7.2) train for one epoch
                sess.run(training_init_op)

                elbos, losses = [], []
                start_time_epoch = time.time()
                while True:
                    try:
                        _, g_s_, elbo_, recon_loss_ = sess.run([optim_step, global_step, elbo_main, recon_loss_main],
                                                               {beta: beta_main, lr: lr_main,
                                                                train_aux_data_placeholder: train_data_dict['aux_data'],
                                                                train_images_placeholder: train_data_dict['images']})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                    except tf.errors.OutOfRangeError:
                        if (epoch + 1) % 1 == 0:
                            print('Epoch {}, opt regime {}, mean ELBO per batch: {}'.format(epoch,
                                                                                            training_regime[epoch],
                                                                                            np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('Epoch {}, opt regime {}, MSE loss on train set: {}'.format(epoch,
                                                                                              training_regime[epoch],
                                                                                              MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}, opt regime {}: {}".format(epoch,
                                                                                        training_regime[epoch],
                                                                                        end_time_epoch - start_time_epoch))


                        break

                # 7.3) calculate loss on eval set
                # TODO

                # 7.4) save metrics to Pandas df for model diagnostics

                sess.run(training_init_op)  # currently metrics are calculated only for first batch of the training data

                if args.save and (epoch + 1) % 5 == 0:
                    if training_regime[epoch] == "VAE":
                        new_res_VAE = sess.run(res_vars_VAE, {beta: beta_main,
                                                              train_aux_data_placeholder: train_data_dict['aux_data'],
                                                              train_images_placeholder: train_data_dict['images']})
                        res_saver_VAE(new_res_VAE, 1)
                    else:
                        new_res = sess.run(res_vars, {beta: beta_main,
                                                      train_aux_data_placeholder: train_data_dict['aux_data'],
                                                      train_images_placeholder: train_data_dict['images']})
                        res_saver(new_res, 1)

                    new_res_GP = sess.run(res_vars_GP)
                    res_saver_GP(new_res_GP, 1)

                # 7.5) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % 5 == 0:
                    # test set: reconstruction
                    # TODO

                    # test set: conditional generation
                    recon_images_cgen, recon_loss_cgen  = sess.run([recon_images_test, recon_loss_test ],
                                                                  feed_dict={train_images_placeholder:
                                                                                 train_data_dict['images'],
                                                                             test_images_placeholder:
                                                                                 test_data_dict['images'],
                                                                             train_aux_data_placeholder:
                                                                                 train_data_dict['aux_data'],
                                                                             test_aux_data_placeholder:
                                                                                 test_data_dict['aux_data']})

                    cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                    print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                                recon_loss_cgen))
                    plot_mnist(test_data_dict['images'],
                               recon_images_cgen,
                               title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                    if args.show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if args.save:
                        plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                        with open(pic_folder + "test_metrics.txt", "a") as f:
                            f.write("{},{}\n".format(epoch + 1, round(recon_loss_cgen, 4)))

                    # save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(nr_epochs, round(end_time - start_time, 2)))

            # report best test set cgen MSE achieved throughout training
            best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                         best_cgen_MSE[1]))

            # save images from conditional generation
            if args.save:
                with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                    pickle.dump(recon_images_cgen, test_pickle)


if __name__=="__main__":

    default_base_dir = os.getcwd()

    parser_mnist = argparse.ArgumentParser(description='Rotated MNIST experiment.')
    parser_mnist.add_argument('--expid', type=str, default="debug_MNIST", help='give this experiment a name')
    parser_mnist.add_argument('--base_dir', type=str, default=default_base_dir,
                              help='folder within a new dir is made for each run')
    parser_mnist.add_argument('--elbo', type=str, choices=['VAE', 'CVAE', 'SVGPVAE_Hensman', 'SVGPVAE_Titsias',
                                                           'GPVAE_Casale', 'GPVAE_Casale_batch', 'SVIGP_Hensman'],
                              default='VAE')
    parser_mnist.add_argument('--mnist_data_path', type=str, default='MNIST_data/',
                              help='Path where rotated MNIST data is stored.')
    parser_mnist.add_argument('--batch_size', type=int, default=256)
    parser_mnist.add_argument('--nr_epochs', type=int, default=1000)
    parser_mnist.add_argument('--beta', type=float, default=0.001)
    parser_mnist.add_argument('--nr_inducing_points', type=float, default=2, help="Number of object vectors per angle.")
    parser_mnist.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser_mnist.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser_mnist.add_argument('--ip_joint', action="store_true", help='Inducing points joint optimization.')
    parser_mnist.add_argument('--ov_joint', action="store_true", help='Object vectors joint optimization.')
    parser_mnist.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser_mnist.add_argument('--save_model_weights', action="store_true",
                              help='Save model weights. For debug purposes.')
    parser_mnist.add_argument('--dataset', type=str, choices=['3', '36', '13679'], default='3')
    parser_mnist.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser_mnist.add_argument('--opt_regime', type=str, default=['joint-1000'], nargs="+")
    parser_mnist.add_argument('--L', type=int, default=16, help="Nr. of latent channels")
    parser_mnist.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')
    parser_mnist.add_argument('--ram', type=float, default=1.0, help='fraction of GPU ram to use')
    parser_mnist.add_argument('--test_set_metrics', action='store_true',
                              help='Calculate metrics on test data. If false, metrics are calculated on eval data.')
    parser_mnist.add_argument('--GECO', action='store_true', help='Use GECO algorithm for training.')
    parser_mnist.add_argument('--alpha', type=float, default=0.99, help='Moving average parameter for GECO.')
    parser_mnist.add_argument('--kappa_squared', type=float, default=0.020, help='Constraint parameter for GECO.')
    parser_mnist.add_argument('--object_kernel_normalize', action='store_true',
                              help='Normalize object (linear) kernel.')
    parser_mnist.add_argument('--save_latents', action='store_true', help='Save Z . For t-SNE plots :)')
    parser_mnist.add_argument('--jitter', type=float, default=0.000001, help='Jitter for numerical stability.')
    parser_mnist.add_argument('--PCA', action="store_true",
                              help='Use PCA embeddings for initialization of object vectors.')
    parser_mnist.add_argument('--bias_analysis', action='store_true',
                              help="Compute bias of estimator for mean vector in hat{q}^Titsias for every epoch.")
    parser_mnist.add_argument('--M', type=int, default=8, help="Dimension of GPLVM vectors.")




    args_mnist = parser_mnist.parse_args()

    if args_mnist.elbo == "GPVAE_Casale":
        run_experiment_rotated_mnist_Casale(args_mnist)

    elif args_mnist.elbo == "SVIGP_Hensman":
        dict_ = vars(args_mnist)
        run_experiment_rotated_mnist_SVIGP_Hensman(args_mnist, dict_)

    else:  # VAE, CVAE, SVGPVAE_Hensman, SVGPVAE_Titsias
        dict_ = vars(args_mnist)  # [update, 23.6.] to get around weirdest bug ever
        run_experiment_rotated_mnist_SVGPVAE(args_mnist, dict_)












