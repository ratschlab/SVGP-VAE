import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import json

import tensorflow as tf

from VAE_utils import spritesVAE, sprites_representation_network
from SVGPVAE_model import forward_pass_SVGPVAE, forward_pass_standard_VAE_rotated_mnist, \
                    batching_encode_SVGPVAE, precompute_GP_params_SVGPVAE, \
                    spritesSVGP, predict_SVGPVAE_sprites_test_character
from utils import make_checkpoint_folder, pandas_res_saver, \
                  print_trainable_vars, parse_opt_regime,  \
                  import_sprites, sprites_PCA_init, plot_sprites, aux_data_sprites_utils, \
                  IndexedSlicesValue_to_numpy, forward_pass_pretraining_repr_NN


def run_experiment_sprites_SVGPVAE(args, dict_):
    """
    SVGPVAE experiment on SPRITES data.

    Currently there is NO support for VAE-GP-joint training regime.
    Also there is NO support for cgen prediction on test_action data.

    :param args:
    :return:
    """

    # define some constants
    N_train = 50000
    N_test_action = 22000
    N_test_character = 21312
    N_actions = 72
    N_frames_per_character_train = 50  # if want to change this, need to rerun function that generates .tfrecord files
    batch_size_test_char = 576

    assert args.batch_size % N_frames_per_character_train == 0.0, \
        "Batch size needs to be divisible by {}".format(N_frames_per_character_train)

    assert np.sum([args.K_tanh, args.object_kernel_normalize, args.K_SE]) <= 1, \
        "At most one of GP kernel engineering flags can be used at once!"

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

        json.dump(dict_, open(chkpnt_dir + "/args.json", "wt"))

        # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data ======================
        iterator, train_init_op, test_action_init_op, \
            test_char_init_op = import_sprites(batch_size=args.batch_size, sprites_path=args.sprites_data_path,
                                               batch_size_test_char=batch_size_test_char)

        # get the batch
        frames, char_IDs, action_IDs = iterator.get_next()

        if 'yes' in args.repr_nn_pretrain:  # a higher batch size is used here (due to the higher learning rate used)
            iterator_repr_nn, train_init_op_repr_nn, \
                test_action_init_op_repr_nn, _ = import_sprites(batch_size=args.batch_size_repr_nn,
                                                                sprites_path=args.sprites_data_path,
                                                                batch_size_test_char=batch_size_test_char)
            # get the batch
            frames_repr_nn, char_IDs_repr_nn, _ = iterator_repr_nn.get_next()

        # ====================== 2) build ELBO graph ======================

        VAE = spritesVAE(L=args.L)

        # optimization params
        C_ma_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=())
        lagrange_mult_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=())
        alpha_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=())
        beta = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=())

        # placeholders for construction of auxiliary data
        segment_ids_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
        repeats_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))

        if "SVGPVAE" in args.elbo:  # SVGPVAE:

            # SVGP params
            titsias = 'Titsias' in args.elbo
            ip_joint = not args.ip_joint
            GPLVM_joint = not args.GPLVM_joint
            GP_joint = not args.GP_joint

            if args.PCA:
                GPLVM_init, IP_init = sprites_PCA_init(path_train_dict=args.sprites_data_path + 'sprites_train_dict.p',
                                                       m=args.m, L_action=args.L_action, L_character=args.L_character)
            else:  # Gaussian initialization.
                GPLVM_init = np.random.normal(0, 1.5, N_actions*args.L_action).reshape(N_actions, args.L_action)
                IP_init = np.random.normal(0, 1.5, N_actions*args.m*(args.L_action +
                                                                     args.L_character)).reshape(N_actions*args.m,
                                                                                                args.L_action +
                                                                                                args.L_character)
            # SVGP init
            SVGP_ = spritesSVGP(titsias=titsias, fixed_inducing_points=ip_joint,
                                initial_inducing_points=IP_init, name='main',
                                jitter=args.jitter, N_train=N_train,L_action=args.L_action,
                                initial_GPLVM_action=GPLVM_init, fixed_GPLVM_action=GPLVM_joint,
                                K_obj_normalize=args.object_kernel_normalize,
                                L=args.L, K_tanh=args.K_tanh, K_SE=args.K_SE, fixed_GP_params=GP_joint)

            # (character) representation network init
            repr_NN = sprites_representation_network(L=args.L_character)

            elbo, recon_loss, KL_term, inside_elbo, ce_term, p_m, p_v, qnet_mu, qnet_var, recon_images, \
            inside_elbo_recon, inside_elbo_kl, _, \
            C_ma, lagrange_mult, _ = forward_pass_SVGPVAE(data_batch=(frames, action_IDs),
                                                       beta=beta,
                                                       vae=VAE,
                                                       svgp=SVGP_,
                                                       C_ma=C_ma_placeholder,
                                                       lagrange_mult=lagrange_mult_placeholder,
                                                       alpha=alpha_placeholder,
                                                       kappa=np.sqrt(args.kappa_squared),
                                                       clipping_qs=args.clip_qs,
                                                       GECO=args.GECO,
                                                       repr_NN=repr_NN,
                                                       segment_ids=segment_ids_placeholder,
                                                       repeats=repeats_placeholder,
                                                       MC_estimators=args.MC_estimators)

            # forward pass standard VAE (for training regime from CASALE: VAE-GP-joint)
            recon_loss_VAE, KL_term_VAE, elbo_VAE, recon_images_VAE, qnet_mu_VAE, \
                qnet_var_VAE, _ = forward_pass_standard_VAE_rotated_mnist(data_batch=(frames, action_IDs), vae=VAE,
                                                                          clipping_qs=args.clip_qs)

            # graph for the pretraining of the representation neural network
            if 'yes' in args.repr_nn_pretrain:
                classification_layer = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(args.L_character),
                                                                                       dtype=tf.float32,
                                                                                       name="repr_NN_class_1"),
                                                            tf.keras.layers.Dense(1000, name="repr_NN_class_2")])

                loss_pretrain_repr = forward_pass_pretraining_repr_NN(frames_repr_nn, char_IDs_repr_nn,
                                                                      repr_NN, classification_layer)

                loss_pretrain_repr_test, acc = forward_pass_pretraining_repr_NN(frames_repr_nn, char_IDs_repr_nn,
                                                                                repr_NN, classification_layer,
                                                                                test_pipeline=True)

        elif args.elbo == "VAE":  # plain VAE
            recon_loss, KL_term, elbo, recon_images, \
            qnet_mu, qnet_var, _ = forward_pass_standard_VAE_rotated_mnist(data_batch=(frames, action_IDs), vae=VAE,
                                                                           clipping_qs=args.clip_qs)

        else:
            raise ValueError

        # conditional generation test loss and predictions
        if "SVGPVAE" in args.elbo:

            # encode training data
            train_aux_data_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype,
                                                                  shape=(N_train, 1 + args.L_character))
            train_encodings_means_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=(N_train, args.L))
            train_encodings_vars_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=(N_train, args.L))

            qnet_mu_train, qnet_var_train, \
                aux_data_train = batching_encode_SVGPVAE(data_batch=(frames, action_IDs), vae=VAE,
                                                         clipping_qs=args.clip_qs, repr_nn=repr_NN,
                                                         segment_ids=segment_ids_placeholder,
                                                         repeats=repeats_placeholder)

            # precompute GP parameters that depend on training data
            # (so that we avoid recomputing them for every test batch)
            # K_mm = SVGP_.kernel_matrix(SVGP_.inducing_index_points, SVGP_.inducing_index_points)
            precomputed_K_mm_inv = tf.linalg.inv(SVGP_.kernel_matrix(SVGP_.inducing_index_points,
                                                                     SVGP_.inducing_index_points))
            precomputed_means, precomputed_vars = precompute_GP_params_SVGPVAE(train_encodings_means_placeholder,
                                                                               train_encodings_vars_placeholder,
                                                                               train_aux_data_placeholder, svgp=SVGP_)
            M = args.m*N_actions
            mean_terms_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=(args.L, M))
            var_terms_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=(args.L, M, M))
            K_mm_inv_placeholder = tf.compat.v1.placeholder(dtype=VAE.dtype, shape=(M, M))

            # cgen predictions for test_character data
            recon_images_test, target_images, \
            recon_loss_test, dummy_a, dummy_b = predict_SVGPVAE_sprites_test_character(data_batch=(frames, action_IDs),
                                                                     vae=VAE,
                                                                     svgp=SVGP_,
                                                                     repr_NN=repr_NN,
                                                                     mean_terms=mean_terms_placeholder,
                                                                     var_terms=var_terms_placeholder,
                                                                     N_context=args.N_context,
                                                                     N_actions=N_actions,
                                                                     batch_size_test=batch_size_test_char,
                                                                     segment_ids=segment_ids_placeholder,
                                                                     repeats=repeats_placeholder,
                                                                     K_mm_inv=K_mm_inv_placeholder)

            # cgen predictions for test_action data
            # TODO

            # GP diagnostics
            GP_GPLVM, GP_IP = SVGP_.variable_summary()
            repr_vecs = repr_NN.repr_nn(frames)  # output of the representation network for the current batch
            K_mm = SVGP_.kernel_matrix(SVGP_.inducing_index_points,
                                               SVGP_.inducing_index_points)

        # ====================== 3) optimizer ops ======================
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        if 'yes' in args.repr_nn_pretrain:
            repr_nn_train_vars = [x for x in train_vars if 'repr_NN' in x.name]
            if 'fixed' in args.repr_nn_pretrain:  # keep repr NN parameters fixed during SVGPVAE training
                train_vars = [x for x in train_vars if 'repr_NN' not in x.name]
            else:  # jointly optimize repr NN params during SVGPVAE training, only removing classification layer params
                train_vars = [x for x in train_vars if 'repr_NN_class' not in x.name]

            gradients_repr_NN = tf.gradients(loss_pretrain_repr, repr_nn_train_vars)
            optim_step_repr_NN = optimizer.apply_gradients(grads_and_vars=zip(gradients_repr_NN, repr_nn_train_vars),
                                                           global_step=global_step)


        # joint optimization
        if args.GECO:
            gradients_joint = tf.gradients(elbo, train_vars)
        else:
            # Minimizing negative elbo!
            gradients_joint = tf.gradients(-elbo, train_vars)

        if args.clip_grad:
            gradients_joint = [tf.clip_by_value(grad, -args.clip_grad_thres, args.clip_grad_thres) for grad in gradients_joint]

        optim_step_joint = optimizer.apply_gradients(grads_and_vars=zip(gradients_joint, train_vars),
                                                     global_step=global_step)
        if "SVGPVAE" in args.elbo:
            # GP optimization
            GP_vars = [x for x in train_vars if 'GP' in x.name]
            if args.GECO:
                gradients_GP = tf.gradients(elbo, GP_vars)
            else:
                # Minimizing negative elbo!
                gradients_GP = tf.gradients(-elbo, GP_vars)

            if args.clip_grad:
                # TODO: clip properly for IndexedSlices gradient array (action GPLVM vectors)
                gradients_GP = [tf.clip_by_value(grad, -args.clip_grad_thres, args.clip_grad_thres) for grad in gradients_GP]
                # gradients_GP = [tf.clip_by_value(grad, -1000.0, 1000.0) if tf.is_tensor(grad) else tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_GP]

            optim_step_GP = optimizer.apply_gradients(grads_and_vars=zip(gradients_GP, GP_vars),
                                                      global_step=global_step)

            # VAE optimization
            VAE_vars = [x for x in train_vars if not 'GP' in x.name]
            gradients_VAE = tf.gradients(-elbo_VAE, VAE_vars)  # here we optimize standard ELBO objective
            if args.clip_grad:
                gradients_VAE = [tf.clip_by_value(grad, -args.clip_grad_thres, args.clip_grad_thres) for grad in gradients_VAE]

            optim_step_VAE = optimizer.apply_gradients(grads_and_vars=zip(gradients_VAE, VAE_vars),
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
                              "C_ma",
                              "lagrange_mult"]

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

                res_vars_GP = [GP_GPLVM,
                               GP_IP,
                               repr_vecs,
                               K_mm]

                res_names_GP = ['GPLVM action vectors', 'inducing points', 'repr_vecs', 'K_mm']

                res_saver_VAE = pandas_res_saver(res_file_VAE, res_names_VAE)
                res_saver_GP = pandas_res_saver(res_file_GP, res_names_GP)

            res_saver = pandas_res_saver(res_file, res_names)

        # ====================== 5) print and init trainable params ======================
        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        if 'yes' in args.repr_nn_pretrain:
            init_op_local = tf.local_variables_initializer()
            print_trainable_vars(repr_nn_train_vars)

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

            if 'yes' in args.repr_nn_pretrain:
                sess.run(init_op_local)

                start_time = time.time()
                print("Start of a pretraining of the representation NN.")
                for epoch in range(args.nr_epochs_repr_nn):

                    sess.run(train_init_op_repr_nn)
                    losses = []
                    while True:
                        try:
                            _, g_s, loss_ = sess.run([optim_step_repr_NN, global_step, loss_pretrain_repr],
                                                     {lr: args.lr_repr_nn})
                            losses.append(loss_)
                        except tf.errors.OutOfRangeError:
                            print("Train ", epoch, np.mean(losses))
                            break

                    if epoch % 10 == 0:
                        sess.run(test_action_init_op_repr_nn)
                        losses_test, acc_test = [], []
                        while True:
                            try:
                                loss_test_, acc_ = sess.run([loss_pretrain_repr_test, acc])
                                losses_test.append(loss_test_)
                                acc_test.append(acc_)
                            except tf.errors.OutOfRangeError:
                                print("Test ", epoch, np.mean(losses_test))
                                print("Test ", epoch, np.mean(acc_test))
                                break

                end_time = time.time()
                print("End of a pretraining of the representation NN. Time elapsed: {}".format(end_time - start_time))

            # training loop
            first_step = True  # switch for initizalition of GECO algorithm
            C_ma_ = 0.0
            lagrange_mult_ = 1.0

            # specify segment ids and repeats arrays (for construction of auxiliary data)
            train_segment_ids, train_repeats = aux_data_sprites_utils(args.batch_size, N_frames_per_character_train,
                                                                      N_frames_per_character_train)
            test_recon_segment_ids, \
                test_recon_repeats = aux_data_sprites_utils(batch_size_test_char, N_actions, N_actions)

            test_cgen_segment_ids, \
                test_cgen_repeats = aux_data_sprites_utils(int(batch_size_test_char * args.N_context / N_actions),
                                                           args.N_context, N_actions - args.N_context)

            start_time = time.time()
            cgen_test_set_MSE = []
            for epoch in range(nr_epochs):

                # 7.1) set objective functions etc (support for different training regimes, handcrafted schedules etc)
                if "SVGPVAE" in args.elbo:
                    if training_regime[epoch] == "VAE":
                        optim_step = optim_step_VAE
                        elbo_main = elbo_VAE
                        recon_loss_main = recon_loss_VAE
                        recon_images_main = recon_images_VAE
                        lr_main = 0.001
                        beta_main = args.beta
                        gradients_main = gradients_VAE
                    elif training_regime[epoch] == "GP":
                        optim_step = optim_step_GP
                        elbo_main = elbo
                        beta_main = args.beta
                        lr_main = 0.01
                        recon_loss_main = recon_loss
                        recon_images_main = recon_images
                        gradients_main = gradients_GP
                    elif training_regime[epoch] == "joint":
                        optim_step = optim_step_joint
                        elbo_main = elbo
                        # handcrafted learning rate and beta schedules - to combat VAE posterior collapse for Hensman
                        if epoch < args.beta_schedule_switch:
                            beta_main = args.beta
                            lr_main = args.lr
                        else:
                            beta_main = args.beta / 10
                            lr_main = args.lr / 10
                        recon_loss_main = recon_loss
                        recon_images_main = recon_images
                        gradients_main = gradients_joint
                    else:
                        raise ValueError

                else:  # plain VAE
                    optim_step = optim_step_joint
                    elbo_main = elbo
                    beta_main = args.beta
                    recon_loss_main = recon_loss
                    recon_images_main = recon_images
                    lr_main = args.lr

                # 7.2) train for one epoch
                sess.run(train_init_op)
                elbos, losses = [], []
                start_time_epoch = time.time()
                while True:
                    try:
                        if args.GECO and "SVGPVAE" in args.elbo and training_regime[epoch] != 'VAE':
                            if first_step:
                                alpha = 0.0
                            else:
                                alpha = args.alpha
                            _, g_s_, elbo_, C_ma_, \
                                lagrange_mult_, recon_loss_, \
                                grads_ = sess.run([optim_step, global_step, elbo_main, C_ma,
                                                              lagrange_mult, recon_loss_main, gradients_main],
                                                                       {beta: beta_main, lr: lr_main,
                                                                        alpha_placeholder: alpha,
                                                                        C_ma_placeholder: C_ma_,
                                                                        lagrange_mult_placeholder: lagrange_mult_,
                                                                        segment_ids_placeholder: train_segment_ids,
                                                                        repeats_placeholder: train_repeats})
                        else:
                            _, g_s_, elbo_, recon_loss_, \
                                grads_ = sess.run([optim_step, global_step, elbo_main, recon_loss_main, gradients_main],
                                                                   {beta: beta_main, lr: lr_main,
                                                                    alpha_placeholder: args.alpha,
                                                                    C_ma_placeholder: C_ma_,
                                                                    lagrange_mult_placeholder: lagrange_mult_,
                                                                    segment_ids_placeholder: train_segment_ids,
                                                                    repeats_placeholder: train_repeats})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                        first_step = False  # switch for initialization of GECO algorithm
                        print("Global step: {}. ELBO {}. Loss: {}".format(g_s_,
                                                                          round(elbo_, 2),
                                                                          round(recon_loss_ / args.batch_size, 4)))
                        if "SVGPVAE" in args.elbo and training_regime[epoch] != 'VAE':
                            print("max grad (GPLVM vectors): {}".format(max([IndexedSlicesValue_to_numpy(arr).max()
                                                                             for arr in grads_ if
                                                                             not (type(arr) == np.ndarray or type(arr) == np.float32)])))
                            print("min grad (GPLVM vectors): {}".format(min([IndexedSlicesValue_to_numpy(arr).min()
                                                                             for arr in grads_ if
                                                                             not (type(arr) == np.ndarray or type(arr) == np.float32)])))

                        print("max grad (without GPLVM grads): {}".format(max([arr.max() for arr in grads_
                                                                               if type(arr) == np.ndarray])))
                        print("min grad (without GPLVM grads): {}".format(min([arr.min() for arr in grads_
                                                                               if type(arr) == np.ndarray])))

                    except tf.errors.OutOfRangeError:
                        if (epoch + 1) % 1 == 0:
                            regime = training_regime[epoch] if "SVGPVAE" in args.elbo else "VAE"
                            print('Epoch {}, opt regime {}, mean ELBO per batch: {}'.format(epoch, regime,
                                                                                            np.mean(elbos)))
                            MSE = np.sum(losses) / N_train
                            print('MSE loss on train set for epoch {} : {}'.format(epoch, MSE))

                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}, opt regime {}: {}".format(epoch,
                                                                                        regime,
                                                                                        end_time_epoch -
                                                                                        start_time_epoch))
                        break

                # 7.3) training diagnostics (pandas saver)
                if args.save and (epoch + 1) % 1 == 0:
                    if args.test_set_metrics:
                        sess.run(test_char_init_op)
                        segment_ids_ = test_recon_segment_ids
                        repeats_ = test_recon_repeats
                    else:
                        sess.run(train_init_op)
                        segment_ids_ = train_segment_ids
                        repeats_ = train_repeats
                    new_res = sess.run(res_vars, {beta: args.beta,
                                                  alpha_placeholder: args.alpha,
                                                  C_ma_placeholder: C_ma_,
                                                  lagrange_mult_placeholder: lagrange_mult_,
                                                  segment_ids_placeholder: segment_ids_,
                                                  repeats_placeholder: repeats_})
                    res_saver(new_res, 1)
                    if "SVGPVAE" in args.elbo:
                        new_res_GP = sess.run(res_vars_GP, {beta: args.beta,
                                                            alpha_placeholder: args.alpha,
                                                            C_ma_placeholder: C_ma_,
                                                            lagrange_mult_placeholder: lagrange_mult_,
                                                            segment_ids_placeholder: segment_ids_,
                                                            repeats_placeholder: repeats_})
                        res_saver_GP(new_res_GP, 1)

                        if training_regime[epoch] == 'VAE':
                            new_res = sess.run(res_vars_VAE, {beta: args.beta})
                            res_saver_VAE(new_res, 1)

                # 7.4) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % 5 == 0:

                    # 7.4.1) VAE reconstruction
                    # Here reconstructed images are not stored due to memory constraints
                    losses = []
                    sess.run(test_char_init_op)
                    while True:
                        try:
                            recon_loss_, recon_images_, frames_ = sess.run([recon_loss_main, recon_images_main, frames],
                                                                  {beta: beta_main,
                                                                   alpha_placeholder: args.alpha,
                                                                   C_ma_placeholder: C_ma_,
                                                                   lagrange_mult_placeholder: lagrange_mult_,
                                                                   segment_ids_placeholder: test_recon_segment_ids,
                                                                   repeats_placeholder: test_recon_repeats})
                            losses.append(recon_loss_)
                        except tf.errors.OutOfRangeError:
                            MSE = np.sum(losses) / N_test_character
                            print('MSE loss on test set for epoch {} : {}'.format(epoch, MSE))
                            plot_sprites(frames_, recon_images_,
                                         title="Epoch: {}. recon MSE test set:{}".format(epoch + 1,
                                                                                         round(MSE, 4)))
                            if args.show_pics:
                                plt.show()
                                plt.pause(0.01)
                            if args.save:
                                plt.savefig(pic_folder + str(g_s_) + "_recon.png")
                            break

                    # 7.4.2) conditional generation
                    if "SVGPVAE" in args.elbo:

                        # encode training data (in batches)
                        sess.run(train_init_op)
                        means_all_train, vars_all_train, aux_data_all_train = [], [], []
                        while True:
                            try:
                                qnet_mu_train_, qnet_var_train_, aux_data_train_ = sess.run([qnet_mu_train,
                                                                                             qnet_var_train,
                                                                                             aux_data_train],
                                                                        {segment_ids_placeholder: train_segment_ids,
                                                                         repeats_placeholder: train_repeats})
                                means_all_train.append(qnet_mu_train_)
                                vars_all_train.append(qnet_var_train_)
                                aux_data_all_train.append(aux_data_train_)
                            except tf.errors.OutOfRangeError:
                                break
                        means_all_train = np.concatenate(means_all_train, axis=0)
                        vars_all_train = np.concatenate(vars_all_train, axis=0)
                        aux_data_all_train = np.concatenate(aux_data_all_train, axis=0)

                        # precompute GP params that depend on the entire dataset
                        precomputed_means_, precomputed_vars_, \
                            precomputed_K_mm_inv_ = sess.run([precomputed_means, precomputed_vars, precomputed_K_mm_inv],
                                                            {train_aux_data_placeholder: aux_data_all_train,
                                                             train_encodings_means_placeholder: means_all_train,
                                                             train_encodings_vars_placeholder: vars_all_train})

                        # predict test data (in batches)
                        # due to high number of test images (around 6500),
                        # we only store images from the last batch in the memory (we keep overwriting pics_ variable)
                        sess.run(test_char_init_op)
                        recon_loss_cgen = []
                        while True:
                            try:
                                loss_, pics_, target_images_ = sess.run([recon_loss_test,
                                                                         recon_images_test,
                                                                         target_images],
                                                         {mean_terms_placeholder: precomputed_means_,
                                                          var_terms_placeholder: precomputed_vars_,
                                                          K_mm_inv_placeholder: precomputed_K_mm_inv_,
                                                          segment_ids_placeholder: test_cgen_segment_ids,
                                                          repeats_placeholder: test_cgen_repeats})
                                recon_loss_cgen.append(loss_)
                            except tf.errors.OutOfRangeError:
                                break
                        recon_loss_cgen = np.sum(recon_loss_cgen) / (N_test_character * (1- args.N_context / N_actions))

                        cgen_test_set_MSE.append((epoch, recon_loss_cgen))
                        print("Conditional generation MSE loss on test set for epoch {}: {}".format(epoch,
                                                                                                    recon_loss_cgen))

                        # save images from conditional generation (only if current model exhibits best performance)
                        best_cgen = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0][1]
                        if recon_loss_cgen <= best_cgen and args.save:
                            print("Saving generations. Epoch {}. Cgen MSE: {}".format(epoch, recon_loss_cgen))
                            with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                                pickle.dump(pics_, test_pickle)
                            with open(chkpnt_dir + '/cgen_images_target.p', 'wb') as test_pickle_target:
                                pickle.dump(target_images_, test_pickle_target)

                        plot_sprites(target_images_, pics_,
                                     title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_cgen, 4)))
                        if args.show_pics:
                            plt.show()
                            plt.pause(0.01)
                        if args.save:
                            plt.savefig(pic_folder + str(g_s_) + "_cgen.png")
                            with open(pic_folder + "test_metrics.txt", "a") as f:
                                f.write("{},{},{}\n".format(epoch + 1, round(MSE, 4), round(recon_loss_cgen, 4)))

                    # 7.4.3) save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(nr_epochs, round(end_time - start_time, 2)))

            # report best test set cgen MSE achieved throughout training
            best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                         best_cgen_MSE[1]))


if __name__ == "__main__":
    default_base_dir = os.getcwd()

    # =============== parser SPRITES data ===============

    # parser rotated MNIST data
    parser_sprites = argparse.ArgumentParser(description='Train SVGPVAE for SPRITES data.')
    parser_sprites.add_argument('--expid', type=str, default="debug_SPRITES", help='give this experiment a name')
    parser_sprites.add_argument('--base_dir', type=str, default=default_base_dir,
                                help='folder within a new dir is made for each run')
    parser_sprites.add_argument('--elbo', type=str, choices=['VAE', 'SVGPVAE_Hensman', 'SVGPVAE_Titsias'], default='VAE')
    parser_sprites.add_argument('--sprites_data_path', type=str, default='SPRITES data/',
                              help='Path where rotated MNIST data is stored.')
    parser_sprites.add_argument('--batch_size', type=int, default=64)
    parser_sprites.add_argument('--nr_epochs', type=int, default=20)
    parser_sprites.add_argument('--beta', type=float, default=0.001)
    parser_sprites.add_argument('--m', type=int, default=15, help="Number of character vectors per action.")
    parser_sprites.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser_sprites.add_argument('--ip_joint', action="store_true", help='Inducing points joint optimization.')
    parser_sprites.add_argument('--GPLVM_joint', action="store_true", help='GPLVM action vectors joint optimization.')
    parser_sprites.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser_sprites.add_argument('--save_model_weights', action="store_true",
                              help='Save model weights. For debug purposes.')
    parser_sprites.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser_sprites.add_argument('--beta_schedule_switch', type=int, default=61)
    parser_sprites.add_argument('--opt_regime', type=str, default=['joint-120'], nargs="+")
    parser_sprites.add_argument('--L', type=int, default=128, help="Nr. of VAE latent channels.")
    parser_sprites.add_argument('--L_action', type=int, default=6, help="Dimension of GPLVM action vectors.")
    parser_sprites.add_argument('--L_character', type=int, default=16, help="Dimension of character vectors.")
    parser_sprites.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')
    parser_sprites.add_argument('--ram', type=float, default=0.33, help='fraction of GPU ram to use')
    parser_sprites.add_argument('--GECO', action='store_true', help='Use GECO algorithm for training.')
    parser_sprites.add_argument('--alpha', type=float, default=0.99, help='Moving average parameter for GECO.')
    parser_sprites.add_argument('--kappa_squared', type=float, default=0.033, help='Constraint parameter for GECO.')
    parser_sprites.add_argument('--jitter', type=float, default=0.000001, help='Jitter for numerical stability.')
    parser_sprites.add_argument('--PCA', action="store_true",
                              help='Use PCA embeddings for initialization of inducing points and GPLVM action vectors.')
    parser_sprites.add_argument('--N_context', type=int, default=50,
                                help="Number of context frames used in cgen prediction for test_character data.")
    parser_sprites.add_argument('--test_set_metrics', action='store_true',
    help='Calculate metrics on (1 batch of) test data. If false, metrics are calculated on (1 batch of) train data.')
    parser_sprites.add_argument('--clip_grad', action="store_true", help='Clip gradients.')
    parser_sprites.add_argument('--repr_nn_pretrain', type=str, choices=['no', 'yes_fixed', 'yes_joint'],
                                default='no', help='Pretraining regime for representation neural net.')
    parser_sprites.add_argument('--lr_repr_nn', type=float, default=0.01,
                                help='Learning rate for Adam optimizer for pretraining of representation neural net.')
    parser_sprites.add_argument('--nr_epochs_repr_nn', type=int, default=100,
                                help='Number of epochs for pretraining of representation neural net.')
    parser_sprites.add_argument('--batch_size_repr_nn', type=int, default=5000,
                                help='Batch size for pretraining of the representation neural net.')
    parser_sprites.add_argument('--object_kernel_normalize', action='store_true',
                                help='Normalize object (linear) kernel.')
    parser_sprites.add_argument('--MC_estimators', action='store_true',
                                help='Add N/b constant to batch estimators for parameters of q^Titsias.')
    parser_sprites.add_argument('--K_tanh', action='store_true',
                                help='Normalize a linear GP kernel using a tanh function.')
    parser_sprites.add_argument('--K_SE', action='store_true',
                                help='Use the squared-exponential kernel instead of the linear kernel.')
    parser_sprites.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser_sprites.add_argument('--clip_grad_thres', type=float, default=1000.0, help="Value at which to clip gradients.")

    # =============== SPRITES data experiments ==========================
    args_sprites = parser_sprites.parse_args()
    dict_ = vars(args_sprites)
    run_experiment_sprites_SVGPVAE(args_sprites, dict_)

