import argparse
import time
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from utils import Make_Video_batch, make_checkpoint_folder, pandas_res_saver, \
                  build_video_batch_graph, plot_latents, MSE_rotation
from utils_circles_grid import Make_circles, Make_squares, plot_circle, plot_square
from SVGPVAE_model import SVGP, build_SVGPVAE_elbo_graph
from GPVAE_Pearce_model import build_pearce_elbo_graphs

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def run_experiment(args):
    """
    Moving ball experiment.

    :param args:
    :return:
    """
    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta0)

        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        print("\nCheckpoint Directory:\n"+str(chkpnt_dir)+"\n")

    # Data synthesis settings
    batch = 35
    tmax = args.tmax
    px = 32
    py = 32
    r = 3
    vid_lt = args.vidlt
    m = args.m

    if args.elbo == 'VAE':
        # A GP prior with a RBF kernel and a very small length scale reduces to the standard Gaussian prior
        model_lt = 0.001
    else:
        model_lt = args.modellt

    assert model_lt == vid_lt or args.GP_joint or args.elbo == 'VAE', \
        "GP params of data and model should match. Except when \
         doing a joint optimization of GP parameters or when fitting normal VAE."

    # Load/create batches of reproducible videos
    if os.path.isfile(args.base_dir + "/Test_Batches_{}_{}.pkl".format(vid_lt, tmax)):
        with open(args.base_dir + "/Test_Batches_{}_{}.pkl".format(vid_lt, tmax), "rb") as f:
            Test_Batches = pickle.load(f)
    else:
        make_batch = lambda s: Make_Video_batch(tmax=tmax, px=px, py=py, lt=vid_lt, batch=batch, seed=s, r=r)
        Test_Batches = [make_batch(s) for s in range(10)]
        with open(args.base_dir + "/Test_Batches_{}_{}.pkl".format(vid_lt, tmax), "wb") as f:
            pickle.dump(Test_Batches, f)

    # Initialise a plots
    # this plot displays a  batch of videos + latents + reconstructions
    if args.save or args.show_pics:
        fig, ax = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
        plt.ion()

    if args.squares_circles:
        truth_c, V_c = Make_circles(tmax=tmax); batch_V_c = np.tile(V_c, (batch,1,1,1))
        truth_sq, V_sq = Make_squares(tmax=tmax); batch_V_sq = np.tile(V_sq, (batch,1,1,1))

    # make sure everything is created in the same graph!
    graph = tf.Graph()
    with graph.as_default():

        # Make all the graphs
        beta = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

        vid_batch = build_video_batch_graph(batch=batch, tmax=tmax, px=px, py=py, r=r, lt=vid_lt)

        if args.elbo in ['GPVAE_Pearce', 'VAE', 'NP']:
            elbo, rec, pkl, p_m, \
                p_v, q_m, q_v, pred_vid, \
                l_GP_x, l_GP_y, _ = build_pearce_elbo_graphs(vid_batch, beta, type_elbo=args.elbo, lt=model_lt,
                                                             GP_joint=args.GP_joint, GP_init=args.GP_init)

        else:  # SVGPVAE_Titsias, SVGPVAE_Hensman

            titsias = 'Titsias' in args.elbo
            fixed_gp_params = not args.GP_joint
            fixed_inducing_points = not args.ip_joint
            svgp_x_ = SVGP(titsias=titsias, num_inducing_points=m,
                           fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='x',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)
            svgp_y_ = SVGP(titsias=titsias, num_inducing_points=m, fixed_inducing_points=fixed_inducing_points,
                           tmin=1, tmax=tmax, vidlt=vid_lt, fixed_gp_params=fixed_gp_params, name='y',
                           jitter=args.jitter, ip_min=args.ip_min, ip_max=args.ip_max, GP_init=args.GP_init)

            elbo, rec, pkl, l3_elbo, ce_term,\
            p_m, p_v, q_m, q_v, pred_vid, l_GP_x, l_GP_y,\
            l3_elbo_recon, l3_elbo_kl, inducing_points_x, inducing_points_y, \
            gp_cov_full_mean_x, gp_cov_full_mean_y, _ = build_SVGPVAE_elbo_graph(vid_batch, beta,
                                                                                 svgp_x=svgp_x_, svgp_y=svgp_y_,
                                                                                 clipping_qs=args.clip_qs)

        # The actual loss functions
        loss = -tf.reduce_mean(elbo)
        e_elb = tf.reduce_mean(elbo)
        e_pkl = tf.reduce_mean(pkl)
        e_rec = tf.reduce_mean(rec)
        if 'SVGPVAE' in args.elbo:
            e_l3_elbo = tf.reduce_mean(l3_elbo)
            e_ce_term = tf.reduce_mean(ce_term)
            e_l3_elbo_recon = tf.reduce_mean(l3_elbo_recon)
            e_l3_elbo_kl = tf.reduce_mean(l3_elbo_kl)

        # Add optimizer ops to graph (minimizing neg elbo!), print out trainable vars
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.compat.v1.train.AdamOptimizer()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if args.clip_grad:
            gradients = tf.gradients(loss, train_vars)
            gradients = [tf.clip_by_value(grad, -100000.0, 100000.0) for grad in gradients]
            optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                                   global_step=global_step)

        else:
            optim_step = optimizer.minimize(loss=loss,
                                            var_list=train_vars,
                                            global_step=global_step)

        print("\n\nTrainable variables:")
        for v in train_vars:
            print(v)

        # Initializer ops for the graph and saver
        init_op = tf.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()

        if args.save:
            # Results to be tracked and Pandas saver
            res_vars = [global_step,
                        loss,
                        e_elb,
                        e_rec,
                        e_pkl,
                        tf.math.reduce_min(q_v),
                        tf.math.reduce_max(q_v),
                        tf.math.reduce_min(p_v),
                        tf.math.reduce_max(p_v),
                        tf.math.reduce_min(q_m),
                        tf.math.reduce_max(q_m),
                        tf.math.reduce_min(p_m),
                        tf.math.reduce_max(p_m),
                        l_GP_x,
                        l_GP_y]
            if 'SVGPVAE' in args.elbo:
                res_vars += [e_l3_elbo,
                             e_ce_term,
                             e_l3_elbo_recon,
                             e_l3_elbo_kl,
                             inducing_points_x,
                             inducing_points_y,
                             gp_cov_full_mean_x,
                             gp_cov_full_mean_y]
            res_names= ["Step",
                        "Loss",
                        "Train ELBO",
                        "Train Reconstruction",
                        "Train Prior KL",
                        "min qs_var",
                        "max qs_var",
                        "min q_var",
                        "max q_var",
                        'min qs_mean',
                        'max qs_mean',
                        'min q_mean',
                        'max q_mean',
                        "l_GP_x",
                        "l_GP_y"]
            if 'SVGPVAE' in args.elbo:
                res_names += ['SVGP elbo',
                              'ce term',
                              'SVGP elbo recon',
                              'SVGP elbo KL',
                               'inducing_points_x',
                              'inducing_points_y',
                              'gp_cov_full_mean_x',
                              'gp_cov_full_mean_y']
            res_names += ["MSE",  "Beta",  "Time"]
            res_saver = pandas_res_saver(res_file, res_names)

        # Now let's start doing some computation!
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            # Attempt to restore weights
            try:
                saver.restore(sess, tf.train.latest_checkpoint(chkpnt_dir))
                print("\n\nRestored Model Weights")
            except:
                sess.run(init_op)
                print("\n\nInitialised Model Weights")

            # Start training that elbo!

            for t in range(args.steps):

                # Train: do an optim step
                _, g_s = sess.run([optim_step, global_step], {beta: args.beta0})

                # Print out diagnostics/tracking
                if g_s % 1000 == 0:
                    TD = Test_Batches[0][1]
                    if 'SVGPVAE' in args.elbo:
                        test_elbo, e_rec_i, e_pkl_i, e_l3_elbo_i, e_ce_term_i, e_l3_elbo_recon_i, e_l3_elbo_kl_i = \
                            sess.run([e_elb, e_rec, e_pkl, e_l3_elbo, e_ce_term, e_l3_elbo_recon, e_l3_elbo_kl],
                                     {vid_batch: TD, beta: 1.0})

                    else:
                        test_elbo, e_rec_i, e_pkl_i = sess.run([e_elb, e_rec, e_pkl], {vid_batch: TD, beta: 1.0})

                    test_qv, test_pv, test_pm, test_qm = sess.run([q_v, p_v, p_m, q_m], {vid_batch: TD, beta: 1.0})

                    print(str(g_s)+": elbo "+str(test_elbo))
                    print("Recon term: {}. KL term: {}.".format(e_rec_i, e_pkl_i))
                    if 'SVGPVAE' in args.elbo:
                        print("L{} elbo term: {}. CE term: {}.".format(2 if titsias else 3, e_l3_elbo_i, e_ce_term_i))
                        if not titsias:
                            print("L3 elbo recon term: {}. L3 elbo KL term: {}.".format(e_l3_elbo_recon_i,
                                                                                              e_l3_elbo_kl_i))
                    print("VAE posterior variance range: min {}, max  {}".format(np.min(test_qv), np.max(test_qv)))
                    print("VAE posterior mean range: min {}, max {}".format(np.min(test_qm), np.max(test_qm)))
                    print("GP approx posterior variance range: min {}, max {}".format(np.min(test_pv), np.max(test_pv)))
                    print("GP approx posterior mean range: min {}, max {}".format(np.min(test_pm), np.max(test_pm)))

                # Save elbo, recon, priorKL....
                if g_s % 1000 == 0:
                    TT, TD = Test_Batches[0]
                    p_m_i, p_v_i = sess.run([p_m, p_v], {vid_batch: TD, beta: 1})

                    _, _, MSE, _ = MSE_rotation(p_m_i, TT, p_v_i)
                    print('MSE : {}'.format(MSE))

                    if args.save:
                        new_res = sess.run(res_vars, {vid_batch: TD, beta: 1})
                        new_res += [MSE, args.beta0, time.time()]
                        res_saver(new_res)

                # show plot and occasionally save
                if g_s % 10000 == 0 and (args.save or args.show_pics):
                    # [[ax_ij.clear() for ax_ij in ax_i] for ax_i in ax]
                    TT, TD = Test_Batches[0]
                    reconpath, reconvar, reconvid = sess.run([p_m, p_v, pred_vid], {vid_batch:TD, beta:1})

                    rp, W, MSE, rv = MSE_rotation(reconpath, TT, reconvar)
                    _ = plot_latents(TD, TT, reconvid, rp, rv, ax=ax, nplots=4)
                    # plt.tight_layout()
                    plt.draw()
                    fig.suptitle(str(g_s)+' ELBO: ' + str(test_elbo))

                    if args.squares_circles:
                        q_m_c = sess.run(q_m, {vid_batch: batch_V_c})
                        q_m_sq = sess.run(q_m, {vid_batch: batch_V_sq})

                        # import pdb; pdb.set_trace()

                        q_m_c = np.hstack([ q_m_c[0,:,:], np.ones((tmax, 1))])
                        rot_qnet_c = np.matmul(q_m_c, W)
                        plot_circle(ax[3][0], ax[3][1], rot_qnet_c)

                        q_m_sq = np.hstack([ q_m_sq[0,:,:], np.ones((tmax, 1))])
                        rot_qnet_sq = np.matmul(q_m_sq, W)
                        plot_square(ax[3][2], ax[3][3], rot_qnet_sq)

                    if args.show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if args.save:
                        plt.savefig(pic_folder + str(g_s).zfill(6)+".png")

                # Save NN weights
                if args.save and g_s % 50000 == 0:
                    saver.save(sess, chkpnt_dir+"model", global_step=g_s)
                    print("\n\nModel Saved: " + chkpnt_dir +"\n\n")

                # Save predicted trajectories for plotting purposes
                if args.save and g_s % 50000 == 0:
                    preds = (TD, TT, reconvid, rp, rv)
                    pickle.dump(preds, open(chkpnt_dir + "preds/preds_{}.p".format(g_s), "wb"))


if __name__=="__main__":

    default_base_dir = os.getcwd()

    parser = argparse.ArgumentParser(description='Moving ball experiment')
    parser.add_argument('--steps', type=int, default=25000, help='Number of steps of Adam')
    parser.add_argument('--beta0', type=float, default=1, help='initial beta annealing value')
    parser.add_argument('--elbo', type=str, choices=['GPVAE_Pearce', 'VAE', 'NP', 'SVGPVAE_Hensman', 'SVGPVAE_Titsias'],
                        default='GPVAE_Pearce',
                        help='Structured Inf Nets ELBO or Neural Processes ELBO')
    parser.add_argument('--modellt', type=float, default=2, help='time scale of model to fit to data')
    parser.add_argument('--base_dir', type=str, default=default_base_dir, help='folder within a new dir is made for each run')
    parser.add_argument('--expid', type=str, default="debug", help='give this experiment a name')
    parser.add_argument('--ram', type=float, default=0.5, help='fraction of GPU ram to use')
    parser.add_argument('--seed', type=int, default=None, help='seed for rng')
    parser.add_argument('--tmax', type=int, default=30, help='length of videos')
    parser.add_argument('--m', type=int, default=10, help='number of inducing points')
    parser.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser.add_argument('--ip_joint', action="store_true", help='Inducing points joint optimization.')
    parser.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')

    parser.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser.add_argument('--squares_circles', action="store_true", help='Whether or not to plot squares and circles.')

    parser.add_argument('--ip_min', type=int, default=1, help='ip start')
    parser.add_argument('--ip_max', type=int, default=30, help='ip end')
    parser.add_argument('--jitter', type=float, default=1e-9, help='noise for GP operations (inverse, cholesky)')
    parser.add_argument('--clip_grad', action="store_true", help='Whether or not to clip gradients.')
    parser.add_argument('--vidlt', type=float, default=2, help='time scale for data generation')
    parser.add_argument('--GP_init', type=float, default=2,
                        help='Initial value for GP kernel length scale. Used when running --GP_joint .')

    args = parser.parse_args()

    run_experiment(args)



