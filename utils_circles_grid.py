import numpy as np
import matplotlib .pyplot as plt
from utils import MSE_rotation
from copy import copy


def pixelate_frame(xy, px=32, py=32, r=3):
        """
        takes a single x,y pixel point and converts to binary image
        with ball centered at x,y.
        """
        x = xy[0]
        y = xy[1]

        sq_x = (np.arange(px) - x)**2
        sq_y = (np.arange(py) - y)**2

        sq = sq_x.reshape(1,-1) + sq_y.reshape(-1,1)

        rr = r*r

        image = 1*(sq < rr)

        return image

def pixelate_series(XY0, px=32, py=32, r=3):

    XY = copy(XY0)

    # convert trajectories to pixel dims
    XY[:,0] = XY[:,0] * (px/5) + (0.5*px)
    XY[:,1] = XY[:,1] * (py/5) + (0.5*py)

    pix = lambda xy: pixelate_frame(xy, px=px, py=py, r=r)
    vid = [pix(xy) for xy in XY]
    return np.asarray(vid)


def plot_heatmap(vid, ax):
    """
    Plots a video with all frames overlayed and shaded by time.
    args:
        vid: tmax, px, py
    returns:
        flat_vid: px, py
    """
    vid = np.array([(t+4)*v for t,v in enumerate(vid)])
    flat_vid = 1-np.max(vid, 0)*(1/(4+30))
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False)
    ax.imshow(flat_vid, cmap='Greys',origin='lower')


def Make_circles(px=32, py=32, r=3, tmax=30):
    """
    Constructs two circles of latent points, and renders them.
    """

    # number of points in each ring
    n_in = 8
    n_ot = 10

    # make the latents
    x_r = [[0], 
        np.sin(2*np.pi*np.arange(n_in)/n_in), 
        2*np.sin(2*np.pi*np.arange(n_ot)/n_ot)]
    x_r = np.concatenate(x_r)

    y_r = [[0], 
        np.cos(2*np.pi*np.arange(n_in)/n_in), 
        2*np.cos(2*np.pi*np.arange(n_ot)/n_ot)]
    y_r = np.concatenate(y_r)

    traj = np.vstack([x_r, y_r]).T #(19, 2)
    # traj = np.concatenate([traj, traj[:(tmax-19)]], axis=0) # padded to (30, 2)
    traj = np.append(traj, np.zeros((tmax-19, 2)), axis=0) # padded to (tmax, 2)

    # make the set of images
    # (1, tmax, 32, 32)
    V_c = pixelate_series(traj, px=px, py=py, r=r)
    V_c = V_c[None,:,:,:]
    
    return traj, V_c


def plot_circle(ax1, ax, rot_qnet=None):
    """
    Plots two circles of points
    Args:
        traj0: (1, 30, 2) ground truth set of latent points in circles
        ax: matplotlib axes object to plot onto
        qnet_mu: predicted latents (batch, tmax, 2)
    """


    ax.clear()
    ax1.clear()
    
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False)

    traj0, V_c = Make_circles()
        
    plot_heatmap(V_c[0,:,:,:], ax1)

    ax.scatter(traj0[:,0],traj0[:,1],color='blue')

    ax.plot(traj0[1:9,0], traj0[1:9,1], color='blue')
    t_x1 = np.array([traj0[1,0], traj0[8,0]])
    t_y1 = np.array([traj0[1,1], traj0[8,1]])
    ax.plot(t_x1, t_y1, color='blue')

    ax.plot(traj0[9:19,0], traj0[9:19,1],color='blue')
    t_x2 = np.array([traj0[9,0], traj0[18,0]])
    t_y2 = np.array([traj0[9,1], traj0[18,1]])
    ax.plot(t_x2, t_y2, color='blue')
    
    if rot_qnet is not None:

        # import pdb; pdb.set_trace()
        # rot_qnet, _, _, _ = MSE_rotation(qnet_mu[:1,:19,:], traj0[None, :19,:])
        ax.scatter(rot_qnet[:,0],rot_qnet[:,1],color= 'orange', zorder=10)
            
        ax.plot(rot_qnet[1:9,0], rot_qnet[1:9,1], color='orange', zorder=10)
        q_x1 = np.array([rot_qnet[1,0], rot_qnet[8,0]])
        q_y1 = np.array([rot_qnet[1,1], rot_qnet[8,1]])
        ax.plot(q_x1, q_y1, color='orange', zorder=10)

        ax.plot(rot_qnet[9:19,0], rot_qnet[9:19,1],color='orange', zorder=10)
        q_x2 = np.array([rot_qnet[9,0], rot_qnet[18,0]])
        q_y2 = np.array([rot_qnet[9,1], rot_qnet[18,1]])
        ax.plot(q_x2, q_y2, color='orange', zorder=10)


def Make_squares(px=32, py=32, r=3, tmax=30):
    base_lin = (np.arange(5)-2) 
    sq_x = np.hstack([base_lin for i in range(5)])
    sq_y = sq_x.reshape((5,5)).T.reshape((-1))

    sq_tr = np.vstack([sq_x, sq_y]).T + 0.01

    # (tmax, 2)
    # sq_tr = np.vstack([sq_tr, sq_tr[:5, :]]) + 0.01
    sq_tr = np.append(sq_tr, np.zeros((tmax-25, 2)), axis=0)

    # (1, tmax, 32, 32)
    V_sq = pixelate_series(sq_tr)
    V_sq = V_sq[None, :,:,:]

    return sq_tr, V_sq


def plot_square(ax0, ax, rot_qsq=None):
    """
    Plots a lattice of points, true and predicted latents.
    Args:
        q_sq: predicted latent positions
        sq_tr: true latent positions
        ax: axes to plot onto
    """

    ax0.clear()
    ax.clear()


    sq_tr, V_sq = Make_squares()

    plot_heatmap(V_sq[0,:,:,:], ax0)

    ax.scatter(sq_tr[:,0], sq_tr[:,1], color='blue')

    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
        right=False,
        labelleft=False,
    labelbottom=False)
    


    for i in range(5):
        for j in range(4):
            dx = np.array( [sq_tr[i+5*j, 0], sq_tr[i+5*(j+1), 0]] )
            dy = np.array( [sq_tr[i+5*j, 1], sq_tr[i+5*(j+1), 1]] )
            ax.plot(dx, dy, color='blue')

            dx = np.array( [sq_tr[j+5*i, 0], sq_tr[1+j+5*i, 0]] )
            dy = np.array( [sq_tr[j+5*i, 1], sq_tr[1+j+5*i, 1]] )
            ax.plot(dx, dy, color='blue')

    if rot_qsq is not None:            
        # rot_qsq, _, _, _ = MSE_rotation(q_sq[:1,:25,:], sq_tr[None, :25,:])
        rot_qsq = rot_qsq[:, :]
        ax.scatter(rot_qsq[:, 0],rot_qsq[:, 1], color='orange', zorder=10)
        for i in range(5):
            for j in range(4):
                dx = np.array( [rot_qsq[i+5*j, 0], rot_qsq[i+5*(j+1), 0]] )
                dy = np.array( [rot_qsq[i+5*j, 1], rot_qsq[i+5*(j+1), 1]] )
                ax.plot(dx, dy, color='orange', zorder=10)

                dx = np.array( [rot_qsq[j+5*i, 0], rot_qsq[1+j+5*i, 0]] )
                dy = np.array( [rot_qsq[j+5*i, 1], rot_qsq[1+j+5*i, 1]] )
                ax.plot(dx, dy, color='orange', zorder=10)
    

if __name__=="__main__":

    fig, ax = plt.subplots(2,2, figsize=(8,8))

    xy_c, V_c = Make_circles()

    xy_sq, V_sq = Make_squares()


    # import pdb; pdb.set_trace()

    # plot_heatmap(V_c[0,:,:,:], ax[0][0])
    # plot_heatmap(V_sq[0,:,:,:], ax[1][0])

    q_sq = xy_sq + 0.1*np.random.normal(size=xy_sq.shape)
    q_sq = q_sq[None, :, :]

    q_c = xy_c + 0.1*np.random.normal(size=xy_c.shape)
    q_c = q_c[None, :,:]

    # plot_circle(xy_c, ax[0][1], q_c)
    # plot_square(xy_sq, ax[1][1], q_sq)

    plot_circle(ax[0][0], ax[0][1], q_c)
    plot_square(ax[1][0], ax[1][1], q_sq)

    plt.show()