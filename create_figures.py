from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
import joypy
from tqdm import tqdm


def learning_rate_schedule(t, tau, beta, mu, mode):
    """
    The learning rate as a function of time.

    Args:
        t (float): The time
        tau (float): The total time of the process
        mu (float): The mobility
        mode (string): The mode, either constant, continuous, or discontinuous

    Returns:
        (float) The learning rate
    """
    if mode == 'constant':
        return 10
    elif mode == 'continuous':
        return 1 / beta / (tau - t + 1 / mu)
    elif mode == 'discontinuous':
        return 1 / beta / (tau - t + 2 / mu ** 2)
    else:
        raise ValueError(f'Unknown learning rate: {mode}')


def train(cfg):
    """
    Train the Harmonic Trap EBM.

    Args:
        cfg (dict): The config parameters for training.

    Returns:
        records (list of pd.DataFrames): The states of the system at each time step.
    """

    # initialize
    x = np.random.normal(loc=cfg["u0"], size=cfg["num_particles"])
    theta = cfg["theta_0"]
    records = []

    # initial state
    record = pd.DataFrame({'x': x})
    record['theta'] = theta
    record['iteration'] = -2
    records.append(record)

    # initial discontinuous jump
    if cfg["learning_rate_mode"] == "discontinuous":
        eta = (x.mean() - theta) / (cfg["theta_star"] - x.mean()) + 1 / (cfg['mu'] * cfg['tau'] + 2/cfg['mu'])
        theta += cfg["beta"] * eta * (cfg["theta_star"] - x.mean())

        record = pd.DataFrame({'x': x})
        record['theta'] = theta
        record['iteration'] = -1
        records.append(record)
    
    # training loop
    for i in tqdm(range(cfg["num_iters"])):

        # Langevin step
        dt = cfg["tau"] / (cfg["num_iters"] - 1)
        dx = cfg["mu"] * (x - theta) * dt
        scale = np.sqrt(2 * cfg["mu"] * dt / cfg["beta"])
        noise = np.random.normal(size=x.shape, scale=scale)
        x += -dx + noise

        # Maximum Likelihood step
        t = i * dt
        eta = learning_rate_schedule(t, cfg["tau"], cfg["beta"], cfg["mu"], mode=cfg["learning_rate_mode"])
        u = x.mean()
        theta += cfg["beta"] * eta * (cfg["theta_star"] - u) * dt

        record = pd.DataFrame({'x': x})
        record['theta'] = theta
        record['iteration'] = i
        records.append(record)

    # final discontinuous jump
    if cfg["learning_rate_mode"] == "discontinuous":
        eta = (1 - cfg["mu"] / 2)
        theta += cfg["beta"] * eta * (cfg["theta_star"] - x.mean())

        record = pd.DataFrame({'x': x})
        record['theta'] = theta
        record['iteration'] = i+1
        records.append(record.copy())

    return records


def ridge_plot(records, cfg):
    """
    Create a ridge plot from records of EBM training in 1 dimension.

    Args:
        records (list of pd.DataFrames): The states of the system at each time step.
        cfg (dict): The config parameters for training.
    
    Returns:
        fig (matplotlib.pyplot.figure): The ridge plot figure.
    """

    df = pd.concat(records)

    # left and right justify the ridges
    df.loc[df.groupby('iteration')['x'].head(1).index, 'x'] = min(df.x.min(), df.theta.min())
    df.loc[df.groupby('iteration')['x'].tail(1).index, 'x'] = max(df.x.max(), max(df.theta.max(), cfg['theta_star']) + (2 if cfg["learning_rate_mode"] in ["continuous", "discontinuous"] else 0))

    # make the ridge plot
    fig, axes = joypy.joyplot(
        df, 
        by="iteration", 
        column="x", 
        xlabels=False, 
        ylabels=False, 
        bins=100, 
        linecolor=(1, 1, 1, 1), 
        alpha=0.2, 
        range_style='own',
        clip_on=False, 
        linewidth=1, 
        tails=0, 
        legend=False, 
        figsize=(16,8)
    )

    # plot the model trajectory
    xs = [record['theta'].iloc[0] for record in records]
    ys = [ax.get_position().ymin for ax in axes[:-1]]
    last_axis = axes[-1]
    box = axes[-1].get_position()
    box.y0 -= 0.03
    box.y1 += 0.065
    last_axis.set_position(box)
    last_axis.plot(xs, ys, 'r', label='model $a$')

    # plot the model and data distributions
    centers =  [(cfg['theta_star'], ys[0]), (xs[0], ys[0])]
    for x0, y0 in centers:
        x_para = np.linspace(x0 - 2, x0 + 2, 100)
        y_para = [y0 + 0.03 + 0.07 * np.exp(-(x - x0) ** 2) for x in x_para]
        x_para, y_para = zip(*[(x,y) for x,y in zip(x_para, y_para) if y - y0 < 0.1])
        last_axis.plot(x_para, y_para, 'w', alpha=0)  # this is invisible, to fix height of figure
        y_para = [y - 0.03 for y in y_para]
        last_axis.plot(x_para, y_para, 'k', alpha=0.5)

    # plot the ground truth as a vertical dotted line
    last_axis.axvline(cfg["theta_star"], ls='dotted', ymin=0.045, ymax=0.955, c='k')

    last_axis.text(cfg["theta_0"] + 1, box.y1 - 0.1, '$p_\\text{m}$', fontsize=18)
    last_axis.text(cfg["theta_star"] + 1, box.y1 - 0.1, '$p_\\text{d}$', fontsize=18)
    shift = -1.2 if cfg["u0"] == cfg["theta_0"] else 0.8
    last_axis.text(cfg["u0"] + shift, box.y1 - 0.1, '$p_\\text{s}$', fontsize=18)

    # time arrow
    x = min(df.x.min(), df.theta.min()) - 0.4
    y1 = axes[0].get_position().y0
    y2 = y1 / 2
    time_arrow = FancyArrowPatch(posA=(x, 0.01), posB=(x, y2), arrowstyle='<|-', color='0.5', mutation_scale=20, shrinkA=0, shrinkB=0)
    last_axis.add_artist(time_arrow)
    time_arrow = FancyArrowPatch(posA=(x, y2 + 0.06 ), posB=(x, y1), arrowstyle='-', color='0.5', mutation_scale=20, shrinkA=0, shrinkB=0)
    last_axis.add_artist(time_arrow)
    last_axis.text(x - 0.12, y2 + 0.02, '$t$', fontsize=18)

    return fig


def _x_range(data, extra=0.2):
    """ Compute the x_range, i.e., the values for which the
        density will be computed. It should be slightly larger than
        the max and min so that the plot actually reaches 0, and
        also has a bit of a tail on both sides.
    """
    try:
        sample_range = np.nanmax(data) - np.nanmin(data)
    except ValueError:
        return []
    if sample_range < 1e-6:
        return [np.nanmin(data), np.nanmax(data)]
    return np.linspace(np.nanmin(data) - extra*sample_range,
                       np.nanmax(data) + extra*sample_range, 1000)


def trim_whitespace(filename):
    """
    Overwrite a saved image with one having whitespace cropped.
    """

    img = Image.open(filename)
    array = np.asarray(img)
    array = array[:, :, :3] # Drop the alpha channel
    
    idx = np.where(array - 255)[0:2] # Drop the color when finding edges
    bbox = list(map(min, idx))[::-1] + list(map(max,idx))[::-1]
    cropped = img.crop(bbox)

    cropped.save(filename)


def plot_excess_work():
    """
    Plot the excess work for various protocols.
    """
    fig, ax = plt.subplots(figsize=(8,6))

    theta_star = 0
    theta_0 = -1
    mu = 1
    tau = 5
    eta = 0.5

    ts = np.linspace(0, tau, 1000)
    
    # minimal entropy
    m = (theta_star - theta_0) / tau
    W_c = [t * (m ** 2) / mu  for t in ts]
    
    # optimal
    m = (theta_star - theta_0) / (mu * tau + 2) 
    W_0 = (m / mu) ** 2
    W_d = [W_0 + t * (m ** 2) / mu  for t in ts]
    W_d[0] = 0
    W_d[-1] += W_0

    # slow driving
    m = (theta_star - theta_0) / tau
    W_sd = [t * (m ** 2) / mu - (m / mu) ** 2 * (1 - np.exp(-mu * t))  for t in ts]

    # quasi static
    W_qs = [eta * (theta_star - theta_0) ** 2 / 2 / mu * (1 - np.exp(-2 * eta * t) ) for t in ts]
    
    
    with plt.style.context('seaborn-v0_8-dark'):
        ax.plot(ts / tau, W_d, label='discontinuous', linestyle='solid')
        ax.plot(ts / tau, W_sd, label='slow driving', linestyle='dotted')
        ax.plot(ts / tau, W_c, label='continuous', linestyle='dashed')
        ax.plot(ts / tau, W_qs, label='quasi-static', linestyle='dashdot')

        ax.legend()
        ax.set_xlabel('Time', fontdict={'fontsize': 18})
        ax.set_ylabel('Excess           \nWork           ', rotation='horizontal', fontdict={'fontsize': 18})
        ax.tick_params(axis='y', length=0, labelleft=False)

    return fig


def main():
    """
    Create the figures, show them to the screen, and save them as image files.
    """

    rcParams['mathtext.fontset'] = 'stix'
    rcParams['font.family'] = 'STIXGeneral'

    fig_params = {
        'transparent': True, 
        'edgecolor': "none", 
        'bbox_inches': 'tight', 
 #       'pad_inches': 0,
    }

    # contstant learning rate
    cfg = {
        'theta_star': 0,
        'theta_0': 10,
        'u0': -10,
        'tau': 1,
        'num_particles': 1000,
        'num_iters': 100,
        'mu': 10,
        'beta': 1,
        'learning_rate_mode': "constant",
    }
    records = train(cfg)
    fig = ridge_plot(records, cfg)
    filename = "ebm_ridge_constant.png"
    fig.savefig(filename, **fig_params)
    trim_whitespace(filename)
    plt.show()

    # quasi static protocol
    cfg = {
        'theta_star': 0,
        'theta_0': -10,
        'u0': -10,
        'tau': 1,
        'num_particles': 1000,
        'num_iters': 100,
        'mu': 100,
        'beta': 1,
        'learning_rate_mode': "constant",
    }
    records = train(cfg)
    fig = ridge_plot(records, cfg)
    filename = "ebm_ridge_quasi_static.png"
    fig.savefig(filename, **fig_params)
    trim_whitespace(filename)
    plt.show()

    # optimal continuous learning rate
    cfg = {
        'theta_star': 0,
        'theta_0': -5,
        'u0': -10,
        'tau': 1,
        'num_particles': 1000,
        'num_iters': 100,
        'mu': 1,
        'beta': 1,
        'learning_rate_mode': "continuous",
    }
    records = train(cfg)
    fig = ridge_plot(records, cfg)
    filename = "ebm_ridge_continuous.png"
    fig.savefig(filename, **fig_params)
    trim_whitespace(filename)
    plt.show()

    # optimal discontinuous learning rate
    cfg = {
        'theta_star': 0,
        'theta_0': -10,
        'u0': -10,
        'tau': 1,
        'num_particles': 1000,
        'num_iters': 100,
        'mu': 1,
        'beta': 1,
        'learning_rate_mode': "discontinuous",
    }
    records = train(cfg)
    fig = ridge_plot(records, cfg)
    filename = "ebm_ridge_discontinuous.png"
    fig.savefig(filename, **fig_params)
    trim_whitespace(filename)
    plt.show()

    # slow driving
    cfg = {
        'theta_star': 0,
        'theta_0': -10,
        'u0': -10,
        'tau': 1,
        'num_particles': 1000,
        'num_iters': 100,
        'mu': 5,
        'beta': 1,
        'learning_rate_mode': "continuous",
    }
    records = train(cfg)
    fig = ridge_plot(records, cfg)
    filename = "ebm_ridge_slow_driving.png"
    fig.savefig(filename, **fig_params)
    trim_whitespace(filename)
    plt.show()

    fig = plot_excess_work()
    fig.savefig("excess_work.png", **fig_params)
    plt.show()


if __name__ == "__main__":
    main()