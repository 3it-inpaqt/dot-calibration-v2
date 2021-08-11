from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from utils.output import save_plot


def plot_bayesian_parameters(means: Sequence[float], stds: Sequence[float], title: str, file_name: str) -> None:
    # Why is the std negative?
    # Why there is a "sampler" as a trainable parameter?

    x_axis = np.arange(-15, 15, 0.01)

    # Create subplots
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(3 * 2, 3 * 2 + 1))

    for row in range(3):
        for col in range(3):
            i = row * 3 + col
            ax = axs[row, col]
            ax.plot(x_axis, norm.pdf(x_axis, means[i], abs(stds[i])))

    fig.suptitle(title)

    save_plot(f'bayesian_parameters_{file_name}')
