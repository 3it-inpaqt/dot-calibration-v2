from typing import Sequence

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from typing import List

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

def plot_resistance_distribution(layers: dict, title: str, file_name: str) -> None:
    """
    Plot the distribution of model's parameters values after the conversion as resistances.

    Args:
        layers: A dictionary that contains all resistance values.
        title: The title of the plot.
        file_name: The name of the plot file.
    """

    # Move every resistance into a flat list
    resistances = []
    for layer_r in layers.values():
        # If it is a double list, flatten it
        if isinstance(layer_r[0], List):
            layer_r = [j for sub in layer_r for j in sub]

        for r_plus, r_minus in layer_r:
            resistances.extend([r_plus, r_minus])

    plt.hist(resistances, bins=100)
    plt.xlabel('Resistances values (Ohm)')
    plt.ylabel('Number')
    plt.title(title)

    save_plot(file_name)