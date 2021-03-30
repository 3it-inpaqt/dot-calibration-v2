from math import ceil, sqrt
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon
from torch.utils.data import DataLoader, Dataset

from utils.output import save_plot
from utils.settings import settings

REGION_SHORT = {
    '0_electron': '0',
    '1_electron': '1',
    '2_electrons': '2',
    '3_electrons': '3',
    '4+_electrons': '4+'
}


def plot_diagram(x_i, y_i, pixels, image_name: str, interpolation_method: str, pixel_size: float,
                 charge_regions: Iterable[Tuple[str, Polygon]] = None, transition_lines: Iterable[LineString] = None,
                 focus_area: Optional[Tuple] = None) -> None:
    """
    Plot the interpolated image.

    :param x_i: The x coordinates of the pixels (post interpolation)
    :param y_i: The y coordinates of the pixels (post interpolation)
    :param pixels: The list of pixels to plot
    :param image_name: The name of the image, used for plot title
    :param interpolation_method: The pixels interpolation method, used for plot title
    :param pixel_size: The size of pixels, in voltage, used for plot title
    :param charge_regions: The charge region annotations to draw on top of the image
    :param transition_lines: The transition line annotation to draw on top of the image
    :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
    """

    plt.imshow(pixels, interpolation='none', cmap='copper',
               extent=[np.min(x_i), np.max(x_i), np.min(y_i), np.max(y_i)])

    if charge_regions is not None:
        for label, polygon in charge_regions:
            polygon_x, polygon_y = polygon.exterior.coords.xy
            plt.fill(polygon_x, polygon_y, 'b', alpha=.3, edgecolor='b', snap=True)
            label_x, label_y = list(polygon.centroid.coords)[0]
            plt.text(label_x, label_y, REGION_SHORT[label], ha="center", va="center", color='b')

    if transition_lines is not None:
        for line in transition_lines:
            line_x, line_y = line.coords.xy
            plt.plot(line_x, line_y, color='lime', alpha=.5)

    plt.title(f'{image_name}\ninterpolated ({interpolation_method}) - pixel size {round(pixel_size, 10) * 1_000}mV')
    plt.xlabel('Gate 1 (V)')
    plt.xticks(rotation=30)
    plt.ylabel('Gate 2 (V)')
    plt.tight_layout()

    if focus_area:
        plt.axis(focus_area)

    save_plot(f'diagram_{image_name}')


def plot_patch_sample(dataset: Dataset, number_per_class: int) -> None:
    """
    Plot randomly sampled patches grouped by class.

    :param dataset: The patches dataset to sample from.
    :param number_per_class: The number of sample per class.
    """
    # Local import to avoid circular mess
    from datasets.qdsd import QDSDLines

    # Data loader for random sample
    data_loader = DataLoader(dataset, shuffle=True)

    nb_classes = len(QDSDLines.classes)
    data_per_class = [list() for _ in range(nb_classes)]

    # Random sample
    for data, label in data_loader:
        label = int(label)  # Convert boolean to integer
        if len(data_per_class[label]) < number_per_class:
            data_per_class[label].append(data)

            # Stop of we sampled enough data
            if all([len(cl) == number_per_class for cl in data_per_class]):
                break

    # Create subplots
    fig, axs = plt.subplots(nrows=nb_classes, ncols=number_per_class,
                            figsize=(number_per_class * 2, nb_classes * 2 + 1))

    for i, cl in enumerate(data_per_class):
        axs[i, 0].set_title(f'{number_per_class} examples of "{QDSDLines.classes[i]}"', loc='left',
                            fontsize='xx-large', fontweight='bold')
        for j, class_data in enumerate(cl):
            axs[i, j].imshow(class_data.reshape(settings.patch_size_x, settings.patch_size_y),
                             interpolation='none',
                             cmap='copper')

            axs[i, j].axis('off')

    save_plot('patch_sample')


def plot_samples(samples: List, title: str, file_name: str) -> None:
    """
    Plot a group of patches.

    :param samples: The list of patches to plot.
    :param title: The title of the plot.
    :param file_name: The file name of the plot if saved.
    """
    plot_length = ceil(sqrt(len(samples)))

    # Create subplots
    fig, axs = plt.subplots(nrows=plot_length, ncols=plot_length, figsize=(plot_length * 2, plot_length * 2 + 1))

    for i, s in enumerate(samples):
        axs[i // plot_length, i % plot_length].imshow(s.reshape(settings.patch_size_x, settings.patch_size_y),
                                                      interpolation='none',
                                                      cmap='copper')

        axs[i // plot_length, i % plot_length].axis('off')

    fig.suptitle(f'{title}\nSample of {len(samples)} patches')

    save_plot(f'sample_{file_name}')
