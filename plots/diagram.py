from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon

from utils.output import save_plot

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

    save_plot(f'diagram-{image_name}')
