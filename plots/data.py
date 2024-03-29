import io
from copy import copy
from functools import partial
from itertools import chain
from math import ceil, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.image import AxesImage
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.ticker import ScalarFormatter
from shapely.geometry import LineString, Polygon
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from classes.data_structures import StepHistoryEntry
from utils.misc import get_nb_loader_workers
from utils.output import save_gif, save_plot, save_video
from utils.settings import settings

CLASS_COLORS = [
    'tab:red',  # False (0)
    'blue'  # True (1)
]
GOOD_COLOR = 'green'
ERROR_COLOR = 'tab:red'
SOFT_ERROR_COLOR = 'blueviolet'
UNKNOWN_COLOR = 'dimgray'
NOT_SCANNED_COLOR = 'lightgrey'

PIXELS_CMAP = matplotlib.cm.copper
PIXELS_CMAP.set_bad(color=NOT_SCANNED_COLOR)

MEASURE_UNIT = {
    'michel_pioro_ladriere': r'$\mathregular{I_{SET}}$',
    'eva_dupont_ferrier': r'$\mathregular{I_{SET}}$',
    'louis_gaudreau': r'$\mathregular{I_{QPC}}$'
}


def plot_diagram(x_i: Sequence[float],
                 y_i: Sequence[float],
                 pixels: Optional[Tensor] = None,
                 title: Optional[str] = None,
                 fog_of_war: bool = False,
                 charge_regions: List[Tuple["ChargeRegime", Polygon]] = None,
                 transition_lines: List[LineString] = None,
                 scan_history: List["StepHistoryEntry"] = None,
                 scan_history_mode: Literal['classes', 'error'] = 'classes',
                 scan_history_alpha: Optional[Literal['uncertainty'] | int] = None,
                 focus_area: Optional[bool | Tuple[float, float, float, float]] = None,
                 focus_area_title: str = 'Focus area',
                 final_volt_coord: Tuple[float, float] | List[Tuple[str, str, Iterable[Tuple[float, float]]]] = None,
                 text: Optional[str | bool] = None,
                 scale_bars: bool = False,
                 legend: Optional[bool] = True,
                 vmin: float = None,
                 vmax: float = None,
                 diagram_boundaries: Optional[Tuple[float, float, float, float]] = None,
                 file_name: str = None,
                 allow_overwrite: bool = False,
                 save_in_buffer: bool = False
                 ) -> Optional[Path | io.BytesIO]:
    """
    Versatile function to plot a stability diagram with the following elements:
    - The diagram itself
    - The tuning steps (class, error and uncertainty)
    - Focus area
    - The labels (line and charge area)
    - A scale bar and legends
    - Text description

    :param x_i: An array that gives the x-axis voltage values of the diagram per coordinate.
    :param y_i: An array that gives the y-axis voltage values of the diagram per coordinate.
    :param pixels: The pixel values of the diagram as measured current.
     If None, the diagram will be blank.
    :param title: The main title of the diagram.
    :param fog_of_war: If True and the scan history is defined, hide the pixels that were not measured yet.
    :param charge_regions: The charge region annotations to draw on top of the image.
    :param transition_lines: The transition line annotation to draw on top of the image.
    :param scan_history: A list of scan history entries from a tuning procedure.
     The type of information plot will depend on the scan_history_mode and scan_history_alpha.
    :param scan_history_mode: Influence what information plot from the scan history.
        - 'classes': Plot the inferred class for each tuning step.
        - 'error': Plot the error of the inferred class for each tuning step.
    :param scan_history_alpha: Influence the transparency of the scan history plot.
        - None: No transparency.
        - 'uncertainty': The transparency is proportional to the uncertainty of the inferred class.
        - int: No transparency for the last N plotted steps, then the transparency increased for older steps.
    :param focus_area: Add a subplot to the right of the diagram to show a zoomed-in section of the diagram.
        - If None, no focus area subplot.
        - If a tuple, the focus area is chosen to be the given coordinates (x_start, x_end, y_start, y_end).
        - If True, the focus area is automatically chosen to be the last patch.
    :param focus_area_title: Title for the focus area subplot.
    :param final_volt_coord: The final voltage coordinates that is within the target area, according to the tuning
     procedure (x, y).
     If it is a list of tuple as (label, color, list of cood), it means that we want to show multiple final coordinates.
    :param text: A text to display at the left of the diagram.
        - If None, no text subplot.
        - If True, the text is automatically created to describe the step history.
        - If str, the text is the given string.
    :param scale_bars: If True, add a scale bar to the right of the diagram. The scale can either represent the current
     range of the pixel, or the uncertainty of classification. If the focus area is enabled, also add a scale bar.
    :param legend: If True, add a legend in a subplot at the bottom of the figure.
    :param vmin: If set, define the minimal value of the colorscale of the diagram's pixels.
    :param vmax: If set, define the maximal value of the colorscale of the diagram's pixels.
    :param diagram_boundaries: If set, define the limits of the main diagram (x_start, x_end, y_start, y_end).
    :param file_name: The name of the output file (without the extension).
    :param allow_overwrite: If True, allow overwriting existing output files.
     If False, add a number to the file name to avoid overwriting.
    :param save_in_buffer: If True, save the figure in a BytesIO buffer instead of a file.
    :return: The path where the plot is saved, or None if not saved. If save_in_buffer is True, return image bytes
     instead of the path.
    """
    # Choose the appropriate layout
    show_focus_area_ax = focus_area is not None
    show_text_ax = text is not None
    show_legend_ax = legend is not None
    fig, diagram_ax, focus_area_ax, text_ax, legend_ax = _get_layout(show_focus_area_ax, show_text_ax, show_legend_ax)
    custom_legend = []

    # Set the plot coordinates to fit the image with the axes, with the center of pixels aligned with the coordinates.
    half_p = settings.pixel_size / 2
    axes_matching = [x_i[0] - half_p, x_i[-1] + half_p, y_i[0] - half_p, y_i[-1] + half_p]

    # Build the main diagram subplot
    _plot_diagram_ax(diagram_ax, x_i, y_i, pixels, title, scan_history, fog_of_war, scale_bars,
                     scan_history_alpha == 'uncertainty', scan_history_mode,
                     vmin, vmax, axes_matching, diagram_boundaries)

    # Show labels if provided
    if charge_regions or transition_lines:
        custom_legend.extend(_plot_labels(diagram_ax, charge_regions, transition_lines))

    # Build the focus area subplot
    if show_focus_area_ax:
        focus_vmin, focus_vmax = vmin, vmax
        sub_scale_bar = False
        if focus_area is True:
            # Get last scan coordinates and convert them to voltage.
            x_start, x_end, y_start, y_end = scan_history[-1].get_area_coord()
            focus_area = (x_i[x_start], x_i[x_end - 1], y_i[y_start], y_i[y_end - 1])
            if scale_bars:
                # If we show the scale bars, we can plot a different color range
                focus_pixels = pixels[y_start: y_end, x_start: x_end]
                focus_vmin = np.nanmin(focus_pixels).item()
                focus_vmax = np.nanmax(focus_pixels).item()
                sub_scale_bar = True

        _plot_focus_area_ax(focus_area_ax, diagram_ax, pixels, focus_area_title, focus_area, sub_scale_bar,
                            focus_vmin, focus_vmax, axes_matching)

    # Plot the scan history visualization
    if scan_history_mode and scan_history and len(scan_history) > 0:
        _plot_scan_history(diagram_ax, focus_area_ax, x_i, y_i, scan_history, scan_history_mode, scan_history_alpha,
                           pixels is None)

    if final_volt_coord is not None:
        _plot_final_coord(diagram_ax, focus_area_ax, final_volt_coord)

    if show_text_ax:
        if not isinstance(text, str):
            # If the text is not a string, it means that we want to generate the text from the scan history
            text = StepHistoryEntry.get_text_description(scan_history)
        _plot_text_ax(text_ax, text)

    if show_legend_ax:
        _plot_legend_ax(legend_ax, diagram_ax, custom_legend, pixels is not None)

    # Save the plot
    return save_plot(file_name, allow_overwrite=allow_overwrite, save_in_buffer=save_in_buffer, figure=fig)


def _get_layout(show_focus_area_ax, show_text_ax, show_legend_ax):
    # Optional axes
    focus_area_ax = text_ax = legend_ax = None

    # Temporary change default the axe style (avoid white ticks)
    with sns.axes_style("ticks"):
        fig = plt.figure(figsize=(16, 9))
        spec = fig.add_gridspec(10, 10)

        y_bottom = 9
        # Layout with legend at the bottom
        if show_legend_ax:
            y_bottom = 8
            legend_ax = fig.add_subplot(spec[y_bottom:, :])

        # 2 columns layout
        if show_focus_area_ax or show_text_ax:
            x_end_col_1 = 6
            diagram_ax = fig.add_subplot(spec[:y_bottom, :x_end_col_1])
            if show_focus_area_ax:
                focus_area_ax = fig.add_subplot(spec[:3, x_end_col_1:])
            if show_text_ax:
                text_ax = fig.add_subplot(spec[3 if show_focus_area_ax else 0:y_bottom, x_end_col_1:])

        # 1 column layout
        else:
            diagram_ax = fig.add_subplot(spec[:y_bottom, :])

    return fig, diagram_ax, focus_area_ax, text_ax, legend_ax


def _plot_diagram_ax(ax, x_i: Sequence[float], y_i: Sequence[float], pixels: Optional[Tensor], title: Optional[str],
                     scan_history: List["StepHistoryEntry"], fog_of_war: bool, scale_bar: bool,
                     scale_bar_uncertainty: bool, history_mode: str, vmin: float, vmax: float,
                     axes_matching: List[float],
                     diagram_boundaries: Optional[Tuple[float, float, float, float]]) -> None:
    # Subplot title
    if title:
        ax.set_title(title)

    # If defined, set boundaries for the diagram axis
    if diagram_boundaries:
        x_start_v, x_end_v, y_start_v, y_end_v = diagram_boundaries
        ax.set_xlim(x_start_v, x_end_v)
        ax.set_ylim(y_start_v, y_end_v)

    # If no pixels provided, plot a blank image to allow other information on the same format
    if pixels is None:
        cmap = LinearSegmentedColormap.from_list('', ['white', 'white'])
        ax.imshow(np.zeros((len(x_i), len(y_i))), cmap=cmap, extent=axes_matching)

        # Add uncertainty scalebar if needed
        if scale_bar and scale_bar_uncertainty:
            _add_uncertainty_scale_bar(ax, history_mode)

        return

    # If the fog of war is enabled, mask the pixels that have not been scanned yet according to the scan history
    if fog_of_war and scan_history is not None:
        mask = np.full_like(pixels, True)
        for scan in scan_history:
            x, y = scan.coordinates
            mask[y: y + settings.patch_size_y, x:x + settings.patch_size_x] = False
        pixels = np.ma.masked_array(pixels, mask)

    # Plot the pixels
    im = ax.imshow(pixels, interpolation='none', cmap=PIXELS_CMAP, extent=axes_matching, vmin=vmin, vmax=vmax,
                   origin='lower')

    # Add the pixel current scale bar
    if scale_bar:
        _add_scale_bar(im, ax)


def _plot_focus_area_ax(focus_ax, diagram_ax, pixels, title, focus_area, scale_bar: bool, vmin: Optional[float],
                        vmax: Optional[float], axes_matching: List[float]):
    # Subplot title
    if title:
        focus_ax.set_title(title)

    # Get coordinates and add half-pixel to match with the border of pixel (instead of the middle).
    half_p = settings.pixel_size / 2
    x_start, x_end, y_start, y_end = focus_area
    x_start -= half_p
    y_start -= half_p
    x_end += half_p
    y_end += half_p

    im = focus_ax.imshow(pixels, interpolation='none', cmap=PIXELS_CMAP, extent=axes_matching, vmin=vmin, vmax=vmax,
                         origin='lower')

    # Add the scale bar
    if scale_bar:
        _add_scale_bar(im, focus_ax, 0.05)

    focus_ax.set_xlim(x_start, x_end)
    focus_ax.set_ylim(y_start, y_end)

    # Show the location of the current focus area in the main diagram using a black rectangle
    loc = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=1, zorder=9999999,
                            edgecolor='black', label=title, facecolor='none', alpha=0.8)
    diagram_ax.add_patch(loc)


def _plot_labels(ax, charge_regions: List[Tuple["ChargeRegime", Polygon]], transition_lines: List[LineString]):
    labels_handlers = []
    z = 999999  # Z order placed between pixels and focus area

    if charge_regions:
        for i, (regime, polygon) in enumerate(charge_regions):
            polygon_x, polygon_y = polygon.exterior.coords.xy
            ax.fill(polygon_x, polygon_y, facecolor=(0, 0, 0.5, 0.3), edgecolor=(0, 0, 0.5, 0.8), snap=True, zorder=z)
            label_x, label_y = list(polygon.centroid.coords)[0]
            params = dict(x=label_x, y=label_y, ha="center", va="center", color='b', weight='bold', zorder=z + 2,
                          bbox=dict(boxstyle='round', pad=0.2, facecolor='w', alpha=0.5, edgecolor='w'))
            ax.text(**params, s=str(regime))

            if i == 0:
                # Create custom label for charge regime
                labels_handlers.append((Text(**params, text='N'), 'Charge regime'))

    if transition_lines:
        for i, line in enumerate(transition_lines):
            line_x, line_y = line.coords.xy
            ax.plot(line_x, line_y, color='lime', label='Line annotation', zorder=z + 1)

    return labels_handlers


def _plot_scan_history(diagram_ax, focus_ax, x_i, y_i, scan_history, scan_history_mode, scan_history_alpha,
                       full_squares):
    from datasets.qdsd import QDSDLines
    nb_scan = len(scan_history)
    half_p = settings.pixel_size / 2
    detection_size_x_v = (settings.patch_size_x - settings.label_offset_x * 2) * settings.pixel_size
    detection_size_y_v = (settings.patch_size_y - settings.label_offset_y * 2) * settings.pixel_size

    for i, scan_entry in enumerate(reversed(scan_history)):
        x, y = scan_entry.coordinates
        # Initial scan square parameters
        color = 'black'
        square_params = {
            'xy': (x_i[x + settings.label_offset_x] - half_p, y_i[y + settings.label_offset_y] - half_p),
            'width': detection_size_x_v,
            'height': detection_size_y_v,
            'edgecolor': None,
            'label': None,
            'facecolor': None,
            'alpha': 1,
            'zorder': nb_scan - i
        }

        # Set the alpha of the scan square according to the mode.
        if scan_history_alpha == 'uncertainty':
            # Alpha is set according to the uncertainty of the classification
            square_params['alpha'] = scan_entry.model_confidence
        elif isinstance(scan_history_alpha, int):
            # Alpha is increasing after a specific number of scans
            if i < scan_history_alpha:
                square_params['alpha'] = 1
            else:
                square_params['alpha'] = (2 * scan_history_alpha - i) / (scan_history_alpha + 1)

            # If the alpha is very low, we don't need to plot the rest of the scan history
            if square_params['alpha'] < 0.01:
                break

        # Set the color and the label of the rectangle according to the scan history mode.
        match scan_history_mode:
            case 'classes':
                color = CLASS_COLORS[scan_entry.model_classification]
                square_params['label'] = f'Infer {QDSDLines.classes[scan_entry.model_classification]}'
            case 'error':
                # Patch color depending on the classification success
                if scan_history_alpha != 'uncertainty' and not scan_entry.is_above_confidence_threshold:
                    # If the uncertainty is not shown with alpha, we show it by a gray patch
                    color = UNKNOWN_COLOR
                    square_params['label'] = 'Unknown'
                elif scan_entry.is_classification_correct():
                    color = GOOD_COLOR
                    square_params['label'] = 'Good'
                elif scan_history_alpha != 'uncertainty' and scan_entry.is_classification_almost_correct():
                    # Soft error is not compatible with uncertainty
                    color = SOFT_ERROR_COLOR
                    square_params['label'] = 'Soft Error'
                else:
                    color = ERROR_COLOR
                    square_params['label'] = 'Error'
            case 'uncertainty':
                color = scan_entry.uncertainty_color

        if full_squares:
            # Filled patch
            square_params['facecolor'] = color
            square_params['edgecolor'] = 'none'
        else:
            # Empty patch to see the diagram behind
            square_params['edgecolor'] = color
            square_params['facecolor'] = 'none'

        rec = patches.Rectangle(linewidth=1, **square_params)
        diagram_ax.add_patch(rec)

        # If we reached the first scan (reverse order), we add a marker to show the start of the tuning
        if i == nb_scan - 1:
            diagram_ax.scatter(x=x_i[x + settings.patch_size_x // 2] - half_p,
                               y=y_i[y + settings.patch_size_x // 2] - half_p,
                               color='skyblue', marker='X', s=200, label='Start', alpha=square_params['alpha'],
                               zorder=0)

        # Show the rectangle label offset for the last scan in focus area
        if focus_ax and i == 0:
            square_params['facecolor'] = 'none'
            square_params['edgecolor'] = color
            focus_ax.add_patch(patches.Rectangle(linewidth=2, **square_params))


def _plot_text_ax(text_ax, text: str):
    text_ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=12,
                 fontfamily='monospace', transform=text_ax.transAxes)
    # Fancy background
    text_ax.axes.get_xaxis().set_visible(False)
    text_ax.axes.get_yaxis().set_visible(False)
    text_ax.set_facecolor('linen')
    plt.setp(text_ax.spines.values(), color='bisque')


def _plot_final_coord(diagram_ax, focus_area_ax,
                      final_volt_coord: Tuple[float, float] | List[Tuple[str, str, Iterable[Tuple[float, float]]]]):
    half_p = settings.pixel_size / 2
    z_order = 9999  # Stack the markers from the first to the last

    # If the final coordinate is a single point, we convert it to a dict to factorize the code
    if not isinstance(final_volt_coord, list):
        final_volt_coord = [('End', 'fuchsia', [final_volt_coord])]

    # Iterate over the different labels group
    for label, color, coords in final_volt_coord:
        # Iterate over the different coordinates in the group
        for last_x, last_y in coords:
            # Plot the final coordinate in the main diagram and / or the focus area
            for ax, cross_size, lw in ((diagram_ax, 200, 2), (focus_area_ax, 600, 4)):
                if ax:
                    # Make white borders using a slightly bigger marker under it
                    ax.scatter(x=last_x - half_p, y=last_y - half_p, color='w', marker='x', s=cross_size * 1.1,
                               linewidths=lw * 1.5, zorder=z_order - 1)
                    ax.scatter(x=last_x - half_p, y=last_y - half_p, color=color, marker='x', s=cross_size, label=label,
                               linewidths=lw, zorder=z_order)
                    z_order -= 2
                label = None  # Only show the label for the first coordinate of this group


def _plot_legend_ax(legend_ax, diagram_ax, custom_legend, pixel_info: bool = True):
    legend_ax.axis('off')
    label_handlers = dict()

    if pixel_info:
        # Add the pixel size to the legend
        label_handlers[f'Pixel size: {settings.pixel_size * 1_000} mV'] = patches.Rectangle((0, 0), 1, 1, alpha=0)

    # Extract legend elements from the diagram axes and the custom list
    for ax_handler, ax_label in chain(zip(*diagram_ax.get_legend_handles_labels()), custom_legend):
        # Remove possible duplicates
        if ax_label not in label_handlers:
            ax_handler.set_alpha(1)  # Avoid alpha in legend
            label_handlers[ax_label] = ax_handler

    if len(label_handlers) == 0:
        return

    class TextHandler(HandlerBase):
        """
        Custom legend handler for text field.
        From: https://stackoverflow.com/a/47382270/2666094
        """

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            h = copy(orig_handle)
            h.set_position((width / 2., height / 2.))
            h.set_transform(trans)
            h.set_ha("center")
            h.set_va("center")
            fp = orig_handle.get_font_properties().copy()
            fp.set_size(fontsize)
            h.set_font_properties(fp)
            return [h]

    # Add every the legend element to the legend axes
    legend_ax.legend(handles=label_handlers.values(), labels=label_handlers.keys(), ncols=6, loc='center',
                     handler_map={Text: TextHandler()})


def _add_scale_bar(im: AxesImage, ax, pad: float = 0.02) -> None:
    """
    Add a scale bar to an imshow plot.

    :param im: The axe image (output of the imshow function)
    :param ax: The axe used to plot the image
    :param pad: The padding of the scale bar
    """
    if settings.research_group in MEASURE_UNIT:
        measuring = MEASURE_UNIT[settings.research_group]
    else:
        measuring = 'I'

    # Custom formatter to fix the number of decimals and avoid flickering video animation
    class CustomScalarFormatter(ScalarFormatter):
        def _set_format(self):
            self.format = r'$\mathdefault{%1.2f}$'

    formatter = CustomScalarFormatter(useMathText=True)

    # Add the scale bar that way because we already are inside a subplot
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=pad, label=f'{measuring} (A)', format=formatter)
    cbar.ax.yaxis.set_offset_position('left')


def _add_uncertainty_scale_bar(ax, history_mode: str, pad: float = 0.02) -> None:
    """
    Add a scale bar to an imshow plot to represent the model uncertainty.

    :param ax: The axe used to plot the image
    :param history_mode: The mode of the scan history (classes or error)
    :param pad: The padding of the scale bar
    """
    formatter = ScalarFormatter(useMathText=True)

    if history_mode == 'error':
        # 3-colors gradient (green, white, red) for the errors uncertainty scale bar
        cmap = LinearSegmentedColormap.from_list('', [GOOD_COLOR, 'white', ERROR_COLOR])
    else:
        # 3-colors gradient (blue, white, red) for the classification uncertainty scale bar
        cmap = LinearSegmentedColormap.from_list('', [CLASS_COLORS[1], 'white', CLASS_COLORS[0]])
    norm = Normalize(vmin=-1, vmax=1)
    cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm),
                        ax=ax, shrink=0.85, pad=pad, format=formatter)
    cbar.outline.set_edgecolor('0.15')
    cbar.set_ticks([-1, 0, 1])

    # Bayesian uncertainty
    if settings.model_type.upper() in ['BCNN', 'BFF']:
        metric_map = {  # This plot is not compatible with not normalized uncertainty
            'norm_std': 'Normalized STD',
            'norm_entropy': 'Normalized entropy'
        }
        uncertainty_label = metric_map[settings.bayesian_confidence_metric]
        min_uncertainty_correct = min_uncertainty_line = min_uncertainty_no_line = 0
        max_uncertainty = 1

    # Heuristic uncertainty
    else:
        uncertainty_label = 'Model output'
        min_uncertainty_line = 1
        min_uncertainty_no_line = 0
        min_uncertainty_correct = f'{min_uncertainty_line} or {min_uncertainty_no_line}'
        max_uncertainty = 0.5

    if history_mode == 'error':
        cbar.set_ticklabels(
            [f'Good class\n{uncertainty_label}: {min_uncertainty_correct}\n(High confidence)',
             f'{uncertainty_label}: {max_uncertainty}\n(Low confidence)',
             f'Bad class\n{uncertainty_label}: {min_uncertainty_correct}\n(High confidence)'])
    else:
        cbar.set_ticklabels([f'Infer line\n{uncertainty_label}: {min_uncertainty_line}\n(High confidence)',
                             f'{uncertainty_label}: {max_uncertainty}\n(Low confidence)',
                             f'Infer no-line\n{uncertainty_label}: {min_uncertainty_no_line}\n(High confidence)'])

    cbar.ax.yaxis.set_offset_position('left')


def plot_diagram_step_animation(d: "Diagram", title: str, image_name: str, scan_history: List["StepHistoryEntry"],
                                final_volt_coord: Tuple[float, float]) -> None:
    """
    Plot an animation of the tuning procedure.

    :param d: The diagram to plot.
    :param title: The title of the main plot.
    :param image_name: The name of the image, used for plot title and file name.
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass).
    :param final_volt_coord: The final tuning coordinates as volt.
    """

    if settings.is_named_run() and (settings.save_gif or settings.save_video):
        values, x_axes, y_axes = d.get_values()
        from datasets.diagram_online import DiagramOnline
        is_online = isinstance(d, DiagramOnline)
        diagram_boundaries = d.get_cropped_boundaries() if is_online else None
        # Compute min / max here because numpy doesn't like to do this on multi thread (ignore NaN values)
        vmin = np.nanmin(values).item()
        vmax = np.nanmax(values).item()
        # The final areas where we assume the target regime is
        margin = settings.patch_size_x * settings.pixel_size * 2
        final_area_start_x = max(final_volt_coord[0] - margin, diagram_boundaries[0] if is_online else x_axes[0])
        final_area_end_x = min(final_volt_coord[0] + margin, diagram_boundaries[1] if is_online else x_axes[-1])
        final_area_start_y = max(final_volt_coord[1] - margin, diagram_boundaries[2] if is_online else y_axes[0])
        final_area_end_y = min(final_volt_coord[1] + margin, diagram_boundaries[3] if is_online else y_axes[-1])
        final_area = (final_area_start_x, final_area_end_x, final_area_start_y, final_area_end_y)
        # Animation speed => Time for an image (ms)
        base_fps = 200
        # Ratio of image to skip for the animation frames (1 means nothing skipped, 4 means 1 keep for 3 skip)
        rate_gif = 4
        rate_video = 1
        # Define frames to compute
        gif_frames_ids = list(range(1, len(scan_history), rate_gif)) if settings.save_gif else []
        video_frames_ids = list(range(1, len(scan_history), rate_video)) if settings.save_video else []
        frame_ids = list(dict.fromkeys(gif_frames_ids + video_frames_ids))  # Remove duplicate and keep order

        # Base arguments for final images
        common_kwargs = dict(
            x_i=x_axes, y_i=y_axes, pixels=values, title=title, scan_history=scan_history, focus_area=final_area,
            focus_area_title='End area', save_in_buffer=True, text=True, scale_bars=True, vmin=vmin, vmax=vmax,
            diagram_boundaries=diagram_boundaries
        )

        with Pool(get_nb_loader_workers()) as pool:
            # Generate images in parallel for speed. Use partial to set constant arguments.
            # The order of the arguments is important because scan_history is sent after the positional args.
            # Main animation frames
            async_result_main = pool.map_async(
                partial(plot_diagram, x_axes, y_axes, values, title, True, None, None, focus_area=True,
                        focus_area_title='Last patch', text=True, scan_history_alpha=8, scale_bars=True,
                        vmin=vmin, vmax=vmax, save_in_buffer=True, diagram_boundaries=diagram_boundaries),
                (scan_history[0:i] for i in frame_ids)
            )

            # Final frames
            async_result_end = [
                # Show diagram with all inferences and fog of war
                pool.apply_async(plot_diagram, kwds=common_kwargs | {'fog_of_war': True}),
                # Show diagram with tuning final coordinate and fog of war
                pool.apply_async(plot_diagram, kwds=common_kwargs | {'fog_of_war': True,
                                                                     'final_volt_coord': final_volt_coord}),
            ]

            # If online, we don't have any label to show
            if is_online:
                end_durations = [base_fps * 15, base_fps * 40]
            else:
                end_durations = [base_fps * 15, base_fps * 15, base_fps * 15, base_fps * 30, base_fps * 40]
                async_result_end += [
                    # Show full diagram with tuning final coordinate (no fog of war)
                    pool.apply_async(plot_diagram, kwds=common_kwargs | {'final_volt_coord': final_volt_coord}),
                    # Show full diagram with tuning final coordinate + line labels
                    pool.apply_async(plot_diagram, kwds=common_kwargs | {'final_volt_coord': final_volt_coord,
                                                                         'transition_lines': d.transition_lines}),
                    # Show full diagram with tuning final coordinate + line & regime labels
                    pool.apply_async(plot_diagram, kwds=common_kwargs | {'final_volt_coord': final_volt_coord,
                                                                         'transition_lines': d.transition_lines,
                                                                         'charge_regions': d.charge_areas}),
                ]

            # Wait for the processes to finish and get results
            pool.close()
            pool.join()
            main_frames = async_result_main.get()
            end_frames = [res.get() for res in async_result_end]

        if settings.save_gif:
            # List of image bytes for the animation
            frames_gif = [main_frames[frame_ids.index(f_id)] for f_id in gif_frames_ids] + end_frames
            # List of duration for each image (ms)
            durations_gif = [base_fps * rate_gif] * len(gif_frames_ids) + end_durations
            durations_gif[0] = base_fps * 2
            save_gif(frames_gif, image_name, duration=durations_gif)

        if settings.save_video:
            # List of image bytes for the animation
            frames_video = [main_frames[frame_ids.index(f_id)] for f_id in video_frames_ids] + end_frames
            # List of duration for each image (ms)
            durations_video = [base_fps * rate_video] * len(video_frames_ids) + end_durations
            durations_video[0] = base_fps * 4
            save_video(frames_video, image_name, duration=durations_video)

        # Close buffers
        for frame in main_frames + end_frames:
            frame.close()


def plot_patch_sample(dataset: Dataset, number_per_class: int, show_offset: bool = True) -> None:
    """
    Plot randomly sampled patches grouped by class.

    :param dataset: The patches dataset to sample from.
    :param number_per_class: The number of sample per class.
    :param show_offset: If True draw the offset rectangle (ignored if both offset x and y are 0)
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
        axs[i, 0].set_title(f'{number_per_class} examples of "{QDSDLines.classes[i].capitalize()}"', loc='left',
                            fontsize='xx-large', fontweight='bold')
        for j, class_data in enumerate(cl):
            axs[i, j].imshow(class_data.reshape(settings.patch_size_x, settings.patch_size_y),
                             interpolation='none', cmap=PIXELS_CMAP, origin='lower')

            if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
                # Create a rectangle patch that represents offset
                rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                         settings.patch_size_x - 2 * settings.label_offset_x,
                                         settings.patch_size_y - 2 * settings.label_offset_y,
                                         linewidth=2, edgecolor='fuchsia', facecolor='none')

                # Add the offset rectangle to the axes
                axs[i, j].add_patch(rect)

            axs[i, j].axis('off')

        if settings.test_noise:
            plt.suptitle(f'Gaussian noise: {settings.test_noise:.0%}')

    save_plot('patch_sample')


def plot_samples(samples: List, title: str, file_name: str, confidences: List[Union[float, Tuple[float]]] = None,
                 show_offset: bool = True) -> None:
    """
    Plot a group of patches.

    :param samples: The list of patches to plot.
    :param title: The title of the plot.
    :param file_name: The file name of the plot if saved.
    :param confidences: The list of confidence score for the prediction of each sample. If it's a tuple then we assume
     it's (mean, std, entropy).
    :param show_offset: If True, draw the offset rectangle (ignored if both offset x and y are 0)
    """
    plot_length = ceil(sqrt(len(samples)))

    if plot_length <= 1:
        return  # FIXME: deal with 1 or 0 sample

    # Create subplots
    fig, axs = plt.subplots(nrows=plot_length, ncols=plot_length, figsize=(plot_length * 2, plot_length * 2 + 1))

    for i, s in enumerate(samples):
        ax = axs[i // plot_length, i % plot_length]
        ax.imshow(s.reshape(settings.patch_size_x, settings.patch_size_y), interpolation='none', cmap=PIXELS_CMAP,
                  origin='lower')

        if confidences:
            # If it's a tuple we assume it is: mean, std, entropy
            if isinstance(confidences[i], tuple):
                mean, std, entropy = confidences[i]
                ax.title.set_text(f'{mean:.2} (std {std:.2})\nEntropy:{entropy:.2}')
            # If it's not a tuple, we assume it is a float representing the confidence score
            else:
                ax.title.set_text(f'{confidences[i]:.2%}')

        if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
            # Create a rectangle patch that represent offset
            rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                     settings.patch_size_x - 2 * settings.label_offset_x,
                                     settings.patch_size_y - 2 * settings.label_offset_y,
                                     linewidth=2, edgecolor='fuchsia', facecolor='none')

            # Add the offset rectangle to the axes
            ax.add_patch(rect)

        ax.axis('off')

    fig.suptitle(title)

    save_plot(f'sample_{file_name}')


def plot_data_space_distribution(datasets: Sequence[Dataset], title: str, file_name: str) -> None:
    """
    Plot the pixel values distribution for each dataset.

    :param datasets: The list of dataset to plot.
    :param title: The title of the plot.
    :param file_name: The file name of the plot if saved.
    """
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize=(len(datasets) * 6, 8))

    for i, dataset in enumerate(datasets):
        # Plot the distribution
        sns.histplot(dataset._patches.flatten(), ax=axes[i], kde=True, stat='count', bins=200)
        axes[i].set_title(dataset.role.capitalize())

    fig.suptitle(title)
    save_plot(f'data_distribution_{file_name}')
