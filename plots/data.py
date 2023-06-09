import io
from copy import copy
from functools import partial
from itertools import chain
from math import ceil, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
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
                focus_area_ax = fig.add_subplot(spec[:2, x_end_col_1:])
            if show_text_ax:
                text_ax = fig.add_subplot(spec[2 if show_focus_area_ax else 0:y_bottom, x_end_col_1:])

        # 1 column layout
        else:
            diagram_ax = fig.add_subplot(spec[:y_bottom, :])

    return fig, diagram_ax, focus_area_ax, text_ax, legend_ax


def _plot_diagram_ax(ax, x_i: Sequence[float], y_i: Sequence[float], pixels: Optional[Tensor], title: Optional[str],
                     scan_history: List["StepHistoryEntry"], fog_of_war: bool, scale_bar: bool, vmin: float,
                     vmax: float, axes_matching: List[float]) -> None:
    # Subplot title
    if title:
        ax.set_title(title)

    # If no pixels provided, plot a blank image to allow other information on the same format
    if pixels is None:
        cmap = LinearSegmentedColormap.from_list('', ['white', 'white'])
        ax.imshow(np.zeros((len(x_i), len(y_i))), cmap=cmap, extent=axes_matching)
        return

    # If the fog of war is enabled, mask the pixels that have not been scanned yet according to the scan history
    if fog_of_war and scan_history is not None:
        mask = np.full_like(pixels, True)
        for scan in scan_history:
            x, y = scan.coordinates
            mask[y + settings.patch_size_y: y, x:x + settings.patch_size_x] = False
        pixels = np.ma.masked_array(pixels, mask)

    # Plot the pixels
    im = ax.imshow(pixels, interpolation='none', cmap=PIXELS_CMAP, extent=axes_matching, vmin=vmin, vmax=vmax,
                   origin='lower')

    # Add the scale bar
    if scale_bar:
        if settings.research_group in MEASURE_UNIT:
            measuring = MEASURE_UNIT[settings.research_group]
        else:
            measuring = 'I'
        # Add the scale bar that way because we already are inside a subplot
        plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=f'{measuring} (A)')


def _plot_focus_area_ax(focus_ax, diagram_ax, pixels, title, focus_area, vmin, vmax, axes_matching: List[float]):
    # Subplot title
    if title:
        focus_ax.set_title(title)

    # Get coordinates and add half-pixel to match with the border of pixel (instead of center).
    half_p = settings.pixel_size / 2
    x_start, x_end, y_start, y_end = focus_area
    x_start -= half_p
    y_start -= half_p
    x_end += half_p
    y_end += half_p

    focus_ax.imshow(pixels, interpolation='none', cmap=PIXELS_CMAP, extent=axes_matching, vmin=vmin, vmax=vmax,
                    origin='lower')
    focus_ax.set_xlim(x_start, x_end)
    focus_ax.set_ylim(y_start, y_end)

    # Show the location of the current focus area in the main diagram using a black rectangle
    loc = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=1, zorder=99999,
                            edgecolor='black', label=title, facecolor='none', alpha=0.8)
    diagram_ax.add_patch(loc)


def _plot_labels(ax, charge_regions: List[Tuple["ChargeRegime", Polygon]], transition_lines: List[LineString]):
    labels_handlers = []

    if charge_regions:
        for i, (regime, polygon) in enumerate(charge_regions):
            polygon_x, polygon_y = polygon.exterior.coords.xy
            ax.fill(polygon_x, polygon_y, facecolor=(0, 0, 0.5, 0.3), edgecolor=(0, 0, 0.5, 0.8), snap=True)
            label_x, label_y = list(polygon.centroid.coords)[0]
            params = dict(x=label_x, y=label_y, ha="center", va="center", color='b', weight='bold',
                          bbox=dict(boxstyle='round', pad=0.2, facecolor='w', alpha=0.5, edgecolor='w'))
            ax.text(**params, s=str(regime))

            if i == 0:
                # Create custom label for charge regime
                labels_handlers.append((Text(**params, text='N'), 'Charge regime'))

    if transition_lines:
        for i, line in enumerate(transition_lines):
            line_x, line_y = line.coords.xy
            ax.plot(line_x, line_y, color='lime', label='Line annotation')

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
            # Filled patch if white background
            square_params['facecolor'] = color
        else:
            # Empty patch if to see the diagram
            square_params['edgecolor'] = color

        rec = patches.Rectangle(linewidth=1, **square_params)
        diagram_ax.add_patch(rec)

        # Show the rectangle label offset for the last scan
        if focus_ax and i == 0:
            square_params['facecolor'] = None
            square_params['edgecolor'] = color
            focus_ax.add_patch(patches.Rectangle(linewidth=2, **square_params))


def _plot_text_ax(text_ax, text: str):
    text_ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=12,
                 fontfamily='monospace', transform=text_ax.transAxes)
    text_ax.axis('off')


def _plot_final_coord(diagram_ax, focus_area_ax, final_volt_coord: Tuple[float, float]):
    last_x, last_y = final_volt_coord
    for ax, cross_size, lw in ((diagram_ax, 200, 2), (focus_area_ax, 600, 4)):
        if ax:
            # Make white borders using a slightly bigger marker under it
            ax.scatter(x=last_x, y=last_y, color='w', marker='x', s=cross_size * 1.1, linewidths=lw * 1.5, zorder=9998)
            ax.scatter(x=last_x, y=last_y, color='fuchsia', marker='x', s=cross_size, label='End',
                       linewidths=lw, zorder=9999)


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


def plot_diagram(x_i: Sequence[float],
                 y_i: Sequence[float],
                 pixels: Optional[Tensor],
                 title: Optional[str] = None,
                 fog_of_war: bool = False,
                 charge_regions: List[Tuple["ChargeRegime", Polygon]] = None,
                 transition_lines: List[LineString] = None,
                 scan_history: List["StepHistoryEntry"] = None,
                 scan_history_mode: Literal['classes', 'error', 'uncertainty'] | None = None,
                 scan_history_alpha: Literal['uncertainty'] | int | None = None,
                 focus_area: Optional[bool | Tuple[float, float, float, float]] = None,
                 focus_area_title: str = 'Focus area',
                 final_volt_coord: Tuple[float, float] = None,
                 text: Optional[str | bool] = None,
                 scale_bar: bool = False,
                 legend: Optional[bool] = True,
                 vmin: float = None,
                 vmax: float = None,
                 file_name: str = None,
                 allow_overwrite: bool = False,
                 save_in_buffer: bool = False
                 ) -> Optional[Path | io.BytesIO]:
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
    _plot_diagram_ax(diagram_ax, x_i, y_i, pixels, title, scan_history, fog_of_war, scale_bar, vmin, vmax,
                     axes_matching)

    # Show labels if provided
    if charge_regions or transition_lines:
        custom_legend.extend(_plot_labels(diagram_ax, charge_regions, transition_lines))

    # Build the focus area subplot
    if show_focus_area_ax:
        if focus_area is True:
            # Get last scan coordinates and convert them to voltage.
            x_start, x_end, y_start, y_end = scan_history[-1].get_area_coord()
            focus_area = (x_i[x_start], x_i[x_end - 1], x_i[y_start], x_i[y_end - 1])
        _plot_focus_area_ax(focus_area_ax, diagram_ax, pixels, focus_area_title, focus_area, vmin, vmax, axes_matching)

    # Plot the scan history visualization
    if scan_history_mode and scan_history and len(scan_history) > 0:
        _plot_scan_history(diagram_ax, show_focus_area_ax, x_i, y_i, scan_history, scan_history_mode,
                           scan_history_alpha, pixels is None)

    if final_volt_coord is not None:
        _plot_final_coord(diagram_ax, focus_area_ax, final_volt_coord)

    if show_text_ax:
        if not isinstance(text, str):
            # If the text is not a string, it means that we want to generate the text from the scan history
            text = StepHistoryEntry.get_text_description(scan_history)
        _plot_text_ax(text_ax, text)

    if show_legend_ax:
        _plot_legend_ax(legend_ax, diagram_ax, custom_legend)

    # Save the plot
    return save_plot(file_name, allow_overwrite=allow_overwrite, save_in_buffer=save_in_buffer, figure=fig)


def plot_diagram_old(x_i, y_i,
                     pixels: Optional,
                     file_name: str = None,
                     title: str = None,
                     charge_regions: Iterable[Tuple["ChargeRegime", Polygon]] = None,
                     transition_lines: Iterable[LineString] = None,
                     show_offset: bool = False,
                     scan_history: List["StepHistoryEntry"] = None,
                     focus_area: Optional[Union[bool, Tuple[float, float, float, float]]] = None,
                     focus_area_title: str = 'Focus area',
                     scan_errors: bool = False,
                     confidence_thresholds: List[float] = None,
                     fog_of_war: bool = False,
                     fading_history: int = 0,
                     history_uncertainty: bool = False,
                     scale_bar: bool = False,
                     final_coord: Tuple[int, int] = None,
                     save_in_buffer: bool = False,
                     description: bool = False,
                     show_title: Optional[bool] = None,
                     show_crosses: bool = True,
                     vmin: float = None,
                     vmax: float = None,
                     diagram_boundaries: Optional[Tuple[float, float, float, float]] = None,
                     allow_overwrite: bool = False) -> Optional[Union[Path, io.BytesIO]]:
    """
    Plot the interpolated image. This function is a multi-tool nightmare.

    :param x_i: The x coordinates of the pixels (post interpolation).
    :param y_i: The y coordinates of the pixels (post interpolation).
    :param pixels: The list of pixels to plot.
    :param file_name: The name of the output file.
    :param title: The title of the diagram axis. If None, use the diagram name.
    :param charge_regions: The charge region annotations to draw on top of the image.
    :param transition_lines: The transition line annotation to draw on top of the image.
    :param show_offset: If True, draw the label offset of the last patch in the focus area
        (ignored if both offset x and y are 0). The color change in fonction of the model prediction.
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass).
    :param focus_area: Show a subregion of the diagram in a separate subplot. If is a tuple of 4 voltages as:
        (x_start, x_end, y_start, y_end), show the focus area defined by the coordinates. If is True and scan_history
        has at least one scan, show the focus area of the last step. If is False, or if scan_history is None or empty,
        create the subplot but plot nothing. If is None, do not create the subplot.
    :param focus_area_title: The title of the focus subplot.
    :param scan_errors: If True, and scan_history defined, plot the step error on the diagram. If False plot the class
        inference instead. Soft errors are shown only if uncertainty is disabled.
    :param confidence_thresholds: The model confidence threshold values for each class. Only necessary if scan_errors
     and not history_uncertainty enabled (yes, this is very specific).
    :param fog_of_war: If True, and scan_history defined, hide the section of the diagram that was never scanned.
    :param fading_history: The number of scan inferences to plot, the latest first. The number set will be plotted with
        solid color and the same number will fad progressively. Not compatible with history_uncertainty.
    :param history_uncertainty: If True and scan_history provided, plot steps with full squares and alpha representing
        the uncertainty.
    :param scale_bar: If True, and pixels provided, plot the pixel color scale at the right of the diagram. If the data
        are normalized, this scale unit doesn't make sense.
    :param final_coord: The final tuning coordinates.
    :param save_in_buffer: If True, save the image in memory. Do not plot or save it on the disk.
    :param description: If True and scan_history has at least one entry, add statistics information about the scan steps
        in the plot. If a string, directly add the given string as description.
    :param show_title: If True, plot figure title. If omitted, show title only if not latex format.
    :param show_crosses: If True, plot the crosses representing the start and the end of the tuning if possible.
    :param vmin: Minimal pixel value for color scaling. Set to keep consistant color between plots. If None, the scaling
        is computed by matplotlib based on pixel currently visible.
    :param vmax: Maximal pixel value for color scaling. Set to keep consistant color between plots. If None, the
        scaling is computed by matplotlib based on pixel currently visible.
    :param allow_overwrite: If True, allow to overwrite existing plot.
    :return: The path where the plot is saved, or None if not saved. If save_in_buffer is True, return image bytes
        instead of the path.
    """
    nb_scan = len(scan_history) if scan_history is not None else 0
    legend = False
    # By default do not plot title for latex format.
    show_title = not settings.image_latex_format if show_title is None else show_title
    show_description = description is not False and description is not None

    # If focus area is True, and scan history is not empty, set focus area to the last step area.
    if focus_area is True:
        if nb_scan > 0:
            # Get last scan coordinates, and convert them to voltage.
            x_start, x_end, y_start, y_end = scan_history[-1].get_area_coord()
            focus_area = (x_i[x_start], x_i[x_end - 1], x_i[y_start], x_i[y_end - 1])
        else:
            focus_area = False

    with sns.axes_style("ticks"):  # Temporary change the axe style (avoid white ticks)
        diagram_ax, focus_ax, description_ax, legend_ax = None, None, None, None
        if focus_area is not None:
            fig, ax = plt.subplot_mosaic("""
            DDF
            DDT
            """, figsize=(16, 9))
            diagram_ax = ax['D']
            focus_ax = ax['F']
            description_ax = ax['T']
        else:
            fig, ax = plt.subplot_mosaic("""
                        DDT
                        DDT
                        """, figsize=(16, 9))
            diagram_ax = ax['D']
            description_ax = ax['T']

    half_p = settings.pixel_size / 2
    # Set the plot coordinates to fit the image with the axes, with the center of pixels aligned with the coordinates.
    axes_matching = [x_i[0] - half_p, x_i[-1] + half_p, y_i[0] - half_p, y_i[-1] + half_p]
    if pixels is None:
        # If no pixels provided, plot a blank image to allow other information on the same format
        diagram_ax.imshow(np.zeros((len(x_i), len(y_i))),
                          cmap=LinearSegmentedColormap.from_list('', ['white', 'white']), extent=axes_matching)
    else:
        if fog_of_war and scan_history is not None and nb_scan > 0:
            # Mask not-scanned area according to the current scan history list
            mask = np.full_like(pixels, True)
            for scan in scan_history:
                x, y = scan.coordinates
                y = len(y_i) - y  # Origine to bottom left
                mask[y - settings.patch_size_y: y, x:x + settings.patch_size_x] = False
            pixels = np.ma.masked_array(pixels, mask)

        cmap = matplotlib.cm.copper
        cmap.set_bad(color=NOT_SCANNED_COLOR)
        diagram_ax.imshow(pixels, interpolation='none', cmap=cmap, extent=axes_matching, vmin=vmin, vmax=vmax)
        if scale_bar:
            if settings.research_group == 'michel_pioro_ladriere' or \
                    settings.research_group == 'eva_dupont_ferrier':
                measuring = r'$\mathregular{I_{SET}}$'
            elif settings.research_group == 'louis_gaudreau':
                measuring = r'$\mathregular{I_{QPC}}$'
            else:
                measuring = 'I'
            # TODO fix the colorbar, or add the colorbar directly with the legend
            # diagram_ax.colorbar(shrink=0.85, label=f'{measuring} (A)')

        if focus_ax and isinstance(focus_area, tuple):
            # Get coordinates and add half-pixel to match with the border of pixel (instead of center).
            x_start, x_end, y_start, y_end = focus_area
            x_start -= half_p
            y_start -= half_p
            x_end += half_p
            y_end += half_p
            focus_ax.imshow(pixels, interpolation='none', cmap=cmap, extent=axes_matching, vmin=vmin, vmax=vmax)
            # Show the location of the focus area in the diagram using a rectangle
            loc = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start, linewidth=1, zorder=9999,
                                    edgecolor='black', label=focus_area_title, facecolor='none', alpha=0.8)
            diagram_ax.add_patch(loc)
            focus_ax.set_xlim(x_start, x_end)
            focus_ax.set_ylim(y_start, y_end)
            if show_title and focus_area_title:
                focus_ax.set_title(focus_area_title)

    charge_text = None  # Keep on text field for legend
    if charge_regions is not None:
        for regime, polygon in charge_regions:
            polygon_x, polygon_y = polygon.exterior.coords.xy
            diagram_ax.fill(polygon_x, polygon_y, facecolor=(0, 0, 0.5, 0.3), edgecolor=(0, 0, 0.5, 0.8), snap=True)
            label_x, label_y = list(polygon.centroid.coords)[0]
            charge_text = diagram_ax.text(label_x, label_y, str(regime), ha="center", va="center", color='b',
                                          weight='bold',
                                          bbox=dict(boxstyle='round', pad=0.2, facecolor='w', alpha=0.5, edgecolor='w'))

    if transition_lines is not None:
        for i, line in enumerate(transition_lines):
            line_x, line_y = line.coords.xy
            diagram_ax.plot(line_x, line_y, color='lime', label='Line annotation' if i == 0 else None)
            legend = True

    if nb_scan > 0:
        from datasets.qdsd import QDSDLines  # Import here to avoid circular import
        first_patch_label = set()

        patch_size_x_v = (settings.patch_size_x - settings.label_offset_x * 2) * settings.pixel_size
        patch_size_y_v = (settings.patch_size_y - settings.label_offset_y * 2) * settings.pixel_size

        for i, scan_entry in enumerate(reversed(scan_history)):
            line_detected = scan_entry.model_classification
            x, y = scan_entry.coordinates
            rec_param = {
                'xy': (x_i[x + settings.label_offset_x] - half_p, y_i[y + settings.label_offset_y] - half_p),
                'width': patch_size_x_v,
                'height': patch_size_y_v,
                'edgecolor': None,
                'label': '',
                'facecolor': None,
                'alpha': 1,
                'zorder': nb_scan - i
            }

            if scan_errors:
                # Patch color depending on the classification success
                if not history_uncertainty and scan_entry.is_under_confidence_threshold(confidence_thresholds):
                    # If the uncertainty is not shown with alpha, we show it by a gray patch
                    color = UNKNOWN_COLOR
                    rec_param['label'] = 'Unknown'
                elif scan_entry.is_classification_correct():
                    color = GOOD_COLOR
                    rec_param['label'] = 'Good'
                elif not history_uncertainty and scan_entry.is_classification_almost_correct():
                    # Soft error is not compatible with uncertainty
                    color = SOFT_ERROR_COLOR
                    rec_param['label'] = 'Soft Error'
                else:
                    color = ERROR_COLOR
                    rec_param['label'] = 'Error'
            else:
                # Patch color depending on the inferred class
                color = LINE_COLOR if line_detected else NO_LINE_COLOR
                rec_param['label'] = f'Infer {QDSDLines.classes[line_detected]}'

            # Add label only if it is the first time we plot a patch with this label
            if rec_param['label'] in first_patch_label:
                rec_param['label'] = None
            else:
                first_patch_label.add(rec_param['label'])

            if history_uncertainty or fading_history == 0 or i < fading_history * 2:  # Condition to plot patches
                if history_uncertainty:
                    # Transparency based on the confidence
                    rec_param['alpha'] = scan_entry.model_confidence
                    rec_param['label'] = None  # No label since we have the scale bar
                elif fading_history == 0 or i < fading_history * 2:
                    if fading_history != 0:
                        # History fading for fancy animation
                        rec_param['alpha'] = 1 if i < fading_history else (2 * fading_history - i) / (
                                fading_history + 1)
                    legend = True

                if pixels is None:
                    # Full patch if white background
                    rec_param['facecolor'] = color
                    rec_param['edgecolor'] = 'none'
                else:
                    # Empty patch if diagram background
                    rec_param['facecolor'] = 'none'
                    rec_param['edgecolor'] = color

                rec = patches.Rectangle(linewidth=1, **rec_param)
                diagram_ax.add_patch(rec)

                # Show the rectangle label offset in the focus area if necessary
                if focus_ax and i == 0 and show_offset and settings.label_offset_x > 0 and settings.label_offset_y > 0:
                    focus_ax.add_patch(patches.Rectangle(linewidth=2, **rec_param))

        # Marker for first point
        if show_crosses and (fading_history == 0 or nb_scan < fading_history * 2):
            first_x, first_y = scan_history[0].coordinates
            first_x = x_i[first_x + settings.patch_size_x // 2] - half_p
            first_y = y_i[first_y + settings.patch_size_y // 2] - half_p
            if fading_history == 0:
                alpha = 1
            else:
                # Fading after the first scans if fading_history is enabled
                i = nb_scan - 2
                alpha = 1 if i < fading_history else (2 * fading_history - i) / (fading_history + 1)
            diagram_ax.scatter(x=first_x, y=first_y, color='skyblue', marker='X', s=200, label='Start', alpha=alpha,
                               zorder=nb_scan + 1)
            legend = True

        if history_uncertainty:
            # Set up the colorbar
            if scan_errors:
                cmap = LinearSegmentedColormap.from_list('', [GOOD_COLOR, 'white', ERROR_COLOR])
            else:
                cmap = LinearSegmentedColormap.from_list('', [LINE_COLOR, 'white', ERROR_COLOR])
            norm = Normalize(vmin=-1, vmax=1)
            cbar = diagram_ax.colorbar(ScalarMappable(cmap=cmap, norm=norm), shrink=0.8, aspect=15)
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

            # Ad hoc uncertainty
            else:
                uncertainty_label = 'Model output'
                min_uncertainty_line = 1
                min_uncertainty_no_line = 0
                min_uncertainty_correct = f'{min_uncertainty_line} or {min_uncertainty_no_line}'
                max_uncertainty = 0.5

            if scan_errors:
                cbar.set_ticklabels(
                    [f'Correct class\n{uncertainty_label}: {min_uncertainty_correct}\n(Low uncertainty)',
                     f'{uncertainty_label}: {max_uncertainty}\n(High uncertainty)',
                     f'Error class\n{uncertainty_label}: {min_uncertainty_correct}\n(Low uncertainty)'])
            else:
                cbar.set_ticklabels([f'Line\n{uncertainty_label}: {min_uncertainty_line}\n(Low uncertainty)',
                                     f'{uncertainty_label}: {max_uncertainty}\n(High uncertainty)',
                                     f'No Line\n{uncertainty_label}: {min_uncertainty_no_line}\n(Low uncertainty)'])

    # Marker for tuning final guess
    if show_crosses and final_coord is not None:
        last_x, last_y = final_coord
        # Get marker position (and avoid going out)
        last_x_i = max(min(last_x, len(x_i) - 1), 0)
        last_y_i = max(min(last_y, len(y_i) - 1), 0)
        last_x = x_i[last_x_i] - half_p
        last_y = y_i[last_y_i] - half_p
        for ax, cross_size, lw in ((diagram_ax, 200, 2), (focus_ax, 600, 4)):
            if ax:
                # Make white borders using a slightly bigger marker under it
                ax.scatter(x=last_x, y=last_y, color='w', marker='x', s=cross_size * 1.1,
                           linewidths=lw * 1.5, zorder=nb_scan + 2)
                ax.scatter(x=last_x, y=last_y, color='fuchsia', marker='x', s=cross_size, label='End',
                           linewidths=lw, zorder=nb_scan + 3)
        legend = True

    if description_ax is not None:
        if isinstance(description, str):
            text = description
        else:
            # If text_stat is not s string, it means that we want to generate the text from the scan history
            text = StepHistoryEntry.get_text_description(scan_history)

        description_ax.text(0.05, 0.95, text, horizontalalignment='left', verticalalignment='top', fontsize=12,
                            fontfamily='monospace', transform=description_ax.transAxes)
        description_ax.axis('off')

    if show_title:
        diagram_ax.set_title(f'{image_name} - pixel size: {round(pixel_size, 10) * 1_000}mV')

    # If defined, set boundaries for the diagram axes
    if diagram_boundaries:
        x_start_v, x_end_v, y_start_v, y_end_v = diagram_boundaries
        diagram_ax.set_xlim(x_start_v, x_end_v)
        diagram_ax.set_ylim(y_start_v, y_end_v)

    diagram_ax.set_xlabel('G1 (V)')
    diagram_ax.tick_params(axis='x', labelrotation=30)
    diagram_ax.set_ylabel('G2 (V)')

    if legend:
        handles, labels = diagram_ax.get_legend_handles_labels()
        handler_map = None
        if charge_text is not None:
            # Create custom legend for charge regime text
            charge_text = copy(charge_text)
            charge_text.set(text='N')
            handler_map = {type(charge_text): TextHandler()}
            handles.append(charge_text)
            labels.append('Charge regime')

        diagram_ax.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.35), handles=handles, labels=labels,
                          handler_map=handler_map)

    return save_plot(file_name, allow_overwrite=allow_overwrite, save_in_buffer=save_in_buffer, figure=fig)


def plot_diagram_step_animation(d: "Diagram", title: str, image_name: str, scan_history: List["StepHistoryEntry"],
                                final_coord: Tuple[int, int], show_crosses: bool = True) -> None:
    """
    Plot an animation of the tuning procedure.

    :param d: The diagram to plot.
    :param image_name: The name of the image, used for plot title and file name
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass)
    :param final_coord: The final tuning coordinates
    :param show_crosses: If True, show the starting and ending crosses on the diagram during the step by step animation.
        They are shown anyway during the final steps.
    """

    if settings.is_named_run() and (settings.save_gif or settings.save_video):
        values, x_axes, y_axes = d.get_values()
        from datasets.diagram_online import DiagramOnline
        is_online = isinstance(d, DiagramOnline)
        diagram_boundaries = d.get_cropped_boundaries() if is_online else None
        # Compute min / max here because numpy doesn't like to do this on multi thread (ignore NaN values)
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        # Final area where we assume the target regime is
        margin = settings.patch_size_x * 3
        final_area_x = d.coord_to_voltage(final_coord[0] - margin, final_coord[0] + margin, clip_in_diagram=True)
        final_area_y = d.coord_to_voltage(final_coord[1] - margin, final_coord[1] + margin, clip_in_diagram=True)
        final_area = final_area_x + final_area_y
        # Animation speed => Time for an image (ms)
        base_fps = 200
        # Ratio of image to skip for the animation frames (1 means nothing skipped, 4 means 1 keep for 3 skip)
        rate_gif = 4
        rate_video = 1
        # Define frames to compute
        gif_frames_ids = list(range(1, len(scan_history), rate_gif)) if settings.save_gif else []
        video_frames_ids = list(range(1, len(scan_history), rate_video)) if settings.save_video else []
        frame_ids = list(dict.fromkeys(gif_frames_ids + video_frames_ids))  # Remove duplicate and keep order

        with Pool(get_nb_loader_workers()) as pool:
            # Generate images in parallel for speed. Use partial to set constants arguments.
            # Main animation frames
            async_result_main = pool.map_async(
                partial(plot_diagram, x_axes, y_axes, values, None, title,
                        None, None, True, focus_area=True, focus_area_title='Last patch', save_in_buffer=True,
                        description=True, show_title=True, fog_of_war=True, fading_history=8,
                        diagram_boundaries=diagram_boundaries, vmin=vmin, vmax=vmax, show_crosses=show_crosses),
                (scan_history[0:i] for i in frame_ids)
            )

            # Final frames
            async_result_end = [
                # Show diagram with all inference and fog of war
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': x_axes, 'y_i': y_axes, 'pixels': values, 'title': title,
                                       'scan_history': scan_history, 'focus_area': final_area,
                                       'focus_area_title': 'End area', 'save_in_buffer': True, 'description': True,
                                       'show_title': True, 'fog_of_war': True, 'diagram_boundaries': diagram_boundaries,
                                       'vmin': vmin, 'vmax': vmax}),
                # Show diagram with tuning final coordinate and fog of war
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': x_axes, 'y_i': y_axes, 'pixels': values, 'title': title,
                                       'scan_history': scan_history, 'focus_area': final_area,
                                       'focus_area_title': 'End area', 'final_coord': final_coord,
                                       'save_in_buffer': True, 'description': True, 'show_title': True,
                                       'fog_of_war': True, 'diagram_boundaries': diagram_boundaries, 'vmin': vmin,
                                       'vmax': vmax})
            ]

            # If online, we don't have label to show
            if is_online:
                end_durations = [base_fps * 15, base_fps * 40]
            else:
                end_durations = [base_fps * 15, base_fps * 15, base_fps * 15, base_fps * 30, base_fps * 40]
                async_result_end += [
                    # Show full diagram with tuning final coordinate
                    pool.apply_async(plot_diagram,
                                     kwds={'x_i': x_axes, 'y_i': y_axes, 'pixels': values, 'title': title,
                                           'scan_history': scan_history, 'focus_area': final_area,
                                           'focus_area_title': 'End area', 'final_coord': final_coord,
                                           'save_in_buffer': True, 'description': True, 'show_title': True,
                                           'diagram_boundaries': diagram_boundaries, 'vmin': vmin, 'vmax': vmax}),
                    # Show full diagram with tuning final coordinate + line labels
                    pool.apply_async(plot_diagram,
                                     kwds={'x_i': x_axes, 'y_i': y_axes, 'pixels': values, 'title': title,
                                           'scan_history': scan_history, 'focus_area': final_area,
                                           'focus_area_title': 'End area', 'final_coord': final_coord,
                                           'save_in_buffer': True, 'description': True, 'show_title': True,
                                           'transition_lines': d.transition_lines,
                                           'diagram_boundaries': diagram_boundaries, 'vmin': vmin, 'vmax': vmax}),
                    # Show full diagram with tuning final coordinate + line & regime labels
                    pool.apply_async(plot_diagram,
                                     kwds={'x_i': x_axes, 'y_i': y_axes, 'pixels': values, 'title': title,
                                           'scan_history': scan_history, 'focus_area': final_area,
                                           'focus_area_title': 'End area', 'final_coord': final_coord,
                                           'save_in_buffer': True, 'description': True, 'show_title': True,
                                           'transition_lines': d.transition_lines, 'charge_regions': d.charge_areas,
                                           'diagram_boundaries': diagram_boundaries, 'vmin': vmin, 'vmax': vmax}),
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
                             interpolation='nearest', cmap='copper')

            if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
                # Create a rectangle patch that represent offset
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
    :param show_offset: If True draw the offset rectangle (ignored if both offset x and y are 0)
    """
    plot_length = ceil(sqrt(len(samples)))

    if plot_length <= 1:
        return  # FIXME: deal with 1 or 0 sample

    # Create subplots
    fig, axs = plt.subplots(nrows=plot_length, ncols=plot_length, figsize=(plot_length * 2, plot_length * 2 + 1))

    for i, s in enumerate(samples):
        ax = axs[i // plot_length, i % plot_length]
        ax.imshow(s.reshape(settings.patch_size_x, settings.patch_size_y), interpolation='nearest', cmap='copper')

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
