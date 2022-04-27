import io
from copy import copy
from functools import partial
from math import ceil, sqrt
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.legend_handler import HandlerBase
from shapely.geometry import LineString, Polygon
from torch.utils.data import DataLoader, Dataset

from utils.misc import get_nb_loader_workers
from utils.output import save_gif, save_plot, save_video
from utils.settings import settings

LINE_COLOR = 'blue'
NO_LINE_COLOR = 'red'
GOOD_COLOR = 'green'
ERROR_COLOR = 'red'


def plot_diagram(x_i, y_i,
                 pixels: Optional,
                 image_name: str,
                 interpolation_method: str,
                 pixel_size: float,
                 charge_regions: Iterable[Tuple["ChargeRegime", Polygon]] = None,
                 transition_lines: Iterable[LineString] = None,
                 focus_area: Optional[Tuple] = None,
                 show_offset: bool = True,
                 scan_history: List["StepHistoryEntry"] = None,
                 scan_errors: bool = False,
                 fog_of_war: bool = False,
                 history_uncertainty: bool = False,
                 scale_bar: bool = False,
                 final_coord: Tuple[int, int] = None,
                 save_in_buffer: bool = False,
                 text_stats: bool = False,
                 show_title: Optional[bool] = None,
                 show_crosses: bool = True) -> Optional[Union[Path, io.BytesIO]]:
    """
    Plot the interpolated image.

    :param x_i: The x coordinates of the pixels (post interpolation).
    :param y_i: The y coordinates of the pixels (post interpolation).
    :param pixels: The list of pixels to plot.
    :param image_name: The name of the image, used for plot title and file name.
    :param interpolation_method: The pixels' interpolation method, used for plot title.
    :param pixel_size: The size of pixels, in voltage, used for plot title.
    :param charge_regions: The charge region annotations to draw on top of the image.
    :param transition_lines: The transition line annotation to draw on top of the image.
    :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
    :param show_offset: If True draw the offset rectangle (ignored if both offset x and y are 0).
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass).
    :param scan_errors: If True and scan_history defined, plot the step error on the diagram. If False plot the class
     inference instead.
    :param fog_of_war: If True and scan_history defined, hide the section of the diagram that was never scanned.
    :param history_uncertainty: If True and scan_history provided, plot steps with full squares and alpha representing
     the uncertainty.
    :param scale_bar: If True and pixels provided, plot the pixel color scale at the right of the diagram. If the data
     are normalized this scale unit doesn't make sense.
    :param final_coord: The final tuning coordinates.
    :param save_in_buffer: If True, save the image in memory. Do not plot or save it on the disk.
    :param text_stats: If True, add statistics information in the plot.
    :param show_title: If True plot figure title. If omitted, show title only if not latex format.
    :param show_crosses: If True plot the crosses representing the start and the end of the tuning if possible.
    :return: The path where the plot is saved, or None if not saved. If save_in_buffer is True, return image bytes
     instead of the path.
    """

    legend = False
    # By default do not plot title for latex format.
    show_title = not settings.image_latex_format if show_title is None else show_title

    with sns.axes_style("ticks"):  # Temporary change the axe style (avoid white ticks)
        boundaries = [np.min(x_i), np.max(x_i), np.min(y_i), np.max(y_i)]
        if pixels is None:
            # If no pixels provided, plot a blank image to allow other information on the same format
            plt.imshow(np.zeros((len(x_i), len(y_i))), cmap=LinearSegmentedColormap.from_list('', ['white', 'white']),
                       extent=boundaries)
        else:
            if fog_of_war and scan_history is not None and len(scan_history) > 0:
                # Mask area not scanned
                mask = np.full_like(pixels, True)
                for scan in scan_history:
                    x, y = scan.coordinates
                    y = len(y_i) - y  # Origine to bottom left
                    mask[y - settings.patch_size_y: y, x:x + settings.patch_size_x] = False
                pixels = np.ma.masked_array(pixels, mask)

            cmap = matplotlib.cm.copper
            cmap.set_bad(color='lightgrey')
            plt.imshow(pixels, interpolation='nearest', cmap=cmap, extent=boundaries)
            if scale_bar:
                plt.colorbar(shrink=0.85, label='Current (A)')

    charge_text = None  # Keep on text field for legend
    if charge_regions is not None:
        for regime, polygon in charge_regions:
            polygon_x, polygon_y = polygon.exterior.coords.xy
            plt.fill(polygon_x, polygon_y, facecolor=(0, 0, 0.5, 0.3), edgecolor=(0, 0, 0.5, 0.8), snap=True)
            label_x, label_y = list(polygon.centroid.coords)[0]
            charge_text = plt.text(label_x, label_y, str(regime), ha="center", va="center", color='b', weight='bold',
                                   bbox=dict(boxstyle='round', pad=0.2, facecolor='w', alpha=0.5, edgecolor='w'))

    if transition_lines is not None:
        for i, line in enumerate(transition_lines):
            line_x, line_y = line.coords.xy
            plt.plot(line_x, line_y, color='lime', label='Line label' if i == 0 else None)
            legend = True

    if scan_history is not None and len(scan_history) > 0:
        from datasets.qdsd import QDSDLines  # Import here to avoid circular import
        first_patch_label = set()

        patch_size_x_v = (settings.patch_size_x - settings.label_offset_x * 2) * pixel_size
        patch_size_y_v = (settings.patch_size_y - settings.label_offset_y * 2) * pixel_size

        for scan_entry in scan_history:
            line_detected = scan_entry.model_classification
            correct_class = scan_entry.model_classification == scan_entry.ground_truth
            x, y = scan_entry.coordinates

            if scan_errors:
                # Green for correct class / Red for bad class
                color = GOOD_COLOR if correct_class else ERROR_COLOR
            else:
                # Blue for Line inference / Red for No Line inference
                color = LINE_COLOR if line_detected else NO_LINE_COLOR

            if history_uncertainty:
                # Plot full square with transparency based on the confidence
                edge_color = 'none'
                face_color = color
                alpha = scan_entry.model_confidence
                label = None
            else:
                # Plot empty square with color based on "line"/"no line"
                label = None if line_detected in first_patch_label else f'Infer {QDSDLines.classes[line_detected]}'
                first_patch_label.add(line_detected)

                edge_color = color
                face_color = 'none'
                alpha = 1
                legend = True

            patch = patches.Rectangle((x_i[x + settings.label_offset_x], y_i[y + settings.label_offset_y]),
                                      patch_size_x_v,
                                      patch_size_y_v,
                                      linewidth=1,
                                      edgecolor=edge_color,
                                      label=label,
                                      facecolor=face_color,
                                      alpha=alpha)
            plt.gca().add_patch(patch)

        # Marker for first point
        if show_crosses:
            first_x, first_y = scan_history[0].coordinates
            plt.scatter(x=x_i[first_x + settings.patch_size_x // 2], y=y_i[first_y + settings.patch_size_y // 2],
                        color='skyblue', marker='X', s=200, label='Start')
            legend = True

        if history_uncertainty:
            # Set up the colorbar
            if scan_errors:
                cmap = LinearSegmentedColormap.from_list('', [GOOD_COLOR, 'white', ERROR_COLOR])
            else:
                cmap = LinearSegmentedColormap.from_list('', [LINE_COLOR, 'white', ERROR_COLOR])
            norm = Normalize(vmin=-1, vmax=1)
            cbar = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), shrink=0.8, aspect=15)
            cbar.outline.set_edgecolor('0.15')
            cbar.set_ticks([-1, 0, 1])
            if scan_errors:
                cbar.set_ticklabels(['Correct class\nLow uncertainty', 'High uncertainty',
                                     'Error class\nLow uncertainty'])
            else:
                cbar.set_ticklabels(['Line\nLow uncertainty', 'High uncertainty', 'No Line\nLow uncertainty'])

    # Marker for tuning final guess
    if show_crosses and final_coord is not None:
        last_x, last_y = final_coord
        # Get marker position (and avoid going out)
        last_x_i = min(last_x, len(x_i) - 1)
        last_y_i = min(last_y, len(y_i) - 1)
        plt.scatter(x=x_i[last_x_i], y=y_i[last_y_i],
                    color='fuchsia', marker='x', s=200, label='End')
        legend = True

    if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
        focus_x, focus_y = focus_area if focus_area else 0, 0

        # Create a Rectangle patch
        rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                 focus_x + settings.patch_size_x - 2 * settings.label_offset_x,
                                 focus_y + settings.patch_size_y - 2 * settings.label_offset_y,
                                 linewidth=2, edgecolor='fuchsia', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    if text_stats:
        text = ''
        if scan_history and len(scan_history) > 0:
            # Local import to avoid circular mess
            from datasets.qdsd import QDSDLines

            accuracy = sum(1 for s in scan_history if s.is_classification_correct()) / len(scan_history)
            nb_line = sum(1 for s in scan_history if s.ground_truth)  # s.ground_truth == True means line
            nb_no_line = sum(1 for s in scan_history if not s.ground_truth)  # s.ground_truth == False means no line

            if nb_line > 0:
                line_success = sum(
                    1 for s in scan_history if s.ground_truth and s.is_classification_correct()) / nb_line
            else:
                line_success = None

            if nb_no_line > 0:
                no_line_success = sum(1 for s in scan_history
                                      if not s.ground_truth and s.is_classification_correct()) / nb_no_line
            else:
                no_line_success = None

            last_correct = scan_history[-1].is_classification_correct()
            last_class = QDSDLines.classes[scan_history[-1].model_classification]

            text += f'Nb step: {len(scan_history): >3n} (acc: {accuracy: >4.0%})\n'
            text += f'{QDSDLines.classes[True].capitalize(): <7}: {nb_line: >3n}'
            text += '\n' if line_success is None else f' (acc: {line_success:>4.0%})\n'
            text += f'{QDSDLines.classes[False].capitalize(): <7}: {nb_no_line: >3n}'
            text += '\n' if no_line_success is None else f' (acc: {no_line_success:>4.0%})\n\n'
            text += f'Last scan:\n'
            text += f'  - Pred: {last_class.capitalize(): <7} {"(good) " if last_correct else "(wrong)"}\n'
            text += f'  - Conf: {scan_history[-1].model_confidence: >4.0%}\n\n'
            text += f'Tuning step:\n'
            text += f'  {scan_history[-1].description}'

        plt.text(1.03, 0.8, text, horizontalalignment='left', verticalalignment='top', fontsize=8,
                 fontfamily='monospace', transform=plt.gca().transAxes)

    if show_title:
        plt.title(f'{image_name}\ninterpolated ({interpolation_method}) - pixel size {round(pixel_size, 10) * 1_000}mV')

    plt.xlabel('Gate 1 (V)')
    plt.xticks(rotation=30)
    plt.ylabel('Gate 2 (V)')

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        handler_map = None
        if charge_text is not None:
            # Create custom legend for charge regime text
            charge_text = copy(charge_text)
            charge_text.set(text='N')
            handler_map = {type(charge_text): TextHandler()}
            handles.append(charge_text)
            labels.append('Charge regime')

        plt.legend(ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.35), handles=handles, labels=labels,
                   handler_map=handler_map)

    if focus_area:
        plt.axis(focus_area)

    return save_plot(f'diagram_{image_name}', save_in_buffer=save_in_buffer)


def plot_diagram_step_animation(d: "Diagram", image_name: str, scan_history: List["StepHistoryEntry"],
                                final_coord: Tuple[int, int]) -> None:
    """
    Plot an animation of the tuning procedure.

    :param d: The diagram to plot.
    :param image_name: The name of the image, used for plot title and file name
    :param scan_history: The tuning steps history (see StepHistoryEntry dataclass)
    :param final_coord: The final tuning coordinates
    """

    if settings.is_named_run() and (settings.save_gif or settings.save_video):
        values = d.values.cpu()
        # Animation speed => Time for an image (ms)
        base_fps = 100
        # Ratio of image to skip for the animation frames (1 means nothing skipped, 4 means 1 keep for 3 skip)
        rate_gif = 4
        rate_video = 1
        # Define frames to compute
        gif_frames_ids = list(range(0, len(scan_history), rate_gif)) if settings.save_gif else []
        video_frames_ids = list(range(0, len(scan_history), rate_video)) if settings.save_video else []
        frame_ids = list(dict.fromkeys(gif_frames_ids + video_frames_ids))  # Remove duplicate and keep order

        with Pool(get_nb_loader_workers()) as pool:
            # Generate images in parallel for speed. Use partial to set constants arguments.
            # Main animation frames
            async_result_main = pool.map_async(
                partial(plot_diagram, d.x_axes, d.y_axes, values, d.file_basename, 'nearest', d.x_axes[1] - d.x_axes[0],
                        None, None, None, False, save_in_buffer=True, text_stats=True, show_title=False,
                        fog_of_war=True), (scan_history[0:i] for i in frame_ids))

            # Final frames
            async_result_end = [
                # Show full diagram with tuning final coordinate and fog of war
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values,
                                       'image_name': d.file_basename,
                                       'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                       'scan_history': scan_history, 'final_coord': final_coord, 'show_offset': False,
                                       'save_in_buffer': True, 'text_stats': True, 'show_title': False,
                                       'fog_of_war': True}),
                # Show full diagram with tuning final coordinate
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values,
                                       'image_name': d.file_basename,
                                       'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                       'scan_history': scan_history, 'final_coord': final_coord, 'show_offset': False,
                                       'save_in_buffer': True, 'text_stats': True, 'show_title': False}),
                # Show full diagram with tuning final coordinate + line labels
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values,
                                       'image_name': d.file_basename,
                                       'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                       'scan_history': scan_history, 'final_coord': final_coord, 'show_offset': False,
                                       'save_in_buffer': True, 'text_stats': True, 'show_title': False,
                                       'transition_lines': d.transition_lines}),
                # Show full diagram with tuning final coordinate + line & regime labels
                pool.apply_async(plot_diagram,
                                 kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values,
                                       'image_name': d.file_basename,
                                       'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                       'scan_history': scan_history, 'final_coord': final_coord, 'show_offset': False,
                                       'save_in_buffer': True, 'text_stats': True, 'show_title': False,
                                       'transition_lines': d.transition_lines, 'charge_regions': d.charge_areas}),
            ]

            # Wait for the processes to finish and get result
            pool.close()
            pool.join()
            main_frames = async_result_main.get()
            end_frames = [res.get() for res in async_result_end]

        end_durations = [base_fps * 10, base_fps * 10, base_fps * 20, base_fps * 70]
        if settings.save_gif:
            # List of image bytes for the animation
            frames_gif = [main_frames[frame_ids.index(f_id)] for f_id in gif_frames_ids] + end_frames
            # List of duration for each image (ms)
            durations_gif = [base_fps * rate_gif] * len(gif_frames_ids) + end_durations
            save_gif(frames_gif, image_name, duration=durations_gif)

        if settings.save_video:
            # List of image bytes for the animation
            frames_video = [main_frames[frame_ids.index(f_id)] for f_id in video_frames_ids] + end_frames
            # List of duration for each image (ms)
            durations_video = [base_fps * rate_video] * len(video_frames_ids) + end_durations
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
                             interpolation='nearest',
                             cmap='copper')

            if show_offset and (settings.label_offset_x != 0 or settings.label_offset_y != 0):
                # Create a rectangle patch that represent offset
                rect = patches.Rectangle((settings.label_offset_x - 0.5, settings.label_offset_y - 0.5),
                                         settings.patch_size_x - 2 * settings.label_offset_x,
                                         settings.patch_size_y - 2 * settings.label_offset_y,
                                         linewidth=2, edgecolor='fuchsia', facecolor='none')

                # Add the offset rectangle to the axes
                axs[i, j].add_patch(rect)

            axs[i, j].axis('off')

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
