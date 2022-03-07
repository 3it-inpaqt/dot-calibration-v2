import io
import re
from collections.abc import Iterable
from dataclasses import asdict
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from codetiming import Timer
from torch.nn import Module

from utils.logger import logger
from utils.misc import yaml_preprocess
from utils.settings import settings

OUT_DIR = './out'
OUT_FILES = {
    'settings': 'settings.yaml',
    'results': 'results.yaml',
    'network_info': 'network_info.yaml',
    'timers': 'timers.yaml',
    'normalization': 'normalization.yaml'
}


def init_out_directory() -> None:
    """
    Prepare the output directory.
    """

    # Skip saving if the name of the run is not set
    if settings.is_unnamed_run():
        logger.warning('Nothing will be saved because the name of the run is not set. '
                       'See "run_name" in the setting file to change this behaviours.')
        return

    run_dir = Path(OUT_DIR, settings.run_name)
    img_dir = run_dir / 'img'

    # If the keyword 'tmp' is used as run name, then remove the previous files
    if settings.is_temporary_run():
        logger.warning(f'Using temporary directory to save this run results.')
        if run_dir.exists():
            logger.warning(f'Previous temporary files removed: {run_dir}')
            remove_out_directory(run_dir)

    try:
        # Create the directories
        img_dir.mkdir(parents=True)
    except FileExistsError as err:
        # Clear error message about file exist
        raise ExistingRunName(settings.run_name, run_dir) from err

    logger.debug(f'Output directory created: {run_dir}')

    # Init the logger file
    if settings.logger_file_enable:
        logger.enable_log_file(file_path=(run_dir / 'run.log'), file_log_level=settings.logger_file_level)

    parameter_file = run_dir / OUT_FILES['settings']
    with open(parameter_file, 'w+') as f:
        yaml.dump(asdict(settings), f)

    logger.debug(f'Parameters saved in {parameter_file}')


def remove_out_directory(directory: Path) -> None:
    """
    Definitely remove an output directory.

    :param directory: The path to the directory to remove.
    """
    img_dir = directory / 'img'
    # Remove text files
    (directory / 'run.log').unlink(missing_ok=True)
    for file_name in OUT_FILES.values():
        (directory / file_name).unlink(missing_ok=True)

    # Remove images
    if img_dir.is_dir():
        # Remove png and gif images files
        for image_file in chain(img_dir.glob('*.png'), img_dir.glob('*.gif'), img_dir.glob('*.mp4')):
            image_file.unlink()
        img_dir.rmdir()

    # Remove saved networks
    for p_file in directory.glob('*.pt'):
        p_file.unlink()

    # Remove tmp directory
    directory.rmdir()


def set_plot_style():
    """
    Set plot style.
    """
    sns.set_theme(rc={
        'axes.titlesize': 15,
        'figure.titlesize': 18,
        'axes.labelsize': 13,
        'figure.autolayout': True
    })


def save_network_info(network_metrics: dict) -> None:
    """
    Save metrics information in a file in the run directory.

    :param network_metrics: The dictionary of metrics with their values.
    """

    # Skip saving if the name of the run is not set or nothing to save
    if settings.is_unnamed_run() or len(network_metrics) == 0:
        return

    network_info_file = Path(OUT_DIR, settings.run_name, OUT_FILES['network_info'])

    with open(network_info_file, 'w+') as f:
        yaml.dump(network_metrics, f)

    logger.debug(f'Network info saved in {network_info_file}')


def save_results(**results: Any) -> None:
    """
    Write a new line in the result file.

    :param results: Dictionary of labels and values, could be anything that implement __str__.
    """

    # Skip saving if the name of the run is not set
    if settings.is_unnamed_run():
        return

    results_path = Path(OUT_DIR, settings.run_name, OUT_FILES['results'])

    # Append to the file, create it if necessary
    with open(results_path, 'a') as f:
        results = yaml_preprocess(results)
        yaml.dump(results, f)

    logger.debug(f'{len(results)} result(s) saved in {results_path}')


def get_save_path(directory: Path, file_name: str, extension: str, allow_overwrite: bool = False,
                  index_limit: int = 100) -> Path:
    """
    Get a valid save path if possible. If not allow overwriting, will iterate the name until one available found, or
    limit reached.

    :param directory: The directory to use.
    :param file_name: The target file name (an index could be added).
    :param extension: The file extension.
    :param allow_overwrite: If True allow to return a file path that already exist. If False, will iterate the name.
    :param index_limit: The maximum number of name interation to try.
    :raise FileExistsError: If allow_overwrite is false and index_limit is reached.
    :return: The file path.
    """

    save_path = Path(directory, f'{file_name}.{extension}')

    # Check if file exist and rename if we want to avoid overwriting
    if save_path.is_file() and not allow_overwrite:
        for i in range(2, index_limit, 1):
            save_path = Path(directory, f'{file_name} ({i:02d}).{extension}')
            if not save_path.is_file():
                break  # We found a valid name

        if save_path.is_file():
            # No valid name found
            raise FileExistsError(f'Image name already exist and maximum index reached ({index_limit}): {save_path}')

    return save_path


def save_plot(file_name: str, allow_overwrite: bool = False, save_in_buffer: bool = False) \
        -> Optional[Union[Path, io.BytesIO]]:
    """
    Save a plot image in the directory

    :param file_name: the output png file name (no extension).
    :param allow_overwrite: If True, overwrite existing file if existing with same name. If False, add a number to the
    file name to avoid overwriting.
    :param save_in_buffer: If True, save the image in memory. Do not plot or save it on the disk.

    :return: The path where the plot is saved, or None if not saved.
    """

    # Adjust the padding between and around subplots
    plt.tight_layout()

    # Keep the image in buffer, but no plot and no save it as a file.
    if save_in_buffer:
        buffer = io.BytesIO()
        plt.savefig(buffer, dpi=200, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    save_path = None
    if settings.is_named_run() and settings.save_images:
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), file_name, 'png', allow_overwrite)

        plt.savefig(save_path, dpi=200)
        logger.debug(f'Plot saved in {save_path}')

    # Plot image or close it
    plt.show(block=False) if settings.show_images else plt.close()

    return save_path


def save_gif(images: List[io.BytesIO], file_name: str, duration: Union[List[int], int] = 200, loop: int = 0,
             allow_overwrite: bool = False) -> Optional[Path]:
    """
    Transform a list of image into an animated gif and save it.

    :param images: The list of sorted image data that should be used as image frames.
    :param file_name: The output gif file name (no extension).
    :param duration: The time to display the current frame of the gif, in milliseconds.
    :param loop: The number of times the gif should loop. 0 means that it will loop forever.
    :param allow_overwrite: If True, overwrite existing file if existing with same name. If False, add a number to the
    file name to avoid overwriting.
    :return:The path where the gif is saved, or None if not saved.
    """
    save_path = None
    if settings.is_named_run() and settings.save_gif:
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), file_name, 'gif', allow_overwrite)

        img, *imgs = (Image.open(f) for f in images)
        img.save(fp=save_path, format='GIF', append_images=imgs, save_all=True, duration=duration, loop=loop)

        logger.debug(f'Gif generated from {len(images)} images. Saved in: {save_path}')

    return save_path


def save_video(images: List[io.BytesIO], file_name: str, duration: Union[List[int], int] = 200,
               allow_overwrite: bool = False) -> Optional[Path]:
    """
    Transform a list of image into a video and save it.

    :param images: The list of sorted image data that should be used as image frames.
    :param file_name: The output video file name (no extension).
    :param duration: The time to display the current frame of the video, in milliseconds.
    :param allow_overwrite: If True, overwrite existing file if existing with same name. If False, add a number to the
    file name to avoid overwriting.
    :return:The path where the video is saved, or None if not saved.
    """
    save_path = None
    if settings.is_named_run() and settings.save_video:
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), file_name, 'mp4', allow_overwrite)

        smallest_duration = min(duration) if isinstance(duration, Iterable) else duration
        writer = imageio.get_writer(save_path, fps=1_000 / smallest_duration)
        for data, d in zip(images, duration):
            current_duration = 0
            # Add frames until we reach the targeted duration
            while current_duration < d:
                writer.append_data(imageio.imread(data))
                current_duration += smallest_duration
        writer.close()

        logger.debug(f'Video generated from {len(images)} images. Saved in: {save_path}')

    return save_path


def save_network(network: Module, file_name: str = 'network') -> None:
    """
    Save a full description of the network parameters and states.

    :param network: The network to save
    :param file_name: The name of the destination file (without the extension)
    """

    # Skip saving if the name of the run is not set
    if settings.is_unnamed_run():
        return

    save_path = Path(OUT_DIR, settings.run_name, file_name + '.pt')
    torch.save(network.state_dict(), save_path)
    logger.debug(f'Network saved in {save_path}')


def save_data_cache(file_path: Path, data: torch.Tensor) -> None:
    """
    Save data in pickle file for later fast load.

    :param file_path: The full path to the cache file to write (shouldn't exist).
    :param data: A torch tensor.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, open(file_path, 'wb'))
    logger.debug(f'Data saved in cache ({file_path})')


def save_timers() -> None:
    """
    Save the named timers in a file in the output directory.
    """

    # Skip saving if the name of the run is not set or nothing to save
    if settings.is_unnamed_run() or len(Timer.timers.data) == 0:
        return

    timers_file = Path(OUT_DIR, settings.run_name, OUT_FILES['timers'])
    with open(timers_file, 'w+') as f:
        # Save with replacing white spaces by '_' in timers name
        f.write('# Values in seconds\n')
        yaml.dump({re.sub(r'\s+', '_', n.strip()): v for n, v in Timer.timers.data.items()}, f)

    logger.debug(f'{len(Timer.timers.data)} timer(s) saved in {timers_file}')


def save_normalization(min_value: float, max_value: float) -> None:
    """
    Save normalisation boundaries used during the training.

    :param min_value: The minimal value in the dataset.
    :param max_value: The maximal value in the dataset.
    """

    # Skip saving if the name of the run is not set
    if settings.is_unnamed_run():
        return

    normalization_file = Path(OUT_DIR, settings.run_name, OUT_FILES['normalization'])
    with open(normalization_file, 'w+') as f:
        yaml.dump({'min': min_value, 'max': max_value}, f)

    logger.debug(f'Normalization values saved in {normalization_file}')


def load_normalization() -> Tuple[float, float]:
    """
    Load normalization values from a file, according the normalization_values_path setting.
    :return: The min value and the max value use for the normalization.
    """

    if settings.normalization_values_path:
        # File from settings
        normalization_file = Path(settings.normalization_values_path)
    else:
        # File from current directory (only possible if line task started before)
        normalization_file = Path(OUT_DIR, settings.run_name, OUT_FILES['normalization'])

    if normalization_file.is_file():
        with open(normalization_file) as f:
            result = yaml.load(f, Loader=yaml.FullLoader)
            if 'max' in result and 'min' in result:
                logger.debug(f'Normalization values loaded from file: {normalization_file}')
                return result['min'], result['max']

    raise ValueError(f'Invalid or missing normalisation file: "{normalization_file}". '
                     f'Should be define in "normalization_values_path" setting or present in current un directory.')


def load_network_(network: Module, file_path: Union[str, Path], device: torch.device) -> bool:
    """
    Load a full description of the network parameters and states from a previous save file.

    :param network: The network to load into (in place)
    :param file_path: The path to the file to load
    :param device: The pytorch device where to load the network
    :return: True if the file exist and is loaded, False if the file is not found.
    """

    cache_path = Path(file_path) if isinstance(file_path, str) else file_path
    if cache_path.is_file():
        network.load_state_dict(torch.load(cache_path, map_location=device))
        logger.info(f'Network parameters loaded from file ({cache_path})')
        return True
    logger.warning(f'Network cache not found in "{cache_path}"')
    return False


def load_previous_network_version_(network: Module, version_name: str, device: torch.device) -> bool:
    """
    Load a previous version of the network saved during the current run.

    :param network: The network to load into (in place)
    :param version_name: The name of the version to load (file name without the '.pt')
    :param device: The pytorch device where to load the network
    :return: True if the file exist and is loaded, False if the file is not found or the run is unnamed.
    """
    if settings.is_unnamed_run():
        logger.warning('Impossible to load a previous version of this network because no name is set for this run.')
        return False

    save_path = Path(OUT_DIR, settings.run_name, version_name + '.pt')
    return load_network_(network, save_path, device)


def load_data_cache(file_path: Path) -> torch.Tensor:
    """
    Load data from pickle file (from previous run).

    :param file_path: The full path to the file to load.
    :return: A torch tensor.
    """
    data = torch.load(open(file_path, 'rb'))
    logger.info(f'{len(data)} items loaded from cache ({file_path})')
    return data


def load_run_files(dir_path: Path) -> dict:
    """
    Load all the information of a run from its files.

    :param dir_path: The path to the directory of the run
    :return: A dictionary of every value starting with the name of the file ("file.key": value)
    """
    data = {}

    # For each output file of the run
    for key, file in OUT_FILES.items():
        if (dir_path / file).is_file():
            with open(dir_path / file) as f:
                content = yaml.load(f, Loader=yaml.FullLoader)
                # For each value of each file
                for label, value in content.items():
                    data[key + '.' + label] = value

    if len(data) == 0:
        logger.warning(f'No data loaded from run directory: "{dir_path}"')

    return data


def load_runs(patterns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load all information form files in the out directory matching with patterns.

    :param patterns: The pattern, or a list of pattern, to filter runs.
    :return: A dataframe containing all information, with the columns as "file.key".
    """
    data = []
    runs_dir = Path(OUT_DIR)

    if isinstance(patterns, str):
        # If simple string, turn it into list to be compatible with the following code
        patterns = [patterns]

    for pattern in patterns:
        for run_dir in runs_dir.glob(pattern):
            data.append(load_run_files(run_dir))

    logger.info(f'{len(data)} run(s) loaded with the pattern "{runs_dir}/{patterns}"')

    return pd.DataFrame(data)


class ExistingRunName(Exception):
    """ Exception raised when the user try to start a run with the same name than a previous one. """

    def __init__(self, run_name: str, path: Path, message: str = None):
        if not message:
            message = f'The run name "{run_name}" is already used in the out directory "{path}". ' \
                      f'Change the name in the run settings to a new one or "tmp" or empty.'
            super().__init__(message)
