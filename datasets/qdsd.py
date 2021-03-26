import gzip
from pathlib import Path
from random import shuffle
from typing import List, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from shapely.geometry import LineString
from torch.utils.data import Dataset
from typing.io import IO

from classes.diagram import Diagram
from utils.logger import logger
from utils.miscs import clip
from utils.output import load_data_cache, save_data_cache, save_results
from utils.settings import settings

DATA_DIR = Path('data')


class QDSDLines(Dataset):
    """
    Quantum Dots Stability Diagrams (QDSD) dataset.
    Transition line classification task.
    """

    classes = [
        'no line',  # False (0)
        'line'  # True (1)
    ]

    def __init__(self, patches: List[Tuple], role: str):
        """
        Create a dataset of transition lines patches.
        Should be build with QDSDLines.build_split_datasets.

        :param patches: The list of patches as (2D array of values, labels as boolean)
        :param role: The role of this dataset ("train" or "test" or "validation")
        """

        self.role = role

        # Get patches and their labels (using ninja unzip)
        self._patches, self._patches_labels = zip(*patches)

        # Convert to torch tensor
        self._patches = torch.Tensor(self._patches)
        self._patches_labels = torch.Tensor(self._patches_labels).bool()

        # TODO the normalisation should be done for train and test at the same time
        # Normalise data voltage
        self._patches -= torch.min(self._patches)
        self._patches /= torch.max(self._patches)

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, index):
        # TODO Use a more generic transform method
        return torch.flatten(self._patches[index]), self._patches_labels[index]

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the dataset to a specific device (cpu or cuda) and/or a convert it to a different type.
        Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self._patches = self._patches.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        self._patches_labels = self._patches_labels.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    def get_stats(self) -> dict:
        """
        :return: Some statistics about this dataset
        """
        nb_patche = len(self)
        nb_line = torch.sum(self._patches_labels)
        return {
            f'{self.role}_dataset_size': nb_patche,
            f'{self.role}_dataset_nb_line': nb_line,
            f'{self.role}_dataset_nb_noline': nb_patche - nb_line,
        }

    @staticmethod
    def build_split_datasets(test_ratio: float, validation_ratio: float = 0, patch_size: Tuple[int, int] = (10, 10),
                             overlap: Tuple[int, int] = (0, 0)) -> Tuple["QDSDLines", ...]:
        """
        Initialize dataset of transition lines patches.
        The sizes of the test and validation dataset depend on the ratio. The train dataset get all remaining data.

        :param test_ratio: The ratio of patches reserved for test data
        :param validation_ratio: The ratio of patches reserved for validation data (ignored if 0)
        :param patch_size: The size in pixel (x, y) of the patch to process (sub-area of the stability diagram)
        :param overlap: The overlapping size in pixel (x, y) of the patch
        :return A tuple of datasets: (train, test, validation) or (train, test) if validation_ratio is 0
        """

        cache_path = Path(DATA_DIR, 'cache', f'qdsd_lines_{patch_size[0]}-{patch_size[1]}_{overlap[0]}-{overlap[1]}.p')
        if settings.use_data_cache and cache_path.is_file():
            # Fast load from cache
            patches = load_data_cache(cache_path)
        else:
            patches = []
            # Open the file containing transition line annotations as a dataframe
            lines_annotations_df = pd.read_csv(Path(DATA_DIR, 'transition_lines.csv'),
                                               usecols=[1, 2, 3, 4, 5],
                                               names=['x1', 'y1', 'x2', 'y2', 'image_name'])

            # Open the zip file and iterate over all csv files
            with ZipFile(Path(DATA_DIR, 'interpolated_csv.zip'), 'r') as zip_file:
                diagram_names = zip_file.namelist()
                nb_diagram = len(diagram_names)
                for diagram_name in diagram_names:
                    file_basename = Path(diagram_name).stem  # Remove extension
                    with zip_file.open(diagram_name) as diagram_file:
                        # Load values from CSV file
                        x, y, values = QDSDLines._load_interpolated_csv(gzip.open(diagram_file))

                        transition_lines = QDSDLines._load_lines_annotations(lines_annotations_df,
                                                                             f'{file_basename}.png',
                                                                             x, y, snap=1)

                        diagram = Diagram(file_basename, x, y, values, transition_lines)
                        diagram.plot()

                        patches.extend(diagram.get_patches(patch_size, overlap))

            logger.info(f'{len(patches)} items loaded from {nb_diagram} diagrams')

            if settings.use_data_cache:
                # Save in cache for later runs
                save_data_cache(cache_path, patches)

        nb_patches = len(patches)

        # Shuffle patches before to split them into different the datasets
        shuffle(patches)

        test_index = round(nb_patches * test_ratio)
        test_set = QDSDLines(patches[:test_index], 'test')

        # With validation dataset
        if validation_ratio != 0:
            valid_index = test_index + round(nb_patches * validation_ratio)
            valid_set = QDSDLines(patches[test_index:valid_index], 'validation')
            train_set = QDSDLines(patches[valid_index:], 'train')

            datasets = (train_set, test_set, valid_set)

        # No validation dataset
        else:
            train_set = QDSDLines(patches[test_index:], 'train')

            datasets = (train_set, test_set)

        # Print and save stats about datasets
        stats = {}
        for dataset in datasets:
            stats.update(dataset.get_stats())

        logger.debug('Dataset:' + ''.join([f'\n\t{key}: {value}' for key, value in stats.items()]))
        save_results(**stats)

        return datasets

    @staticmethod
    def _load_interpolated_csv(file_path: Union[IO, str, Path]) -> Tuple:
        """
        Load the stability diagrams from CSV file.

        :param file_path: The path to the CSV file or the byte stream.
        :return: The stability diagram data as a tuple: x, y, values
        """
        compact_diagram = np.loadtxt(file_path, delimiter=',')
        # Extract information
        x_start, y_start, step = compact_diagram[0][0], compact_diagram[0][1], compact_diagram[0][2]

        # Remove the information row
        values = np.delete(compact_diagram, 0, 0)

        # Reconstruct the axes

        x = np.arange(values.shape[1]) * step + x_start
        y = np.arange(values.shape[0]) * step + y_start

        return x, y, values

    @staticmethod
    def _load_lines_annotations(lines_annotations_df, image_name: str, x, y, snap: int = 1) -> List[LineString]:
        """
        Load transition line annotations for an image.

        :param lines_annotations_df: The dataframe structure containing all annotations
        :param image_name: The name of the image (should match with the name in the annotation file)
        :param x: The x axis of the diagram (in volt)
        :param y: The y axis of the diagram (in volt)
        :param snap: The snap margin, every points near to image border at this distance will be rounded to the image border
        (in number of pixels)
        :return: The list of line annotation for the image, as shapely.geometry.LineString
        """
        current_file_lines = lines_annotations_df[lines_annotations_df['image_name'] == image_name]

        # Define borders for snap
        min_x, max_x = 0, len(x) - 1
        min_y, max_y = 0, len(y) - 1
        # Step (should be the same for every measurement)
        step = x[1] - x[0]

        lines = []
        for _, l in current_file_lines.iterrows():
            line_x = [l['x1'], l['x2']]
            line_y = [l['y1'], l['y2']]

            line_x = QDSDLines._coord_to_volt(line_x, min_x, max_x, x[0], step, snap)
            line_y = QDSDLines._coord_to_volt(line_y, min_y, max_y, y[0], step, snap, True)

            line = LineString(zip(line_x, line_y))
            lines.append(line)

        return lines

    @staticmethod
    def _coord_to_volt(coord: List[float], min_coord: int, max_coord: int, value_start: float, value_step: float,
                       snap: int = 1, is_y: bool = False) -> List[float]:
        """
        Convert some coordinates to volt value for a specific stability diagram.

        :param coord: The list coordinates to convert
        :param min_coord: The minimal valid value for the coordinate (before volt conversion)
        :param max_coord: The maximal valid value for the coordinate (before volt conversion)
        :param value_start: The voltage value of the 0 coordinate
        :param value_step: The voltage difference between two coordinates (pixel size)
        :param snap: The snap margin, every points near to image border at this distance will be rounded to the image border
        (in number of pixels)
        :param is_y: If true this is the y axis (to apply a rotation)
        :return: The list of coordinates as gate voltage values
        """
        if is_y:
            # Flip Y axis (I don't know why it's required)
            coord = list(map(lambda t: max_coord - t, coord))

        # Snap to border to avoid errors
        coord = list(map(lambda t: clip(t, min_coord, max_coord), coord))

        # Convert coordinates to actual voltage value
        coord = list(map(lambda t: t * value_step + value_start, coord))

        return coord
