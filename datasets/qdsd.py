import gzip
from pathlib import Path
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

DATA_DIR = Path('data')


class QDSDLines(Dataset):
    """
    Quantum Dots Stability Diagrams (QDSD) dataset.
    """

    def __init__(self, test: bool = False, patch_size: Tuple[int, int] = (10, 10), overlap: Tuple[int, int] = (0, 0)):
        """
        Create the dataset.

        :param test: If True load the testing data
        :param patch_size: The size in pixel (x, y) of the patch to process (sub-area of the stability diagram)
        :param overlap: The overlapping size in pixel (x, y) of the patch
        """

        self._diagrams = []
        self._patches = []
        self._patches_labels = []

        # Open the file containing transition line annotations as a dataframe
        lines_annotations_df = pd.read_csv(Path(DATA_DIR, 'transition_lines.csv'),
                                           usecols=[1, 2, 3, 4, 5],
                                           names=['x1', 'y1', 'x2', 'y2', 'image_name'])

        # Open the zip file and iterate over all csv files
        with ZipFile(Path(DATA_DIR, 'interpolated_csv.zip'), 'r') as zip_file:
            for diagram_name in zip_file.namelist():
                file_basename = Path(diagram_name).stem  # Remove extension
                with zip_file.open(diagram_name) as diagram_file:
                    # Load values from CSV file
                    x, y, values = QDSDLines._load_interpolated_csv(gzip.open(diagram_file))

                    transition_lines = QDSDLines._load_lines_annotations(lines_annotations_df, f'{file_basename}.png',
                                                                         x, y, snap=1)

                    diagram = Diagram(file_basename, x, y, values, transition_lines)
                    self._diagrams.append(diagram)
                    diagram.plot()

                    # Get patches and their labels (using ninja unzip)
                    patches, labels = zip(*list(diagram.get_patches(patch_size, overlap)))

                    self._patches.extend(patches)
                    self._patches_labels.extend(labels)

        # Convert to torch tensor
        self._patches = torch.Tensor(self._patches)
        self._patches_labels = torch.Tensor(self._patches_labels)

        logger.debug(f'{len(self._patches)} patches loaded from {len(self._diagrams)} diagrams')

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, index):
        return self._patches[index], self._patches_labels[index]

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
        pass

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
