import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, IO, List, Optional, Sequence, Tuple, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon

from plots.data import plot_diagram
from utils.miscs import clip


@dataclass
class Diagram:
    """ Handle the diagram data and its annotations. """

    # The file name of this diagram (without file extension)
    file_basename: str

    # The list of voltage for the first gate
    x: Sequence[float]

    # The list of voltage for the second gate
    y: Sequence[float]

    # The list of measured voltage according to the 2 gates
    values: Sequence[float]

    # The transition lines annotations
    transition_lines: List[LineString]

    def get_patches(self, patch_size: Tuple[int, int] = (10, 10), overlap: Tuple[int, int] = (0, 0)) -> Generator:
        """
        Create patches from diagrams sub-area.

        :param patch_size: The size of the desired patches in number of pixels (x, y)
        :param overlap: The size of the patches overlapping in number of pixels (x, y)
        :return: A generator of patches.
        """
        patch_size_x, patch_size_y = patch_size
        overlap_size_x, overlap_size_y = overlap
        diagram_size_y, diagram_size_x = self.values.shape

        # Extract each patches
        i = 0
        for patch_y in range(0, diagram_size_y - patch_size_y, patch_size_y - overlap_size_y):
            # Patch coordinates (indexes)
            start_y = patch_y
            end_y = patch_y + patch_size_y
            # Patch coordinates (voltage)
            start_y_v = self.y[start_y]
            end_y_v = self.y[end_y]
            for patch_x in range(0, diagram_size_x - patch_size_x, patch_size_x - overlap_size_x):
                i += 1
                # Patch coordinates (indexes)
                start_x = patch_x
                end_x = patch_x + patch_size_x
                # Patch coordinates (voltage)
                start_x_v = self.x[start_x]
                end_x_v = self.x[end_x]

                # Create patch shape to find line intersection
                patch_shape = Polygon([(start_x_v, start_y_v),
                                       (end_x_v, start_y_v),
                                       (end_x_v, end_y_v),
                                       (start_x_v, end_y_v)])

                # Extract patch value
                patch = self.values[start_y:end_y, start_x:end_x]
                # Label is True if any line intersect the patch shape
                label = any([line.intersects(patch_shape) for line in self.transition_lines])

                # self.plot((start_x_v, end_x_v, start_y_v, end_y_v), f' - patch {i:n} - line {label}')
                yield patch, label

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        """
        Plot the diagram with matplotlib (save and/or show it depending on the settings).
        This method is a shortcut of plots.diagram.plot_diagram.

        :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
        :param label_extra: Optional extra information for the plot label.
        """
        plot_diagram(self.x, self.y, self.values, self.file_basename + label_extra, 'nearest', self.x[1] - self.x[0],
                     transition_lines=self.transition_lines, focus_area=focus_area)

    @staticmethod
    def load_diagrams(diagrams_path, transition_lines_path) -> List["Diagram"]:
        """
        Load stability diagrams and annotions from files.

        :param diagrams_path: The path to the zip file containing all stability diagrams data.
        :param transition_lines_path: The path to the csv file containing all stability diagrams annotations.
        :return: A list of Diagram objects.
        """
        # Open the file containing transition line annotations as a dataframe
        lines_annotations_df = pd.read_csv(transition_lines_path,
                                           usecols=[1, 2, 3, 4, 5],
                                           names=['x1', 'y1', 'x2', 'y2', 'image_name'])

        diagrams = []
        # Open the zip file and iterate over all csv files
        with ZipFile(diagrams_path, 'r') as zip_file:
            diagram_names = zip_file.namelist()
            for diagram_name in diagram_names:
                file_basename = Path(diagram_name).stem  # Remove extension
                with zip_file.open(diagram_name) as diagram_file:
                    # Load values from CSV file
                    x, y, values = Diagram._load_interpolated_csv(gzip.open(diagram_file))

                    transition_lines = Diagram._load_lines_annotations(lines_annotations_df,
                                                                       f'{file_basename}.png',
                                                                       x, y, snap=1)

                    diagram = Diagram(file_basename, x, y, values, transition_lines)
                    diagram.plot()
                    diagrams.append(diagram)

        return diagrams

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

            line_x = Diagram._coord_to_volt(line_x, min_x, max_x, x[0], step, snap)
            line_y = Diagram._coord_to_volt(line_y, min_y, max_y, y[0], step, snap, True)

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
