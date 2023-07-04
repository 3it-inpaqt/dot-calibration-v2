import gzip
import json
import platform
import zipfile
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import torch
from shapely.geometry import LineString, Point, Polygon

from classes.data_structures import ChargeRegime
from datasets.diagram_offline import DiagramOffline
from utils.logger import logger
from utils.settings import settings


class DiagramOfflineNDot(DiagramOffline):
    """ Handle the diagram data and its annotations for Ndots. """

    # The transition lines annotations
    # A list of each line [line_1: [...], line_2: [...], etc...]
    transition_lines: Optional[List[List[LineString]]]

    def get_charge(self, coord_x: int, coord_y: int) -> ChargeRegime:
        """
        Get the charge regime of a specific location in the diagram.

        :param coord_x: The x coordinate to check (not the voltage)
        :param coord_y: The y coordinate to check (not the voltage)
        :return: The charge regime
        """
        volt_x = self.x_axes[coord_x]
        volt_y = self.y_axes[coord_y]
        point = Point(volt_x, volt_y)

        # Coordinates not found in labeled areas. The charge area in this location is thus unknown.
        regime = [ChargeRegime['UNKNOWN']] * settings.dot_number

        # Check coordinates in each labeled area
        for diagram_regime, area in self.charge_areas:
            if area.contains(point):
                regime[int(diagram_regime[-1]) - 1] = diagram_regime
        return regime

    def is_line_in_patch(self, coordinate: Tuple[int, int],
                         patch_size: Tuple[int, int],
                         offsets: Tuple[int, int] = (0, 0)) -> int:
        """
        Check if a line label intersect a specific sub-area (patch) of the diagram.

        :param coordinate: The patch top left coordinates
        :param patch_size: The patch size
        :param offsets: The patch offset (area to ignore lines)
        :return: True if a line intersect the patch (offset excluded)
        """
        coord_x, coord_y = coordinate
        size_x, size_y = patch_size
        offset_x, offset_y = offsets

        # Subtract the offset and convert to voltage
        start_x_v = self.x_axes[coord_x + offset_x]
        start_y_v = self.y_axes[coord_y + offset_y]
        end_x_v = self.x_axes[coord_x + size_x - offset_x]
        end_y_v = self.y_axes[coord_y + size_y - offset_y]

        # Create patch shape to find line intersection
        patch_shape = Polygon([(start_x_v, start_y_v),
                               (end_x_v, start_y_v),
                               (end_x_v, end_y_v),
                               (start_x_v, end_y_v)])

        from datasets.qdsd import QDSDLines

        area_class = []
        for line_type in self.transition_lines:
            area_class.append(any([line.intersects(patch_shape) for line in line_type]))
        return QDSDLines.class_mapping(torch.Tensor(area_class))

    @staticmethod
    def load_diagrams(pixel_size,
                      research_group,
                      diagrams_path: Path,
                      labels_path: Path = None,
                      single_dot: bool = True,
                      load_lines: bool = True,
                      load_areas: bool = True,
                      white_list: List[str] = None) -> List["DiagramOfflineNDot"]:
        """
        Load stability diagrams and annotions from files.

        :param pixel_size: The size of one pixel in volt
        :param research_group: The research_group name for the dataset to load
        :param single_dot: If True, only the single dot diagram will be loaded, if False only the double dot
        :param diagrams_path: The path to the zip file containing all stability diagrams data.
        :param labels_path: The path to the json file containing line and charge area labels.
        :param load_lines: If True the line labels should be loaded.
        :param load_areas: If True the charge area labels should be loaded.
        :param white_list: If defined, only diagrams with base name include in this list will be loaded (no extension).
        :param black_list: If defined, diagrams with base name include in this list will be excluded (no extension).
        :return: A list of Diagram objects.
        """

        # Open the json file that contains annotations for every diagrams
        with open(labels_path, 'r') as annotations_file:
            labels_json = json.load(annotations_file)

        logger.debug(f'{len(labels_json)} labeled diagrams found')
        labels = {obj['External ID']: obj for obj in labels_json}

        # Open the zip file and iterate over all csv files
        # in_zip_path should use "/" separator, no matter the current OS
        in_zip_path = f'{pixel_size * 1000}mV/' + ('single' if single_dot else 'double') + f'/{research_group}/'

        # for general load
        # in_zip_path = f'{pixel_size * 1000}mV/{settings.dot_number}_dot/{research_group}/'

        logger.debug(f'Path file: {in_zip_path}')

        zip_dir = zipfile.Path(diagrams_path, at=in_zip_path)

        if not zip_dir.is_dir():
            raise ValueError(f'Folder "{in_zip_path}" not found in the zip file "{diagrams_path}".'
                             f'Check if pixel size and research group exist in this folder.')

        diagrams = []
        nb_no_label = 0
        nb_excluded_blacklist = 0
        nb_excluded_whitelist = 0
        list_blacklist = []
        list_whitelist = []

        # Iterate over all csv files inside the zip file
        for diagram_name in zip_dir.iterdir():
            file_basename = Path(str(diagram_name)).stem  # Remove extension

            if file_basename == 'nov14100s':
                file_basename += '.gz'

            if white_list and not (file_basename in white_list):
                nb_excluded_whitelist += 1
                list_whitelist.append(file_basename)
                continue

            if settings.black_list and (file_basename in settings.black_list):
                nb_excluded_blacklist += 1
                list_blacklist.append(file_basename)
                continue

            if f'{file_basename}.png' not in labels:
                logger.warning(f'No label found for {file_basename}')
                nb_no_label += 1
                continue

            # Windows needs the 'b' option
            open_options = 'rb' if platform.system() == 'Windows' else 'r'
            with diagram_name.open(open_options) as diagram_file:
                # Load values from CSV file
                x, y, values = DiagramOfflineNDot._load_interpolated_csv(gzip.open(diagram_file))

                current_labels = labels[f'{file_basename}.png']['Label']
                label_pixel_size = float(next(filter(lambda l: l['title'] == 'pixel_size_volt',
                                                     current_labels['classifications']))['answer'])

                transition_lines = []
                charge_areas = None

                if load_lines:
                    for nb in range(1, settings.dot_number + 1):
                        # Load transition line annotations
                        transition_line = DiagramOfflineNDot._load_lines_annotations(
                            filter(lambda l: l['title'] == f'line_{nb}', current_labels['objects']), x, y,
                            pixel_size=label_pixel_size,
                            snap=1)

                        transition_lines.append(transition_line)

                    if len(transition_lines) != settings.dot_number:
                        logger.warning(f'Wrong number of line found: expected {settings.dot_number}, '
                                       f'found {len(transition_lines)} for {file_basename}')
                        nb_no_label += 1
                        continue

                if load_areas:
                    # Load charge area annotations
                    charge_areas = \
                        DiagramOfflineNDot._load_charge_annotations(filter(lambda l: l['title'] in ChargeRegime.keys(),
                                                                           current_labels['objects']), x, y,
                                                                    pixel_size=label_pixel_size, snap=1)

                    if not charge_areas:
                        logger.warning(f'No charge label found for {file_basename}')
                        nb_no_label += 1
                        continue

                diagram = DiagramOfflineNDot(file_basename, x, y, values, transition_lines, charge_areas)
                diagrams.append(diagram)
                if settings.plot_diagrams:
                    diagram.plot()

        if nb_no_label > 0:
            logger.warning(f'{nb_no_label} diagram(s) skipped because no label found')

        if nb_excluded_whitelist > 0:
            logger.warning(f'{nb_excluded_whitelist} diagram(s) excluded because not in white list')
            logger.debug(f'Diagram not in the whitelist: {list_whitelist}')
        if nb_excluded_blacklist > 0:
            logger.warning(f'{nb_excluded_blacklist} diagram(s) excluded because in black list')
            logger.debug(f'Diagram in the blacklist: {list_blacklist}')

        if len(diagrams) == 0:
            logger.error(f'No diagram loaded in "{zip_dir}"')

        return diagrams

    def get_patches(self, patch_size: Tuple[int, int] = (10, 10), overlap: Tuple[int, int] = (0, 0),
                    label_offset: Tuple[int, int] = (0, 0)) -> Generator:
        """
        Create patches from diagrams sub-area.

        :param patch_size: The size of the desired patches, in number of pixels (x, y)
        :param overlap: The size of the patches overlapping, in number of pixels (x, y)
        :param label_offset: The width of the border to ignore during the patch labeling, in number of pixel (x, y)
        :return: A generator of patches.
        """
        patch_size_x, patch_size_y = patch_size
        overlap_size_x, overlap_size_y = overlap
        label_offset_x, label_offset_y = label_offset
        diagram_size_y, diagram_size_x = self.values.shape

        # Extract each patches
        i = 0
        for patch_y in range(0, diagram_size_y - patch_size_y, patch_size_y - overlap_size_y):
            # Patch coordinates (indexes)
            start_y = patch_y
            end_y = patch_y + patch_size_y
            # Patch coordinates (voltage)
            start_y_v = self.y_axes[start_y + label_offset_y]
            end_y_v = self.y_axes[end_y - label_offset_y]
            for patch_x in range(0, diagram_size_x - patch_size_x, patch_size_x - overlap_size_x):
                i += 1
                # Patch coordinates (indexes)
                start_x = patch_x
                end_x = patch_x + patch_size_x
                # Patch coordinates (voltage) for label area
                start_x_v = self.x_axes[start_x + label_offset_x]
                end_x_v = self.x_axes[end_x - label_offset_x]

                # Create patch shape to find line intersection
                patch_shape = Polygon([(start_x_v, start_y_v),
                                       (end_x_v, start_y_v),
                                       (end_x_v, end_y_v),
                                       (start_x_v, end_y_v)])

                # Extract patch value
                # Invert Y axis because the diagram origin (0,0) is top left
                patch = self.values[diagram_size_y - end_y:diagram_size_y - start_y, start_x:end_x]
                # Label is True if any line intersect the patch shape
                labels = []
                for transition_lines in self.transition_lines:
                    labels.append(any([line.intersects(patch_shape) for line in transition_lines]))

                # Verification plots
                # plot_diagram(self.x[start_x:end_x], self.y[start_y:end_y],
                #              self.values[diagram_size_y-end_y:diagram_size_y-start_y, start_x:end_x],
                #              self.name + f' - patch {i:n} - line {label} - REAL',
                #              'nearest', self.x[1] - self.x[0])
                # self.plot((start_x_v, end_x_v, start_y_v, end_y_v), f' - patch {i:n} - line {label}')
                yield patch, labels
