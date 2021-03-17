import gzip
import json
from pathlib import Path
from typing import List, Tuple
from zipfile import ZipFile

import pandas as pd
from shapely.geometry import LineString, Polygon

DATA_DIR = Path('data')


def clip(n, smallest, largest):
    """ Shortcut to clip a value between 2 others """
    return max(smallest, min(n, largest))


def coord_to_volt(coord: List[float], min_coord: int, max_coord: int, value_start: float, value_step: float,
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


def load_charge_annotations(annotations_json, image_name: str, x, y, snap: int = 1) -> List[Tuple[str, Polygon]]:
    """
    Load regions annotation for an image.

    :param annotations_json: The json structure containing all annotations
    :param image_name: The name of the image (should match with the name in the annotation file)
    :param x: The x axis of the diagram (in volt)
    :param y: The y axis of the diagram (in volt)
    :param snap: The snap margin, every points near to image border at this distance will be rounded to the image border
    (in number of pixels)
    :return: The list of regions annotation for the image, as (label, shapely.geometry.Polygon)
    """

    if image_name not in annotations_json:
        raise RuntimeError(f'"{image_name}" annotation not found')

    annotation_json = annotations_json[image_name]

    # Define borders for snap
    min_x, max_x = 0, len(x) - 1
    min_y, max_y = 0, len(y) - 1
    # Step (should be the same for every measurement)
    step = x[1] - x[0]

    regions = []

    for region in annotation_json['regions'].values():
        x_r = region['shape_attributes']['all_points_x']
        y_r = region['shape_attributes']['all_points_y']
        label_r = region['region_attributes']['label']

        x_r = coord_to_volt(x_r, min_x, max_x, x[0], step, snap)
        y_r = coord_to_volt(y_r, min_y, max_y, y[0], step, snap, True)

        # Close regions
        x_r.append(x_r[-1])
        y_r.append(y_r[-1])

        polygon = Polygon(zip(x_r, y_r))
        regions.append((label_r, polygon))

    return regions


def load_lines_annotations(lines_annotations_df, image_name: str, x, y, snap: int = 1):
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

        line_x = coord_to_volt(line_x, min_x, max_x, x[0], step, snap)
        line_y = coord_to_volt(line_y, min_y, max_y, y[0], step, snap, True)

        line = LineString(zip(line_x, line_y))
        lines.append(line)

    return lines


def main():
    # Open the json file that can contain annotations for every diagrams
    with open(Path(DATA_DIR, 'charge_area.json'), 'r') as annotations_file:
        charge_annotations_json = json.load(annotations_file)

    lines_annotations_df = pd.read_csv(Path(DATA_DIR, 'transition_lines.csv'),
                                       usecols=[1, 2, 3, 4, 5],
                                       names=['x1', 'y1', 'x2', 'y2', 'image_name'])

    # Open the zip file and iterate over all csv files
    with ZipFile(Path(DATA_DIR, 'interpolated_csv.zip'), 'r') as zip_file:
        for diagram_name in zip_file.namelist():
            file_basename = Path(diagram_name).stem  # Remove extension
            with zip_file.open(diagram_name) as diagram_file:
                # Load values from CSV file
                x, y, values = load_interpolated_csv(gzip.open(diagram_file))

                # Load annotation and convert the coordinates to volt
                charge_regions = load_charge_annotations(charge_annotations_json, f'{file_basename}.png', x, y, snap=1)
                transition_lines = load_lines_annotations(lines_annotations_df, f'{file_basename}.png', x, y, snap=1)

                plot_image(x, y, values, file_basename, 'nearest', x[1] - x[0], charge_regions, transition_lines)
