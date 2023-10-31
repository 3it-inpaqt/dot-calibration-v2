from random import randrange
from typing import List, Tuple, Optional

import numpy as np
import torch

from classes.data_structures import ChargeRegime, ExperimentalMeasurement
from datasets.diagram import Diagram
from plots.data import plot_diagram
from utils.logger import logger
from utils.output import load_normalization
from utils.settings import settings


class DiagramOnline(Diagram):
    _measurement_history: List[ExperimentalMeasurement]
    _norm_min_value: Optional[float]
    _norm_max_value: Optional[float]

    def __init__(self, name: str, connector: "Connector"):
        """
        Create an instance of a DiagramOnline associated with a connector (interface to the measurement tool).

        :param name: The name of the diagram.
        :param connector: The connector to the measurement tool.
        """
        super().__init__(name)
        self._connector = connector
        self._measurement_history = []

        if settings.normalization == 'train-set':
            # Fetch the normalization values used during the training
            self._norm_min_value, self._norm_max_value = load_normalization()
        else:
            # Normalization None or per patch
            self._norm_min_value = self._norm_max_value = None

        # Create a virtual axes and discret grid that represent the voltage space to explore.
        # Where NaN values represent the voltage space that has not been measured yet.
        # The min value is included but not the max value (to match with python standards).
        min_x_v, max_x_v = settings.range_voltage_x
        min_y_v, max_y_v = settings.range_voltage_y
        space_size_x = int((max_x_v - min_x_v) / settings.pixel_size)
        space_size_y = int((max_y_v - min_y_v) / settings.pixel_size)
        self.x_axes = np.linspace(min_x_v, max_x_v, space_size_x, endpoint=False)
        self.y_axes = np.linspace(min_y_v, max_y_v, space_size_y, endpoint=False)
        self.values = torch.full((space_size_y, space_size_x), torch.nan)

        logger.info(f'Initialized online diagram with {space_size_x:,d}x{space_size_y:,d} pixels in range '
                    f'[{min_x_v:.4f}, {max_x_v:.4f}]V x [{min_y_v:.4f}, {max_y_v:.4f}]V')

    def get_random_starting_point(self) -> Tuple[int, int]:
        """
        Generate (pseudo) random coordinates for the top left corder of a patch inside starting range.
        :return: The (pseudo) random coordinates.
        """
        # Get the voltage range
        min_x_v, max_x_v = settings.start_range_voltage_x
        min_y_v, max_y_v = settings.start_range_voltage_y

        # Convert the voltage to coordinates
        min_x, min_y = self.voltage_to_coord(min_x_v, min_y_v)
        max_x, max_y = self.voltage_to_coord(max_x_v, max_y_v)

        # Make sure the patch is fully inside the starting range
        max_x = max_x - settings.patch_size_x
        max_y = max_y - settings.patch_size_y

        if min_x > max_x or min_y > max_y:
            raise ValueError(f'The starting range ({min_x_v}V, {min_y_v}V) to ({max_x_v}V, {max_y_v}V) '
                             f'is too small for the patch size: {settings.patch_size_x}×{settings.patch_size_y}.')

        # Generate random coordinates inside the range
        return randrange(min_x, max_x), randrange(min_y, max_y)

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int], normalized: bool = True) \
            -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixels)
        :param normalized: If True, the patch will be normalized between 0 and 1
        :return: The patch.
        """
        x_patch, y_patch = patch_size
        x_start, y_start = coordinate
        x_end = x_start + x_patch
        y_end = y_start + y_patch
        x_start_v, y_start_v = self.coord_to_voltage(x_start, y_start)
        x_end_v, y_end_v = self.coord_to_voltage(x_end, y_end)

        if settings.use_cached_measurement:
            # Try to optimize the patch measurement by using cached data
            measurements_coords = self._split_patch_in_unmeasured_rectangles((x_start, x_end, y_start, y_end))
            if len(measurements_coords) == 0:
                logger.debug(f'Using only cached data for the current patch, no new measurement required')
            elif len(measurements_coords) == 1 and (x_start, x_end, y_start, y_end) == measurements_coords[0]:
                logger.debug(f'No cached data found for the current patch, requesting full patch measurement')
            else:
                logger.debug(f'Partial cached data found for the current patch, requesting {len(measurements_coords)} '
                             f'measurement(s)')
        else:
            # Request a full patch measurement
            measurements_coords = [(x_start, x_end, y_start, y_end)]

        # Request the measurements to the connector (it could be 0, 1 or multiple measurements)
        for i, (sub_x_start, sub_x_end, sub_y_start, sub_y_end) in enumerate(measurements_coords, start=1):
            # Convert the coordinate to voltage
            sub_x_start_v, sub_y_start_v = self.coord_to_voltage(sub_x_start, sub_y_start)
            sub_x_end_v, sub_y_end_v = self.coord_to_voltage(sub_x_end, sub_y_end)
            # Request a new measurement to the connector
            nb_points = (sub_x_end - sub_x_start) * (sub_y_end - sub_y_start)
            logger.debug(f'Requesting measurement ({nb_points:,d} points) to the {self._connector} connector: '
                         f'|X|{sub_x_start}->{sub_x_end}| ({sub_x_start_v:.4f}V->{sub_x_end_v:.4f}V) '
                         f'|Y|{sub_y_start}->{sub_y_end}| ({sub_y_start_v:.4f}V->{sub_y_end_v:.4f}V)')
            measurement = self._connector.measurement(sub_x_start_v, sub_x_end_v, settings.pixel_size,
                                                      sub_y_start_v, sub_y_end_v, settings.pixel_size)
            measurement.note = f'Patch:' \
                               f'\n  -|X|{x_start}->{x_end}| ({x_start_v:.4f}V->{x_end_v:.4f}V)' \
                               f'\n  -|Y|{y_start}->{y_end}| ({y_start_v:.4f}V->{y_end_v:.4f}V)' \
                               f'\n\nSub-section: {i}/{len(measurements_coords)}' \
                               f'\n  -|X|{sub_x_start}->{sub_x_end}| ({sub_x_start_v:.4f}V->{sub_x_end_v:.4f}V)' \
                               f'\n  -|Y|{sub_y_start}->{sub_y_end}| ({sub_y_start_v:.4f}V->{sub_y_end_v:.4f}V)'

            # Save the measurement in the history to keep track of it
            self._measurement_history.append(measurement)
            # Send the data matrix to the same device as the values
            measurement.to(self.values.device)
            # Save the measurement in the grid
            self.values[sub_y_start:sub_y_end, sub_x_start: sub_x_end] = measurement.data

            # Plot the diagram with all current measurements
            if settings.is_named_run() and (settings.save_images or settings.show_images):
                self.plot()

        patch_data = self.values[y_start:y_end, x_start: x_end]
        if patch_data.isnan().any():
            raise ValueError(f'The patch ({x_start}, {y_start}) to ({x_end}, {y_end}) contains NaN values.')

        # Normalize the measurement with the normalization range used during the training, then return it.
        return self.normalize(patch_data.clone()) if normalized else patch_data

    def plot(self) -> None:
        """
        Plot or update the last image of the diagram.
        """
        values, x_axes, y_axes = self.get_values()
        focus_area = False
        text = None
        if self._measurement_history and len(self._measurement_history) > 0:
            last_m = self._measurement_history[-1]
            focus_area = (last_m.x_axes[0], last_m.x_axes[-1], last_m.y_axes[0], last_m.y_axes[-1])
            text = last_m.note

        plot_diagram(x_axes, y_axes, values, title=f'Online diagram {self.name}', focus_area_title='Last measurement',
                     allow_overwrite=True, focus_area=focus_area, file_name=f'diagram_{self.name}', scale_bars=True,
                     diagram_boundaries=self.get_cropped_boundaries(), text=text)

    def get_charge(self, coord_x: int, coord_y: int) -> ChargeRegime:
        """
        In the case of online diagram we cannot automatically know the charge regime.
        Therefore, we always return an UNKNOWN regime.

        :param coord_x: Doesn't matter.
        :param coord_y: Doesn't matter.
        :return: Always an UNKNOWN regime
        """
        return ChargeRegime.UNKNOWN

    def normalize(self, measurement: torch.Tensor) -> torch.Tensor:
        """
        Normalize the datasets in function of the method defined in the settings.

        :param measurement: Values to normalize.
        :return: The normalized values.
        """
        if settings.normalization == 'patch':
            # Normalize the measurement with the min/max values of the current patch
            min_value = measurement.min()
            max_value = measurement.max()
        elif settings.normalization == 'train-set':
            # Normalize the measurement with the min/max values of the train set
            min_value = self._norm_min_value
            max_value = self._norm_max_value
        else:
            # No normalization
            return measurement

        measurement -= min_value
        measurement /= max_value - min_value
        return measurement

    def get_cropped_boundaries(self) -> Tuple[float, float, float, float]:
        """
        Get the coordinates of the diagram that crop to the measured area.

        :return: Voltages of the cropped boundaries as: x_start, x_end, y_start, y_end.
        """
        values, x_axis, y_axis = self.get_values()
        # Crop the nan values from the image (not measured area)
        # Solution from: https://stackoverflow.com/a/25831190/2666094
        nans = torch.isnan(values)
        nan_cols = torch.all(nans, axis=0).int()  # True where col is all NAN
        nan_rows = torch.all(nans, axis=1).int()  # True where row is all NAN

        # The first index where not NAN
        first_col = nan_cols.argmin()
        first_row = nan_rows.argmin()

        # The last index where not NAN
        last_col = len(nan_cols) - nan_cols.flip(0).argmin()
        last_row = len(nan_rows) - nan_rows.flip(0).argmin()

        # Apply margins
        margin = settings.patch_size_x
        first_col = max(0, first_col - margin)
        first_row = max(0, first_row - margin)
        last_col = min(last_col + margin, len(x_axis) - 1)
        last_row = min(last_row + margin, len(y_axis) - 1)

        return x_axis[first_col], x_axis[last_col], y_axis[first_row], y_axis[last_row]

    def _split_patch_in_unmeasured_rectangles(self, patch_coords) -> List[Tuple[int, int, int, int]]:
        """
        Split a patch in multiple rectangles of unmeasured diagram area.

        :param patch_coords: The coordinates of the patch to measure.
        :return: The list of rectangles as: x_start, x_end, y_start, y_end.
        """
        # Get a mask of scanned values
        x_start, x_end, y_start, y_end = patch_coords
        scanned = self.values[y_start:y_end, x_start: x_end].isnan().logical_not()

        # All values have been scanned, return an empty list
        if scanned.all():
            return []

        # No cached data for this patch, return one rectangle that covers the whole patch
        if scanned.logical_not().all():
            return [(x_start, x_end, y_start, y_end)]

        # Patch partially scanned, split it in multiple rectangles that cover the unscanned areas
        to_scan_rectangles = []
        while not torch.all(scanned):
            # Get the first index where not scanned (False)
            # Indexes relative to the current patch
            first_y, first_x = tuple(scanned.logical_not().int().nonzero()[0].tolist())

            # Search for the last consecutive x index where not scanned
            last_x = first_x + 1
            for x in range(first_x + 1, len(scanned[first_y])):
                if scanned[first_y, x]:
                    break
                last_x = x + 1

            # Search for the last consecutive y index where everything is not scanned in the current x range
            last_y = first_y + 1
            for y in range(first_y + 1, len(scanned)):
                if scanned[y, first_x:last_x].any():
                    break
                last_y = y + 1

            # Set current rectangle as scanned
            scanned[first_y:last_y, first_x:last_x] = True
            # Add the rectangle to the list
            to_scan_rectangles.append((first_x + x_start, last_x + x_start, first_y + y_start, last_y + y_start))

        return to_scan_rectangles
