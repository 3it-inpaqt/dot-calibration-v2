from math import prod
from random import randrange
from typing import List, Optional, Sequence, Tuple

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
    _norm_min_value: float
    _norm_max_value: float

    def __init__(self, name: str, connector: "Connector"):
        """
        Create an instance of a DiagramOnline associated with a connector (interface to the measurement tool).

        :param name: The name of the diagram.
        :param connector: The connector to the measurement tool.
        """
        super().__init__(name)
        self._connector = connector
        self._measurement_history = []

        # Fetch the normalization values used during the training
        self._norm_min_value, self._norm_max_value = load_normalization()

        # Create a virtual axes and discret grid that represent the voltage space to explore.
        # Where NaN values represent the voltage that have not been measured yet.
        # The min value is included but not the max value (to match with python standard).
        space_size = int((settings.max_voltage - settings.min_voltage) / settings.pixel_size)  # Assume square space
        self.x_axes = np.linspace(settings.min_voltage, settings.max_voltage, space_size, endpoint=False)
        self.y_axes = np.linspace(settings.min_voltage, settings.max_voltage, space_size, endpoint=False)
        self.values = torch.full((space_size, space_size), torch.nan)

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

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixel)
        :return: The patch.
        """
        x_patch, y_patch = patch_size
        x_start, y_start = coordinate
        x_end = x_start + x_patch
        y_end = y_start + y_patch

        # Convert the coordinate to voltage
        x_start_v, y_start_v = self.coord_to_voltage(x_start, y_start)
        x_end_v, y_end_v = self.coord_to_voltage(x_end, y_end)

        # Request a new measurement to the connector
        logger.debug(f'Requesting measurement ({prod(patch_size):,d} points) to the {self._connector} connector: '
                     f'|X|{x_start}->{x_end}| ({x_start_v:.4f}V->{x_end_v:.4f}V) '
                     f'|Y|{y_start}->{y_end}| ({y_start_v:.4f}V->{y_end_v:.4f}V)')
        measurement = self._connector.measurement(x_start_v, x_end_v, settings.pixel_size,
                                                  y_start_v, y_end_v, settings.pixel_size)

        # Validate the measurement size
        if tuple(measurement.data.shape) != patch_size:
            raise ValueError(f'Unexpected measurement size: {tuple(measurement.data.shape)} while {patch_size} has'
                             f' been requested.')

        # Flip y-axis because we save the diagram values with origin is in the top left corner
        measurement.data = measurement.data.flip(0)
        # Save the measurement in the history to keep track of it
        self._measurement_history.append(measurement)
        # Send the data matrix to the same device as the values
        measurement.to(self.values.device)
        # Save the measurement in the grid (invert y-axis because the diagram origin is in the top left corner)
        diagram_size_y, _ = self.values.shape
        self.values[diagram_size_y - y_end:diagram_size_y - y_start, x_start: x_end] = measurement.data

        # Plot the diagram with all current measurements
        # if settings.is_named_run() and (settings.save_images or settings.show_images):
        #     self.plot()

        # Normalize the measurement with the normalization range used during the training, then return it.
        return self.normalize(measurement.data)

    def plot(self, label_extra: Optional[str] = '') -> None:
        values, x_axes, y_axes = self.get_cropped_values()
        plot_diagram(x_axes, y_axes, values, 'Online intermediate step' + label_extra, 'None', settings.pixel_size,
                     allow_overwrite=True, show_offset=False)

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
        Normalize some data with the same min/max value used during the training.
        The normalisation values have been fetch via the normalization_values_path setting at the initialisation.

        :param measurement: The values to normalize.
        :return: The normalized values.
        """
        measurement -= self._norm_min_value
        measurement /= self._norm_max_value - self._norm_min_value
        return measurement

    def get_cropped_values(self) -> Tuple[Optional[torch.Tensor], Sequence[float], Sequence[float]]:
        """
        Get the values of the diagram, cropped to the measured area.

        :return: The values as a tensor, the list of x-axis values, the list of y-axis values.
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
        last_col = min(last_col + margin, len(x_axis))
        last_row = min(last_row + margin, len(y_axis))

        return values[first_row:last_row, first_col:last_col], x_axis[first_col:last_col], y_axis[first_row:last_row]
