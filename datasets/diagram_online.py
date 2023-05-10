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
    _torch_device: torch.device = None
    _measurement_history: List[ExperimentalMeasurement]
    _origin_voltage: Tuple[float, float]
    _norm_min_value: float
    _norm_max_value: float

    def __init__(self, name: str, connector: "Connector"):
        super().__init__(name)
        self._connector = connector
        self._measurement_history = []

        # Arbitrary define the mapping between the voltage and the coordinate at the origin (0, 0)
        self._origin_voltage = settings.min_voltage, settings.min_voltage

        # Fetch the normalization values used during the training
        self._norm_min_value, self._norm_max_value = load_normalization()

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
                     f'|X|{x_start}->{x_end}| ({x_start_v:.3f}V->{x_end_v:.3f}V) '
                     f'|Y|{y_start}->{y_end}| ({y_start_v:.3f}V->{y_end_v:.3f}V)')
        measurement = self._connector.measurement(x_start_v, x_end_v, settings.pixel_size,
                                                  y_start_v, y_end_v, settings.pixel_size)

        # Validate the measurement size
        if tuple(measurement.data.shape) != patch_size:
            raise ValueError(f'Unexpected measurement size: {tuple(measurement.data.shape)} while {patch_size} has'
                             f' been requested.')

        # Save the measurement in the history to keep track of it
        self._measurement_history.append(measurement)
        # Send the data matrix to the appropriate device (cpu or gpu)
        measurement.data.to(self._torch_device)

        # Plot the diagram with all current measurements
        if settings.is_named_run() and (settings.save_images or settings.show_images):
            self.plot()

        # Normalize the measurement with the normalization range used during the training, then return it.
        return self.normalize(measurement.data)

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        values, x_axis, y_axis = self.get_values()
        if values is not None:
            plot_diagram(x_axis, y_axis, values, 'Online intermediate step' + label_extra, 'None', settings.pixel_size,
                         focus_area=focus_area, allow_overwrite=True)

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Save the torch device to use. Every new data will be sent to this device after being fetched from the connector.

        :param device: A valid torch device.
        :param dtype: Not used for online diagram.
        :param non_blocking: Not used for online diagram.
        :param copy: Not used for online diagram.
        """
        self._torch_device = device

    def get_values(self) -> Tuple[Optional[torch.Tensor], Sequence[float], Sequence[float]]:
        """
        Get all measured values of the diagram and the corresponding axis.

        :return: The values as a tensor, the list of x-axis values, the list of y-axis values
        """
        if len(self._measurement_history) == 0:
            return None, [], []

        space_size = int((settings.max_voltage - settings.min_voltage) / settings.pixel_size)  # Assume square space
        values = torch.full((space_size, space_size), torch.nan)
        x_axis = np.linspace(settings.min_voltage, settings.max_voltage, space_size)
        y_axis = np.linspace(settings.min_voltage, settings.max_voltage, space_size)

        # Fill the spaces with the measurements
        first_col, last_col, first_row, last_row = None, None, None, None
        for measurement in self._measurement_history:
            start_x, end_x = measurement.x_axes[0], measurement.x_axes[-1]
            start_y, end_y = measurement.y_axes[0], measurement.y_axes[-1]
            start_x, start_y = self.voltage_to_coord(start_x, start_y)
            end_x, end_y = self.voltage_to_coord(end_x, end_y)

            values[start_x: end_x, start_y: end_y] = measurement.data

            # Keep track of data border to crop later
            if first_col is None or start_x < first_col:
                first_col = start_x
            if last_col is None or end_x > last_col:
                last_col = end_x
            if first_row is None or start_y < first_row:
                first_row = start_y
            if last_row is None or end_y > last_row:
                last_row = end_y

        # Apply margins
        margin = settings.patch_size_x
        first_col = max(0, first_col - margin)
        first_row = max(0, first_row - margin)
        last_col = min(last_col + margin, len(x_axis))
        last_row = min(last_row + margin, len(y_axis))

        # TODO crop the data to remove the NaN values
        # return values[first_row:last_row,first_col:last_col], x_axis[first_col:last_col], y_axis[first_row:last_row]
        return values, x_axis, y_axis

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

    def voltage_to_coord(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert a voltage to a coordinate in the diagram relatively to the origin chosen.

        :param x: The voltage (x axes) to convert.
        :param y: The voltage (y axes) to convert.
        :return: The coordinate (x, y) in the diagram.
        """
        origin_x, origin_y = self._origin_voltage
        return round(((x - origin_x) / settings.pixel_size)), round(((y - origin_y) / settings.pixel_size))

    def coord_to_voltage(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert a coordinate in the diagram to a voltage.

        :param x: The coordinate (x axes) to convert.
        :param y: The coordinate (y axes) to convert.
        :return: The voltage (x, y) in this diagram.
        """
        min_x_v, min_y_v = self._origin_voltage
        x_volt = min_x_v + x * settings.pixel_size
        y_volt = min_y_v + y * settings.pixel_size
        return x_volt, y_volt

    def get_max_patch_coordinates(self) -> Tuple[int, int]:
        """
        Get the maximum coordinates of a patch in this diagram.

        :return: The maximum coordinates as (x, y)
        """
        x_max, y_max = self.voltage_to_coord(settings.max_voltage, settings.max_voltage)
        return x_max - settings.patch_size_x, y_max - settings.patch_size_y
