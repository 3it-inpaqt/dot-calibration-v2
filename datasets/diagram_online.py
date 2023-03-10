from math import prod
from random import randrange
from typing import List, Optional, Tuple

import torch

from classes.data_structures import ChargeRegime, ExperimentalMeasurement
from datasets.diagram import Diagram
from utils.logger import logger
from utils.output import load_normalization
from utils.settings import settings
from utils.timer import SectionTimer


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
        min_x, min_y = self._voltage_to_coord(min_x_v, min_y_v)
        max_x, max_y = self._voltage_to_coord(max_x_v, max_y_v)

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

        # Convert the coordinate to voltage
        x, y = coordinate
        min_x_v, min_y_v = settings.min_voltage, settings.min_voltage
        max_x_v, max_y_v = settings.max_voltage, settings.max_voltage
        x_patch, y_patch = patch_size
        x_range_coord = [x, x + x_patch]
        y_range_coord = [y, y + y_patch]
        x_range_volt = Diagram._coord_to_volt(x_range_coord, min_x_v, max_x_v, settings.pixel_size)
        y_range_volt = Diagram._coord_to_volt(y_range_coord, min_y_v, max_y_v, settings.pixel_size)

        # Request a new measurement to the connector
        logger.debug(f'Requesting measurement ({prod(patch_size):,d} points) to the {self._connector} connector: '
                     f'|X|{x_range_coord[0]}→{x_range_coord[1]}| ({x_range_volt[0]:.3f}V→{x_range_volt[1]:.3f}V) '
                     f'|Y|{y_range_coord[0]}→{y_range_coord[1]}| ({y_range_volt[0]:.3f}V→{y_range_volt[1]:.3f}V)')

        with SectionTimer('experimental measurement', 'debug'):
            measurement = self._connector.measurement(x_range_volt[0], x_range_volt[1], settings.pixel_size,
                                                      y_range_volt[0], y_range_volt[1], settings.pixel_size)

        # Validate the measurement size
        if tuple(measurement.values.shape) != patch_size:
            raise ValueError(f'Unexpected measurement size: {tuple(measurement.values.shape)} while {patch_size} has'
                             f' been requested.')

        # Save the measurement in the history to keep track of it
        self._measurement_history.append(measurement)
        # Send the values matrix to the appropriate device (cpu or gpu)
        measurement.values.to(self._torch_device)

        # Normalize the measurement with the normalization range used during the training, then return it.
        return self.normalize(measurement.values)

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        pass

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

    def _voltage_to_coord(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert a voltage to a coordinate in the diagram relatively to the origin chosen.

        :param x: The voltage (x axes) to convert.
        :param y: The voltage (y axes) to convert.
        :return: The coordinate (x, y) in the diagram.
        """
        origin_x, origin_y = self._origin_voltage
        return round(((x - origin_x) / settings.pixel_size)), round(((y - origin_y) / settings.pixel_size))

    def get_max_patch_coordinates(self) -> Tuple[int, int]:
        """
        Get the maximum coordinates of a patch in this diagram.

        :return: The maximum coordinates as (x, y)
        """
        return self._voltage_to_coord(settings.max_voltage, settings.max_voltage)
