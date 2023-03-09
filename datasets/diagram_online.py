from math import prod
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
    _origine_voltage: Tuple[float, float]
    _norm_min_value: float
    _norm_max_value: float

    def __init__(self, name: str, connector: "Connector"):
        super().__init__(name)
        self._connector = connector
        self._measurement_history = []

        # Arbitrary define the mapping between the voltage and the coordinate at the origin (0, 0)
        self._origine_voltage = settings.min_voltage

        # Fetch the normalization values used during the training
        self._norm_min_value, self._norm_max_value = load_normalization()

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixel)
        :return: The patch.
        """

        # Convert the coordinate to voltage
        x, y = coordinate
        x_min, y_min = settings.min_voltage
        x_max, y_max = settings.max_voltage
        x_patch, y_patch = patch_size
        x_range_coord = [x, x + x_patch]
        y_range_coord = [y, y + y_patch]
        x_range_volt = Diagram._coord_to_volt(x_range_coord, x_min, x_max, settings.pixel_size)
        y_range_volt = Diagram._coord_to_volt(y_range_coord, y_min, y_max, settings.pixel_size)

        # Request a new measurement to the connector
        logger.info(f'Requesting {prod(patch_size)} measurements to the {self._connector} connector: '
                    f'X: {x_range_coord[0]}-{x_range_coord[1]} ({x_range_volt[0]}V-{x_range_volt[1]}V), '
                    f'Y: {y_range_coord[0]}-{y_range_coord[1]} ({y_range_volt[0]}V-{y_range_volt[1]}V)')

        with SectionTimer('experimental measurement', 'debug'):
            measurement = self._connector.measurement()

        # Validate the measurement size
        if list(measurement.values.shape) != patch_size:
            raise ValueError(f'Unexpected measurement size: {list(measurement.values.shape)} while {patch_size} has'
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
