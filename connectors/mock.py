from time import sleep

import numpy as np
import torch

from classes.data_structures import ExperimentalMeasurement
from connectors.connector import Connector
from utils.logger import logger
from utils.settings import settings


class Mock(Connector):
    """
    Fake connector for testing purposes.
    Will generate random values for the measurement.
    """

    def _setup_connection(self) -> None:
        self._is_connected = True
        logger.info('Mock connector ready.')

    def _close_connection(self) -> None:
        self._is_connected = False
        logger.info('Mock connector closed.')

    def _measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float, start_volt_y: float,
                     end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:
        """
        Make a fake measurement. Output random values.

        :param start_volt_x: Only used to compute the size of the output tensor.
        :param end_volt_x: Only used to compute the size of the output tensor.
        :param step_volt_x: Only used to compute the size of the output tensor.
        :param start_volt_y: Only used to compute the size of the output tensor.
        :param end_volt_y: Only used to compute the size of the output tensor.
        :param step_volt_y: Only used to compute the size of the output tensor.
        :return: An experimental measurement with random values.
        """
        # Interaction mode is used to control the speed of the mock measurement generation ("auto" goes full speed)
        mode = settings.interaction_mode.lower().strip()
        if mode == 'manual':
            input(f'Mock measurement #{self._nb_measurement:03,d}. Press enter to continue.')
        elif mode == 'semi-auto':
            sleep(2)  # Wait 2 seconds

        size_x = round((end_volt_x - start_volt_x) / step_volt_x)
        size_y = round((end_volt_y - start_volt_y) / step_volt_y)
        # Last point excluded
        return ExperimentalMeasurement(
            np.linspace(start_volt_x, end_volt_x, size_x, endpoint=False),
            np.linspace(start_volt_y, end_volt_y, size_y, endpoint=False),
            torch.rand((size_y, size_x)), None)
