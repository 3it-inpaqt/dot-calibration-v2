import torch

from classes.data_structures import ExperimentalMeasurement
from connectors.connector import Connector
from utils.logger import logger


class Mock(Connector):
    def _setup_connection(self) -> None:
        self._is_connected = True
        logger.info('Mock connector ready.')

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
        size_x = round((end_volt_x - start_volt_x) / step_volt_x)
        size_y = round((end_volt_y - start_volt_y) / step_volt_y)
        return ExperimentalMeasurement(start_volt_x, start_volt_y, torch.rand((size_x, size_y)))
