from classes.data_structures import ExperimentalMeasurement
from datasets.diagram_online import DiagramOnline
from plots.data import plot_diagram
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


class Connector:
    _is_connected: bool = False
    _nb_measurement: int = 0

    def _setup_connection(self) -> None:
        """
        Set up the connection to the experimental setup.
        """
        raise NotImplementedError

    def measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float,
                    start_volt_y: float, end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:
        """
        Request an experimental measurement to the connector.

        :param start_volt_x: The beginning of the voltage sweep in the x-axis.
        :param end_volt_x: The end of the voltage sweep in the x-axis.
        :param step_volt_x: The step size of the voltage sweep in the x-axis.
        :param start_volt_y: The beginning of the voltage sweep in the y-axis.
        :param end_volt_y: The end of the voltage sweep in the y-axis.
        :param step_volt_y: The step size of the voltage sweep in the y-axis.
        :return: The experimental measurement.
        """

        # Check the voltage (RuntimeError raised if not valid)
        self._is_valid_voltage(start_volt_x, start_volt_y)
        self._is_valid_voltage(end_volt_x, end_volt_y)

        self._nb_measurement += 1

        with SectionTimer('measurement', 'debug'):
            result = self._measurement(start_volt_x, end_volt_x, step_volt_x, start_volt_y, end_volt_y, step_volt_y)

        if settings.plot_measurements:
            plot_diagram(result.x_axes, result.y_axes, result.data, f'measurement_{self._nb_measurement:03}', None,
                         step_volt_x, scale_bar=True)

        return result

    def _measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float,
                     start_volt_y: float, end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:
        """
        Request an experimental measurement to the connector.
        The internal logic of the measurement has to be overwritten by each connector implementation.

        :param start_volt_x: The beginning of the voltage sweep in the x-axis.
        :param end_volt_x: The end of the voltage sweep in the x-axis.
        :param step_volt_x: The step size of the voltage sweep in the x-axis.
        :param start_volt_y: The beginning of the voltage sweep in the y-axis.
        :param end_volt_y: The end of the voltage sweep in the y-axis.
        :param step_volt_y: The step size of the voltage sweep in the y-axis.
        :return: The experimental measurement.
        """
        raise NotImplementedError

    def get_diagram(self) -> "DiagramOnline":
        """
        Connect to the experimental setup if not already done and return an online diagram linked to this connector.

        :return: An online diagram linked to this connector
        """
        if not self._is_connected:
            self._setup_connection()
        return DiagramOnline('online_diagram', self)

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def _is_valid_voltage(x_v: float, y_v: float) -> bool:
        """
        Check if the given voltage is valid for the experimental setup according to the settings.

        :param x_v: The voltage in the x-axis.
        :param y_v: The voltage in the y-axis.
        :return: True if the voltage is valid, a RuntimeError is raised otherwise.
        """
        if x_v < settings.min_voltage or x_v > settings.max_voltage:
            raise RuntimeError(f'Voltage {x_v:.4f}V is not valid in the x-axis. '
                               f'Not in range [{settings.min_voltage:.4f}V, {settings.max_voltage:.4f}V].')
        if y_v < settings.min_voltage or y_v > settings.max_voltage:
            raise RuntimeError(f'Voltage {x_v:.4f}V is not valid in the x-axis. '
                               f'Not in range [{settings.min_voltage:.4f}V, {settings.max_voltage:.4f}V].')
        return True

    @staticmethod
    def get_connector() -> "Connector":
        """
        Factory method to get a connector according to the settings.
        :return: The connector.
        """

        # Chose the corrector in function of the settings
        if settings.connector_name == 'mock':
            logger.warning('Using the mock connector, every measurement will output random values.')
            from connectors.mock import Mock
            return Mock()

        mode = settings.interaction_mode.lower().strip()
        if mode == 'manual':
            logger.warning('Manual mode is activated. The online tuning task will not be able to run automatically.')
        elif mode == 'semi-auto':
            logger.info('Semi-auto mode is activated. The user will need to validate every command.')
        elif mode == 'auto':
            logger.warning('Auto mode is activated. The online tuning task will run automatically.')

        if settings.connector_name == 'py_hegel':
            from connectors.py_hegel import PyHegel
            return PyHegel()

        raise NotImplementedError(f'Connector not implemented: "{settings.connector_name}".')
