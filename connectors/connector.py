from classes.data_structures import ExperimentalMeasurement
from datasets.diagram_online import DiagramOnline
from utils.settings import settings


class Connector:
    _is_connected: bool = False

    def _setup_connection(self) -> None:
        """
        Set up the connection to the experimental setup.
        """
        raise NotImplementedError

    def measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float,
                    start_volt_y: float, end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:
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
    def get_connector() -> "Connector":
        """
        Factory method to get a connector according to the settings.
        :return: The connector.
        """

        # Fake connector for testing
        if settings.use_mock:
            from connectors.mock import Mock
            return Mock()

        # Chose the corrector in function of the dataset
        if settings.research_group.startswith('eva_dupont_ferrier'):
            from connectors.py_hegel import PyHegel
            return PyHegel()
        else:
            raise ValueError(f'Connector not implemented for the dataset "{settings.research_group}".')
