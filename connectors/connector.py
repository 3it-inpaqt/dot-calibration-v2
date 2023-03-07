from datasets.diagram_online import DiagramOnline
from utils.settings import settings


class Connector:

    def setup_connection(self) -> None:
        """
        Set up the connection to the experimental setup.
        """
        raise NotImplementedError

    def get_diagram(self) -> "DiagramOnline":
        self.setup_connection()
        return DiagramOnline(self)

    @staticmethod
    def get_connector() -> "Connector":
        """
        Get the connector according to the settings.
        :return: The connector.
        """
        if settings.research_group.startswith('eva_dupont_ferrier'):
            from connectors.py_hegel import PyHegel
            return PyHegel()
        else:
            raise ValueError(f'Connector not implemented for the dataset "{settings.research_group}".')
