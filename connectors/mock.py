from connectors.connector import Connector
from utils.logger import logger


class Mock(Connector):
    def _setup_connection(self) -> None:
        logger.info('Mock connector connected.')
        self._is_connected = True
