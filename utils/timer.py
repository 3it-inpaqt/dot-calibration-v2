import logging
import time
from typing import Union

from codetiming import Timer, TimerError

from utils.logger import logger
from utils.str_formatting import duration_to_str


class SectionTimer(Timer):
    def __init__(self, section_name: str, log_level: Union[str, int] = 'info'):
        """
        Create a time that log the start and the end of a code section.

        :param section_name: The name of the section, for the log and the label in output file
        :param log_level: The log level to use for the start and end messages
        """
        super().__init__(name=section_name)

        # Add a started boolean to handle the pause an resume
        self.started = False

        # Disable default print
        self.logger = None

        # Check log level
        if isinstance(log_level, str):
            log_level = log_level.upper()
        # Use logging private method to check and convert the log level value
        self.log_level = logging._checkLevel(log_level)

    def start(self) -> None:
        """
        Override the start function of the parent object.
        """
        logger.log(self.log_level, f'Start {self.name}...')
        self.started = True
        super().start()

    def stop(self) -> float:
        """
        Override the stop function of the parent object.

        :return: The time interval since it was started, in seconds
        """
        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        self.last = time.perf_counter() - self._start_time
        self._start_time = None

        # Report elapsed time
        if self.log_level:
            logger.log(self.log_level, f'Completed {self.name} in {duration_to_str(self.last)}')
        if self.name:
            self.timers.add(self.name, self.last)

        return self.last

    def pause(self) -> None:
        """
        Stop the timer with no end log message.
        """
        super().stop()

    def resume(self) -> None:
        """
        Restart the timer with no start log message.
        """
        if not self.started:
            TimerError("A timer should be started before to be resumed")
        super().start()
