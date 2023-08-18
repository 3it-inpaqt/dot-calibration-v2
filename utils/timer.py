import logging
import time
from typing import Dict, Union

from codetiming import Timer, TimerError

from utils.logger import logger


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


def duration_to_str(sec: float, nb_units_display: int = 2, precision: str = 'ms'):
    """
    Transform a duration (in sec) into a human readable string.
    d: day, h: hour, m: minute, s: second, ms: millisecond, us or μs: microsecond

    :param sec: The duration in second. Decimals are milliseconds.
    :param nb_units_display: The maximum number of units we want to show. If 0 print all units.
    :param precision: The smallest unit we want to show.
    :return: A human-readable representation of the duration.
    """

    assert sec >= 0, f'Negative duration not supported ({sec})'
    precision = precision.strip().lower().replace('us', 'μs')

    periods = {
        'd': 1_000 * 1_000 * 60 * 60 * 24,
        'h': 1_000 * 1_000 * 60 * 60,
        'm': 1_000 * 1_000 * 60,
        's': 1_000 * 1_000,
        'ms': 1_000,
        'μs': 1
    }

    assert precision in periods, f'Precision should be a valid unit: {", ".join(periods.keys())}'

    # Null duration
    if sec == 0:
        return '0' + precision

    # Infinite duration
    if sec == float('inf'):
        return 'infinity'

    # Convert second to microsecond and round it to the nearest microsecond
    micro_sec = round(sec * 1_000_000)

    values_per_period: Dict[str][int] = {}

    for period_name, period_micro_s in periods.items():
        # End condition: we reach the last unit or the number of units to display
        nb = len(values_per_period)
        is_last_unit = period_name == precision or (nb > 0 and nb + 1 == nb_units_display)
        if is_last_unit:
            # We round the last unit
            period_value = round(micro_sec / period_micro_s)
        else:
            # We extract the value of the current unit, and we keep the rest for the next unit
            period_value, micro_sec = divmod(micro_sec, period_micro_s)

        if period_value > 0 or nb > 0:
            # Add a 0 only if we already have a value in larger unit
            values_per_period[period_name] = period_value
        if is_last_unit:
            break

    # In the case of a duration lower than the precision unit
    if len(values_per_period) == 0:
        return '<1' + precision

    return ' '.join(f'{value}{period_name}' for period_name, value in values_per_period.items())
