import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from utils.timer import duration_to_str


@dataclass
class ProgressBarMetrics:
    name: str
    enable_color: bool = True
    last_value: Optional[float] = None
    last_printed_value: Optional[float] = None
    print_value: Callable[[Optional[float]], str] = lambda x: f'{x:7.5f}'
    evolution_indicator: bool = True
    more_is_good: bool = True
    _is_printed: bool = False

    def update(self, value: float) -> None:
        """
        Update the value and keep track of the previous one if it was already printed.
        :param value: The new value for this metric.
        """
        if value is not None:
            # Keep the last value until then so we can always show the good indicator
            if self._is_printed:
                self.last_printed_value = self.last_value
            self._is_printed = False
            self.last_value = value

    def printed(self) -> None:
        """ Register than the current value was printed (useful for indicators) """
        self._is_printed = True

    def evolution_indicator_str(self) -> str:
        """
        Return a colored character depending of the direction of the value.
        """
        good_color = '\033[0;92m'  # Green text
        bad_color = '\033[0;91m'  # Red text
        no_evolution_color = '\033[0;33m'  # Orange text
        reset_color = '\033[0m'

        # Case with None previous value
        if self.last_printed_value is None:
            return ' '

        value_diff = self.last_value - self.last_printed_value

        if value_diff > 0:
            if self.enable_color:
                return f'{good_color if self.more_is_good else bad_color}▲{reset_color}'
            else:
                return '▲'
        elif value_diff < 0:
            if self.enable_color:
                return f'{bad_color if self.more_is_good else good_color}▼{reset_color}'
            else:
                return '▼'
        else:
            if self.enable_color:
                return f'{no_evolution_color}={reset_color}'
            else:
                return '='

    def __str__(self) -> str:
        # Case with None value
        if self.last_value is None:
            return f'{self.name}: None   '

        indicator = self.evolution_indicator_str() if self.evolution_indicator else ' '

        return f'{self.name}:{indicator}' + self.print_value(self.last_value)


class ProgressBar:
    def __init__(self, tasks_size: int, nb_subtasks: int = 1, task_name: str = 'progress', subtask_name: str = '',
                 metrics: Iterable[ProgressBarMetrics] = tuple(), bar_length: int = 60, subtask_char: str = '⎼',
                 fill_char: str = ' ', refresh_time: float = 0.5, auto_display: bool = True, enable_color: bool = None):
        """
        Create a progress bar to visually tracking progress from task, subtasks and metrics.

        :param tasks_size: The number of iterations before to reach the end of the task, or the end of a subtask if
        there is several of them.
        :param nb_subtasks: The number of subtasks, each subtask will have the same size (tasks_size). If 1 there is no
        subtasks.
        :param task_name: The name of the global task.
        :param subtask_name: The name of the subtasks.
        :param metrics: A list of metrics to track (they can be updated during the task).
        :param bar_length: The size of the visual progress bar (number of characters).
        :param subtask_char: The character used for subtask progress done.
        :param fill_char: The character used for subtask progress pending.
        :param refresh_time: The minimal time delta (in seconds) between two auto print, 0 to see all auto print.
        :param auto_display: If true the bar will be automatically printed at the start, the end and after every value.
        :param enable_color: If not None, will enable or disable color for all metrics. If None, keep default value of
        each metrics.
        update if the minimal refresh time allow it.
        """
        self.tasks_size = tasks_size
        self.task_progress = 0

        self.nb_subtasks = nb_subtasks
        self.current_subtask = 0

        self.task_name = task_name
        self.subtask_name = subtask_name
        self.bar_length = bar_length
        self.task_char = subtask_char
        self.fill_char = fill_char

        self._start_time = None
        self._end_time = None
        self._last_print = None
        self._refresh_time = refresh_time
        self._auto_display = auto_display

        # If enable color set, override it for all metrics
        if enable_color is not None:
            for metric in metrics:
                metric.enable_color = enable_color
        self.metrics = {metric.name: metric for metric in metrics}

    def update(self, **metrics: Any):
        """
        Update one or several metric values.
        Every metric should have been define at the bar initialisation.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        """
        for metric, value in metrics.items():
            assert metric in self.metrics, f'Unknown metric "{metric}", impossible to update'
            self.metrics[metric].update(value)

        if self._auto_display:
            self.lazy_print()

    def incr(self, **metrics: Any) -> None:
        """
        Increase the progression of the current subtask and update one or several metric values.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        """
        self.task_progress += 1
        self.update(**metrics)
        if self._auto_display:
            self.lazy_print()

    def next_subtask(self) -> None:
        """
        Increase by one the index of the current subtask.
        Print the bar if auto display is enable and the minimal refresh time allow it.
        """
        self.current_subtask += 1
        if self._auto_display:
            self.lazy_print()

    def get_eta(self) -> float:
        """
        Get the "estimated time of arrival" to reach 100% of the global task.
        :return: The estimated value in second.
        """
        delta_t = time.perf_counter() - self._start_time
        progress = self.get_task_progress() * 100
        # Deal with not started task
        if delta_t == 0 or progress == 0:
            return float('inf')
        # Max is use to avoid negative value at the end of the task
        return max(((100 * delta_t) / progress) - delta_t, 0)

    def get_task_progress(self) -> float:
        """
        :return: The completed percentage of this task.
        """
        return self.task_progress / (self.nb_subtasks * self.tasks_size)

    def get_subtask_progress(self) -> float:
        """
        :return: The completed percentage of the current subtask.
        """
        # Force 100% for the end of each sub tasks (if not, modulo will give 0)
        if self.current_subtask != 0 and self.task_progress == self.tasks_size * self.current_subtask:
            return 1.0
        return (self.task_progress % self.tasks_size) / self.tasks_size

    def start(self):
        """
        Start timer and display the bar if auto display enable.
        """
        assert self._start_time is None, 'The progress bar can\'t be started twice.'
        self._start_time = time.perf_counter()
        if self._auto_display:
            self.print()

    def __enter__(self) -> "ProgressBar":
        """
        Start timer and display the bar if auto display enable.
        :return: The current progress bar object.
        """
        self.start()
        return self

    def stop(self):
        """
        Save end time and display the bar if auto display enable.
        """
        assert self._end_time is None, 'The progress bar can\'t be stopped twice.'
        self._end_time = time.perf_counter()
        if self._auto_display:
            self.print()
            print()  # New line at the end

    def __exit__(self, *exc_info: Any) -> None:
        """
        Final display of the bar if auto display enable.
        """
        self.stop()

    def print(self) -> None:
        """ Force print the progression. """
        print(f'{self}', end='\r', flush=True)
        self._last_print = time.perf_counter()
        for metric in self.metrics.values():
            metric.printed()

    def lazy_print(self) -> None:
        """ Print the bar if the minimal refresh time allow it. """
        if self._last_print is None or self._refresh_time == 0 or \
                (time.perf_counter() - self._last_print) >= self._refresh_time:
            self.print()

    def __str__(self) -> str:
        """
        :return: The progress bar formatted as a string.
        """
        if self.nb_subtasks > 1:
            # Subtask progress
            subtask_progress = self.get_subtask_progress()
            subtask_formatted_name = self.subtask_name + ' ' if self.subtask_name else ''
            nb_fill_char = int(subtask_progress * self.bar_length)
            bar = self.task_char * nb_fill_char + self.fill_char * (self.bar_length - nb_fill_char)
        else:
            bar = self.fill_char * self.bar_length

        # Global task
        task_progress = self.get_task_progress()
        nb_task_char = int(task_progress * self.bar_length)
        # Gray background for task loading
        bar = '\033[0;100m' + bar[:nb_task_char] + '\033[0m' + bar[nb_task_char:]

        # Progress bar
        string = f'{self.task_name}⎹ {task_progress:7.2%}⎹{bar}⎸'

        # Multiple subtask information
        if self.nb_subtasks > 1:
            string += f'{subtask_formatted_name}{self.current_subtask}/{self.nb_subtasks} {subtask_progress:4.0%}⎹ '

        # Metrics information
        if len(self.metrics) > 0:
            string += '⎹ '.join(map(str, self.metrics.values())) + '⎹ '

        # Time information
        if self._end_time is None:
            # Still in progress
            eta = duration_to_str(self.get_eta(), precision='s')
            string += f'ETA: {eta:<10}'  # Fill the end with space to override previous characters
        else:
            # Task ended or interrupted
            duration = duration_to_str(self._end_time - self._start_time, precision='s')
            string += f'{duration:<15}'  # Fill the end with space to override previous characters

        return string
