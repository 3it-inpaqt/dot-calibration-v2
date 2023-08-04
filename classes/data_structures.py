"""
Bunch of dataclasses and enumerations to structure information and simplify code.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Deque, Iterable, List, Optional, Sequence, Tuple

import torch

from utils.settings import settings
from utils.timer import duration_to_str


@unique
class ChargeRegime(Enum):
    """ Charge regime enumeration """
    UNKNOWN = 'unknown'
    ELECTRON_0 = '0_electron_1'
    ELECTRON_1 = '1_electron_1'
    ELECTRON_2 = '2_electrons_1'
    ELECTRON_3 = '3_electrons_1'
    ELECTRON_4_PLUS = '4+_electrons_1'

    def __str__(self) -> str:
        """
        Convert a charge regime to short string representation.

        :return: Short string name.
        """
        short_map = {ChargeRegime.UNKNOWN: 'unk.', ChargeRegime.ELECTRON_0: '0', ChargeRegime.ELECTRON_1: '1',
                     ChargeRegime.ELECTRON_2: '2', ChargeRegime.ELECTRON_3: '3', ChargeRegime.ELECTRON_4_PLUS: '4+'}
        return short_map[self]


@unique
class BoundaryPolicy(Enum):
    """ Enumeration of policies to apply if a scan is requested outside the diagram borders. """
    HARD = 0  # Don't allow going outside the diagram
    SOFT_RANDOM = 1  # Allow going outside the diagram and fill unknown data with random values
    SOFT_VOID = 2  # Allow going outside the diagram and fill unknown data with 0


@dataclass(frozen=True)
class ClassMetrics:
    """ Store classification result metrics for one class. """
    nb: int
    precision: float
    recall: float
    f1: float

    @property
    def main(self):
        return getattr(self, settings.main_metric)

    def __str__(self):
        return f'{settings.main_metric}: {self.main:.2%}'

    def __repr__(self):
        return f'nb: {self.nb} | precision: {self.precision:.2%} | recall: {self.recall:.2%} | f1: {self.f1:.2%}'


@dataclass(frozen=True)
class ClassificationMetrics(ClassMetrics):
    """ Store classification result metrics. """
    accuracy: float
    classes: List[ClassMetrics]

    def __iter__(self):
        return iter(self.classes)

    def __getitem__(self, i):
        return self.classes[i]

    def __repr__(self):
        return f'nb: {self.nb} | accuracy:  {self.accuracy} | precision: {self.precision:.2%} |' \
               f' recall: {self.recall:.2%} | f1: {self.f1:.2%}\n' + '\n\t- '.join([cls.__repr__() for cls in self])


@dataclass(frozen=True)
class StepHistoryEntry:
    coordinates: Tuple[int, int]
    model_classification: bool  # True = Line / False = No line
    model_confidence: float
    ground_truth: bool
    soft_truth_larger: bool  # Ground truth if the active area was larger (smaller offset)
    soft_truth_smaller: bool  # Ground truth if the active area was smaller (larger offset)
    is_above_confidence_threshold: bool
    description: str
    timestamp_start: float  # In seconds
    timestamp_data_fetched: float  # In seconds
    timestamp_data_processed: float  # In seconds
    is_online: bool

    def is_classification_correct(self) -> bool:
        """ :return: True only if model_classification is the same as the ground_truth. """
        return self.model_classification == self.ground_truth

    def is_classification_almost_correct(self) -> bool:
        """
        :return: True if model_classification is a line and a line is near the active area or model_classification is a
         line and all lines are almost outside the active area.
        """
        return (self.model_classification and self.soft_truth_larger) or \
            (not self.model_classification and not self.soft_truth_smaller)

    def get_area_coord(self) -> Tuple[int, int, int, int]:
        start_x, start_y = self.coordinates
        return start_x, start_x + settings.patch_size_x, start_y, start_y + settings.patch_size_y

    @staticmethod
    def get_text_description(scan_history: List["StepHistoryEntry"]) -> str:
        """
        Generate a text description of the given scan history.

        :param scan_history: The scan history to describe.
        :return: The text description.
        """
        text = ''
        if scan_history:
            nb_scan = len(scan_history)
            # Local import to avoid circular messes
            from datasets.qdsd import QDSDLines
            is_online = scan_history[0].is_online

            # History statistics
            line_success = accuracy = no_line_success = None
            if is_online:
                # Count line detected
                nb_line = sum(1 for s in scan_history if s.model_classification)  # Line detected
                nb_no_line = sum(1 for s in scan_history if not s.model_classification)  # No line detected
            else:
                # Count ground truth
                accuracy = sum(1 for s in scan_history if s.is_classification_correct()) / nb_scan
                nb_line = sum(1 for s in scan_history if s.ground_truth)  # s.ground_truth == True means line
                nb_no_line = sum(1 for s in scan_history if not s.ground_truth)  # s.ground_truth == False means no line

                if nb_line > 0:
                    line_success = sum(
                        1 for s in scan_history if s.ground_truth and s.is_classification_correct()) / nb_line

                if nb_no_line > 0:
                    no_line_success = sum(1 for s in scan_history
                                          if not s.ground_truth and s.is_classification_correct()) / nb_no_line

            # Time information
            time_start_tuning = scan_history[0].timestamp_start
            time_end_last_scan = scan_history[-1].timestamp_data_processed
            tuning_duration = duration_to_str(time_end_last_scan - time_start_tuning, nb_units_display=2)

            time_start_last_scan = scan_history[-1].timestamp_start
            time_end_data_fetched = scan_history[-1].timestamp_data_fetched
            fetch_duration = duration_to_str(time_end_data_fetched - time_start_last_scan, 1, 'us')
            inference_duration = duration_to_str(time_end_last_scan - time_end_data_fetched, 1, 'us')

            # Last classification information
            if scan_history[-1].is_classification_correct():
                class_error = 'good'
            elif scan_history[-1].is_classification_almost_correct():
                class_error = 'soft error'
            else:
                class_error = 'error'
            last_class = QDSDLines.classes[scan_history[-1].model_classification]

            text += f'Nb step: {nb_scan: >3n}'
            text += '\n' if is_online else f' (acc: {accuracy:>4.0%})\n'
            text += f'{QDSDLines.classes[True].capitalize(): <7}: {nb_line: >3n}'
            text += '\n' if line_success is None else f' (acc: {line_success:>4.0%})\n'
            text += f'{QDSDLines.classes[False].capitalize(): <7}: {nb_no_line: >3n}'
            text += '\n' if no_line_success is None else f' (acc: {no_line_success:>4.0%})\n\n'
            text += f'Last patch:\n'
            text += f'  - Pred: {last_class.capitalize(): <7}'
            text += '\n' if is_online else f' ({class_error})\n'
            text += f'  - Conf: {scan_history[-1].model_confidence: >4.0%}\n'
            text += f'  - Fetch data: {fetch_duration}\n'
            text += f'  - Inference : {inference_duration}\n\n'
            text += f'Tuning duration: {tuning_duration}\n\n'
            text += f'Tuning step:\n'
            text += f'  {scan_history[-1].description}'

        return text


@dataclass
class Direction:
    """ Data class to factorise code. """
    is_stuck: bool = False
    last_x: int = 0
    last_y: int = 0
    move: Callable = None
    check_stuck: Callable = None

    @staticmethod
    def all_stuck(directions: Iterable["Direction"]):
        return all(d.is_stuck for d in directions)


@dataclass(frozen=True)
class AutotuningResult:
    diagram_name: str
    procedure_name: str
    model_name: str
    nb_steps: int
    nb_classification_success: int
    charge_area: ChargeRegime
    final_coord: Tuple[int, int]
    final_volt_coord: Tuple[float, float]

    @property
    def is_success_tuning(self):
        return self.charge_area is ChargeRegime.ELECTRON_1

    @property
    def success_rate(self):
        return self.nb_classification_success / self.nb_steps if self.nb_steps > 0 else 0


@dataclass
class SearchLineSlope:
    scans_results: Deque[bool]
    scans_positions: Deque[Tuple[int, int]]

    def __init__(self):
        self.scans_results = deque()
        self.scans_positions = deque()

    def is_valid_sequence(self) -> bool:
        """
        The detection is valid only if we found the sequence: "No Line" * m --> "Line" * n --> "No Line" * p.
        Where n, m and p >= 1.
        :return: True if the scans sequence is valid.
        """
        seq = []
        for is_line in self.scans_results:
            if len(seq) == 0 or seq[-1] != is_line:
                seq.append(is_line)

        return seq == [False, True, False]

    def get_line_boundary(self, first: bool) -> Optional[Tuple[int, int]]:
        """
        Find line detection boundary in the scan list.

        :param first: If set to True, return the first line in the list. If False, return the last one.
        :return: The first or last line coordinates of the scan sequence. Or None if no line detected.
        """
        results_positions = list(zip(self.scans_results, self.scans_positions))
        if first:
            results_positions.reverse()
        for is_line, position in results_positions:
            if is_line:
                return position
        return None


@dataclass
class ExperimentalMeasurement:
    """ Data class to keep track of experimental measurement at each step. """
    x_axes: Sequence[float]
    y_axes: Sequence[float]
    data: torch.Tensor
    note: Optional[str] = None

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the data to a specific device (cpu or cuda) and/or a convert it to a different type. Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self.data = self.data.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
