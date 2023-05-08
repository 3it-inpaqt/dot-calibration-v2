"""
Bunch of dataclasses and enumerations to structure information and simplify code.
"""
from collections import deque
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Deque, Iterable, List, Optional, Sequence, Tuple

import torch

from utils.settings import settings

# Enumerate all regime for each quantum dot
# for example ChargeRegime = {'0_electron_1': '0_1', '1_electron_1': '1_1', '2_electron_1': '2_1',
# '3_electron_1': '3_1', '4+_electron_1': '4_1', 'UNKNOWN': 'unknown'} for a single dot
regimes = ["0", "1", "2", "3", "4+"]
ChargeRegime = {'UNKNOWN': 'unknown'}
for dot in range(1, settings.dot_number + 1):
    count = 0
    for regime in regimes:
        if count <= 1:
            ChargeRegime[f'{regime}_electron_{dot}'] = f'{regime}_{dot}'
        else:
            ChargeRegime[f'{regime}_electrons_{dot}'] = f'{regime}_{dot}'
        count += 1


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
    description: str

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

    def is_under_confidence_threshold(self, confidence_thresholds: List[float]) -> bool:
        """
        :return: True if the confidence threshold is defined and the model classification confidence is under the
         threshold.
         """

        if confidence_thresholds:
            return self.model_confidence < confidence_thresholds[self.model_classification]

        return False  # No confidence thresholds defined


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

    @property
    def is_success_tuning(self):
        success_list = [ChargeRegime[f'1_electron_{dot}'] for dot in range(1, settings.dot_number + 1)]
        return self.charge_area is success_list

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


@dataclass(frozen=True)
class ExperimentalMeasurement:
    """ Data class to keep track of experimental measurement at each step. """
    x_axes: Sequence[float]
    y_axes: Sequence[float]
    data: torch.Tensor
