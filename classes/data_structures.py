"""
Bunch of dataclasses and enumerations to structure information and simplify code.
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Iterable, List, Tuple

from utils.settings import settings


@unique
class ChargeRegime(Enum):
    """ Charge regime enumeration """
    UNKNOWN = 'unknown'
    ELECTRON_0 = '0_electron'
    ELECTRON_1 = '1_electron'
    ELECTRON_2 = '2_electrons'
    ELECTRON_3 = '3_electrons'
    ELECTRON_4_PLUS = '4_electrons'  # The value is no '4+_electrons' because labelbox remove the '+'

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
    model_classification: bool
    model_confidence: bool
    ground_truth: bool
    description: str

    def is_classification_correct(self) -> bool:
        """ Return True only if model_classification is the same as the ground_truth. """
        return self.model_classification == self.ground_truth


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
    nb_steps: int
    nb_classification_success: int
    charge_area: ChargeRegime
    final_coord: Tuple[int, int]

    @property
    def is_success_tuning(self):
        return self.charge_area is ChargeRegime.ELECTRON_1

    @property
    def success_rate(self):
        return self.nb_classification_success / self.nb_steps if self.nb_steps > 0 else 0
