"""
Bunch of dataclasses to structure information and simplify code.
"""
from dataclasses import dataclass
from typing import List

from utils.settings import settings


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
