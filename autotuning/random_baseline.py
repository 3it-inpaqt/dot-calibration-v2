import random
from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class RandomBaseline(AutotuningProcedure):
    """
    Baseline use to compare with other procedures.
    Return random coordinates within the current diagram.
    """

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        return random.randint(0, len(diagram.x) - 1), random.randint(0, len(diagram.y) - 1)
