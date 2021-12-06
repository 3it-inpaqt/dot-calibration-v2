import random
from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class RandomBaseline(AutotuningProcedure):
    """
    Baseline use to compare with other procedures.
    Return random coordinates within the current diagram.
    """

    def __init__(self, patch_size: Tuple[int, int], label_offsets: Tuple[int, int] = (0, 0)):
        """
        It is not possible to provide a model for this procedure.
        """
        super().__init__(None, patch_size, label_offsets, True)

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        return random.randint(0, len(diagram.x_axes) - 1), random.randint(0, len(diagram.y_axes) - 1)

    def __str__(self):
        return 'Random Baseline'
