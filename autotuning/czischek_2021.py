from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class Czischek2021(AutotuningProcedure):
    """ Autotuning procedure from https://arxiv.org/abs/2101.03181 """

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        pass
