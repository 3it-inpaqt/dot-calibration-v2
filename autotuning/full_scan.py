from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class FullScan(AutotuningProcedure):
    """
    Scan the full diagram, from bottom left to top right.
    For debugging purpose.
    """

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        self.x, self.y = (0, 0)

        while not self.is_max_up(diagram):
            while not self.is_max_right(diagram):
                line_detected, _ = self.is_transition_line(diagram)
                self.move_right()
            self.move_to_coord(x=0)
            self.move_up()

        return 0, 0
