from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure


class FullScan(AutotuningProcedure):
    """
    Scan the full diagram, from bottom left to top right.
    For debugging purpose.
    """

    def tune(self) -> Tuple[int, int]:
        self.x, self.y = (0, 0)

        while not self.is_max_up():
            while not self.is_max_right():
                line_detected, _ = self.is_transition_line()
                self.move_right()
            self.move_to_coord(x=0)
            self.move_up()

        return 0, 0
