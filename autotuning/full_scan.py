from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from utils.settings import settings


class FullScan(AutotuningProcedure):
    """
    Scan the full diagram, from bottom left to top right.
    For debugging purpose.
    """

    def _tune(self) -> Tuple[int, int]:
        self.x, self.y = (0, 0)

        while not self.is_max_up():
            while not self.is_max_right():
                self.add_to_inference_batch()  # Add to inference batch but wait to process
                if self.nb_pending() >= settings.batch_size:
                    self.is_transition_line_batch()  # Group inference for speed improvement

                self.move_right()
            self.move_to_coord(x=0)
            self.move_up()

        self.is_transition_line_batch()  # Final group inference
        return 0, 0

    def plot_step_history_animation(self, final_coord: Tuple[float, float], success_tuning: bool) -> None:
        # Never plot animation for this procedure.
        pass
