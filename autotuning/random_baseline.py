import random
from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure


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

    def _tune(self) -> Tuple[int, int]:
        return random.randint(0, len(self.diagram.x_axes) - 1), random.randint(0, len(self.diagram.y_axes) - 1)

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        # Never plot animation for this procedure.
        pass

    def plot_step_history(self, final_volt_coord: Tuple[float, float], success_tuning: bool) -> None:
        # Never plot this procedure.
        pass

    def __str__(self):
        return 'Random Baseline'
