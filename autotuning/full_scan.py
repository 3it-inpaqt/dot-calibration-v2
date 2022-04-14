from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from plots.data import plot_diagram


class FullScan(AutotuningProcedure):
    """
    Scan the full diagram, from bottom left to top right.
    For debugging purpose.
    """

    def _tune(self) -> Tuple[int, int]:
        self.x, self.y = (0, 0)

        while not self.is_max_up():
            while not self.is_max_right():
                line_detected, _ = self.is_transition_line()
                self.move_right()
            self.move_to_coord(x=0)
            self.move_up()

        return 0, 0

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        # Never plot animation for this procedure.
        pass

    def plot_step_history(self, final_coord: Tuple[int, int], success_tuning: bool, plot_vanilla: bool = True) -> None:
        """
            Plot the diagram with the tuning steps of the current procedure.

            :param final_coord: The final coordinate of the tuning procedure
            :param success_tuning: Result of the tuning (True = Success)
            :param plot_vanilla: If True, also plot the diagram with no label and steps
            """
        d = self.diagram
        values = d.values.cpu()
        name = f'{self.diagram.file_basename} steps\n{self}'

        if plot_vanilla:
            # diagram
            plot_diagram(d.x_axes, d.y_axes, values, f'{d.file_basename}', 'nearest', d.x_axes[1] - d.x_axes[0])

        # diagram + step with classification color
        plot_diagram(d.x_axes, d.y_axes, values, name, 'nearest', d.x_axes[1] - d.x_axes[0],
                     scan_history=self._scan_history, show_offset=False, history_uncertainty=False, show_crosses=False)
        # step with classification color and uncertainty
        plot_diagram(d.x_axes, d.y_axes, None, name + ' uncertainty', 'nearest', d.x_axes[1] - d.x_axes[0],
                     scan_history=self._scan_history, show_offset=False, history_uncertainty=True, show_crosses=False)
        # diagram + label + step with classification color
        plot_diagram(d.x_axes, d.y_axes, values, name + ' labels', 'nearest', d.x_axes[1] - d.x_axes[0],
                     transition_lines=d.transition_lines, scan_history=self._scan_history,
                     show_offset=False, history_uncertainty=False, show_crosses=False)
        # label + step with classification color and uncertainty
        plot_diagram(d.x_axes, d.y_axes, None, name + ' uncertainty labels', 'nearest',
                     d.x_axes[1] - d.x_axes[0], transition_lines=d.transition_lines, scan_history=self._scan_history,
                     show_offset=False, history_uncertainty=True, show_crosses=False)
