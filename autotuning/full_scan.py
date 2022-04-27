from multiprocessing import Pool
from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from plots.data import plot_diagram
from utils.misc import get_nb_loader_workers
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

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        # Never plot animation for this procedure.
        pass

    def plot_step_history(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        """
            Plot the diagram with the tuning steps of the current procedure.

            :param final_coord: The final coordinate of the tuning procedure
            :param success_tuning: Result of the tuning (True = Success)
            """
        d = self.diagram
        values = d.values.cpu()
        name = f'{self.diagram.file_basename} steps\n{self}'

        # Parallel plotting for speed.
        with Pool(get_nb_loader_workers()) as pool:
            # diagram + step with classification color
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values, 'image_name': name,
                                   'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                   'scan_history': self._scan_history, 'show_offset': False,
                                   'history_uncertainty': False, 'show_crosses': False})
            # step with classification color and uncertainty
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None,
                                   'image_name': name + ' uncertainty', 'interpolation_method': 'nearest',
                                   'pixel_size': d.x_axes[1] - d.x_axes[0], 'scan_history': self._scan_history,
                                   'show_offset': False, 'history_uncertainty': True, 'show_crosses': False})
            # diagram + label + step with classification color
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values, 'image_name': name + ' labels',
                                   'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                   'transition_lines': d.transition_lines, 'scan_history': self._scan_history,
                                   'show_offset': False, 'history_uncertainty': False, 'show_crosses': False})
            # label + step with classification color and uncertainty
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None,
                                   'image_name': name + ' uncertainty labels', 'interpolation_method': 'nearest',
                                   'pixel_size': d.x_axes[1] - d.x_axes[0], 'transition_lines': d.transition_lines,
                                   'scan_history': self._scan_history, 'show_offset': False,
                                   'history_uncertainty': True, 'show_crosses': False})
            # step with error color and uncertainty
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None, 'image_name': name + ' error',
                                   'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                   'transition_lines': d.transition_lines, 'scan_history': self._scan_history,
                                   'show_offset': False, 'scan_errors': True, 'history_uncertainty': True,
                                   'show_crosses': False})

            # Wait for the processes to finish
            pool.close()
            pool.join()
