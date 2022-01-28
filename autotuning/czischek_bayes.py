from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class CzischekBayes(AutotuningProcedure):
    """
    Autotuning procedure adapted from https://arxiv.org/abs/2101.03181
    But using confidence to validate a line instead of following it
    """

    _search_line_step_limit: int = 20
    _search_zero_electron_limit: int = 40
    _search_one_electron_limit: int = 50

    _nb_validation_empty: int = 40  # Number of steps

    # Number estimated by grid search with Michel diagrams and BCNN
    # 0.50 => 71% | 0.75 => 82% | 0.80 => 81% | 0.85 => 87% | 0.90 => 87% | 0.95 => 87%
    _confidence_valid: float = 0.90

    _shift_size_follow_line: int = 1  # Number of pixels
    _shift_size_backward_down: int = 2  # Number of pixels

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        self.x, self.y = start_coord

        self._search_line(diagram)
        self._search_zero_electron(diagram)
        self._search_one_electron(diagram)

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(diagram, force=True)
        return self.get_patch_center()

    def _search_line(self, diagram: Diagram) -> bool:
        """
        Search any line from the tuning starting point.

        :param diagram: The diagram to explore.
        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """
        step_count = 0
        # Search until step limit reach, or we arrive at the top left corder of the diagram (for hard policy only)
        while step_count < self._search_line_step_limit and not (self.is_max_left(diagram) and self.is_max_up(diagram)):
            step_count += 1
            line_detected, confidence = self.is_transition_line(diagram)

            if line_detected and confidence > self._confidence_valid:
                # Line detected and validated
                return True
            else:
                # No line detected, move top left
                self.move_left()
                self.move_up()

        return False  # At this point we reached the step limit, we assume we passed the first line

    def _search_zero_electron(self, diagram: Diagram) -> bool:
        """
        Search the 0 electron regime.

        :param diagram: The diagram to explore.
        """
        no_line_in_a_row = 0
        nb_steps = 0
        # Search until the empty regime is found (K no line in a row) or step limit is reach or we arrive at the top
        # left corder of the diagram (for hard policy only)
        while no_line_in_a_row < self._nb_validation_empty and \
                nb_steps < self._search_zero_electron_limit and \
                not (self.is_max_left(diagram) and self.is_max_up(diagram)):
            nb_steps += 1
            line_detected, confidence = self.is_transition_line(diagram)

            # If the model is confident about the prediction, update the number of no line in a row
            # If not, ignore this step and continue
            if confidence > self._confidence_valid:
                if line_detected:
                    no_line_in_a_row = 0
                else:
                    no_line_in_a_row += 1

            self.move_left()

        # This step is a success if the no line in a row condition is reached
        return no_line_in_a_row < self._nb_validation_empty

    def _search_one_electron(self, diagram: Diagram) -> bool:
        """
        Search the first line starting from the 0 electron regime.

        :param diagram: The diagram to explore.
        :return: True if we found the first line, False if we reach the step limit without detecting a line.
        """
        nb_steps = 0
        # Search until step limit reach or we arrive at the bottom right corder of the diagram (for hard policy only)
        while nb_steps < self._search_one_electron_limit and \
                not (self.is_max_right(diagram) and self.is_max_down(diagram)):
            nb_steps += 1
            line_detected, confidence = self.is_transition_line(diagram)

            if line_detected and confidence > self._confidence_valid:
                # Line detected and validated
                # Just move a bit to don't be in the line
                self.move_right()
                self.move_down()
                return True

            else:
                # No line detected, keep moving right
                self.move_right()
                self.move_down(self._shift_size_backward_down)

        return False  # At this point we reached the step limit, we assume we passed the first line
