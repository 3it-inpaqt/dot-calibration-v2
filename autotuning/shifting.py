from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class Shifting(AutotuningProcedure):
    """ Autotuning procedure from https://doi.org/10.1088/2632-2153/ac34db """

    _search_line_step_limit: int = 20
    _search_zero_electron_limit: int = 40
    _search_one_electron_limit: int = 50

    _nb_validation_line_forward: int = 10  # Number of steps
    _nb_validation_line_backward: int = 4  # Number of steps
    _nb_validation_empty: int = 40  # Number of steps

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

            if line_detected and not self.is_max_up(diagram):
                # Follow line up to validate the line detection
                if self._is_confirmed_line(diagram, True, line_detected, confidence):
                    return True  # First line found and validated

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
        # Search until the empty regime is found (K no line in a row) or step limit is reach, or we arrive at the top
        # left corder of the diagram (for hard policy only)
        while no_line_in_a_row < self._nb_validation_empty and \
                nb_steps < self._search_zero_electron_limit and \
                not (self.is_max_left(diagram) and self.is_max_up(diagram)):
            nb_steps += 1
            line_detected, _ = self.is_transition_line(diagram)

            if line_detected:
                no_line_in_a_row = 0
                if self.is_max_up(diagram):
                    self.move_left()
                else:
                    # Follow line up
                    self.move_left(self._shift_size_follow_line)
                    self.move_up()
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
        # Search until step limit reach, or we arrive at the bottom right corder of the diagram (for hard policy only)
        while nb_steps < self._search_one_electron_limit and \
                not (self.is_max_right(diagram) and self.is_max_down(diagram)):
            nb_steps += 1
            line_detected, confidence = self.is_transition_line(diagram)

            if line_detected:
                # Follow line up to validate the line detection
                if self._is_confirmed_line(diagram, False, line_detected, confidence):
                    # We assume we are on the first transition line
                    self.move_right()
                    self.move_down()
                    return True  # First line found and validated

            else:
                # No line detected, keep moving right
                self.move_right()
                self.move_down(self._shift_size_backward_down)

        return False  # At this point we reached the step limit, we assume we passed the first line

    def _is_confirmed_line(self, diagram: Diagram, up: bool, current_line: bool, current_confidence: float) -> bool:
        """
        Follow the approximate direction of the line to valid, or not, the line.

        :param diagram: The diagram to explore.
        :param up: If True, follow the line in top direction, if False follow in bottom direction?
        :param current_line: The line classification inference for the current position.
        :param current_confidence: The line classification confidence for the inference of the current position.
        :return: True if it was possible to follow the line the required number of time in a row.
        """

        nb_to_confirm = self._nb_validation_line_forward if up else self._nb_validation_line_backward

        if current_line:
            nb_to_confirm -= 1

        for _ in range(nb_to_confirm):
            if up:
                self.move_left(self._shift_size_follow_line)
                self.move_up()
            else:
                self.move_right(self._shift_size_follow_line)
                self.move_down()
            line_detected, _ = self.is_transition_line(diagram)
            if not line_detected or self.is_max_up(diagram):
                return False
        return True  # Line confirmed
