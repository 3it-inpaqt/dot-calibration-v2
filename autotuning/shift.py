from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure


class Shift(AutotuningProcedure):
    """ Autotuning procedure from https://doi.org/10.1088/2632-2153/ac34db """

    _search_line_step_limit: int = 100
    _search_zero_electron_limit: int = 150
    _search_one_electron_limit: int = 200

    _nb_validation_line_forward: int = 10  # Number of steps
    _nb_validation_line_backward: int = 4  # Number of steps
    _nb_validation_empty: int = 40  # Number of steps

    _shift_size_follow_line: int = 1  # Number of pixels
    _shift_size_backward_down: int = 2  # Number of pixels

    def _tune(self) -> Tuple[int, int]:
        self._search_line()
        self._search_zero_electron()
        self._search_one_electron()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return self.get_patch_center()

    def _search_line(self) -> bool:
        """
        Search any line from the tuning starting point.

        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """
        self._step_name = '1. Search first line'

        step_count = 0
        # Search until step limit reach, or we arrive at the top left corder of the diagram (for hard policy only)
        while step_count < self._search_line_step_limit and not (self.is_max_left() and self.is_max_up()):
            step_count += 1
            line_detected, confidence = self.is_transition_line()

            if line_detected and not self.is_max_up():
                # Follow line up to validate the line detection
                if self._is_confirmed_line(True, line_detected, confidence):
                    return True  # First line found and validated

            else:
                # No line detected, move top left
                self.move_left()
                self.move_up()

        return False  # At this point we reached the step limit, we assume we passed the first line

    def _search_zero_electron(self) -> bool:
        """
        Search the 0 electron regime.
        """
        self._step_name = '2. Search 0 e-'

        no_line_in_a_row = 0
        nb_steps = 0
        # Search until the empty regime is found (K no line in a row) or step limit is reach, or we arrive at the top
        # left corder of the diagram (for hard policy only)
        while no_line_in_a_row < self._nb_validation_empty and \
                nb_steps < self._search_zero_electron_limit and \
                not self.is_max_left():
            nb_steps += 1
            line_detected, _ = self.is_transition_line()

            if line_detected:
                no_line_in_a_row = 0
                if self.is_max_up():
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

    def _search_one_electron(self) -> bool:
        """
        Search the first line starting from the 0 electron regime.

        :return: True if we found the first line, False if we reach the step limit without detecting a line.
        """
        self._step_name = '3. Search 1 e−'

        nb_steps = 0
        # Search until step limit reach, or we arrive at the bottom right corder of the diagram (for hard policy only)
        while nb_steps < self._search_one_electron_limit and \
                not (self.is_max_right() and self.is_max_down()):
            nb_steps += 1
            line_detected, confidence = self.is_transition_line()

            if line_detected:
                # Follow line up to validate the line detection
                if self._is_confirmed_line(False, line_detected, confidence):
                    # We assume we are on the first transition line
                    self.move_right(self._default_step_x * 2)
                    self.move_down()
                    return True  # First line found and validated

            else:
                # No line detected, keep moving right
                self.move_right()
                self.move_down(self._shift_size_backward_down)

        return False  # At this point we reached the step limit, we assume we passed the first line

    def _is_confirmed_line(self, up: bool, current_line: bool, current_confidence: float) -> bool:
        """
        Follow the approximate direction of the line to valid, or not, the line.

        :param up: If True, follow the line in top direction, if False follow in bottom direction?
        :param current_line: The line classification inference for the current position.
        :param current_confidence: The line classification confidence for the inference of the current position.
        :return: True if it was possible to follow the line the required number of time in a row.
        """
        self._step_descr = 'checking line'

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
            line_detected, _ = self.is_transition_line()
            if not line_detected or self.is_max_up():
                self._step_descr = ''
                return False
        self._step_descr = ''
        return True  # Line confirmed
