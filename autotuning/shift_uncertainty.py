from autotuning.shift import Shift


class ShiftUncertainty(Shift):
    """
    Autotuning procedure adapted from https://doi.org/10.1088/2632-2153/ac34db
    But using confidence to validate a line instead of following it
    """

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
            line_detected, confidence = self.is_transition_line()

            # If the model is confident about the prediction, update the number of no line in a row
            # If not, ignore this step and continue
            if self.model.is_above_confident_threshold(line_detected, confidence):
                if line_detected:
                    no_line_in_a_row = 0
                else:
                    no_line_in_a_row += 1

            self.move_left()

        # This step is a success if the no line in a row condition is reached
        return no_line_in_a_row < self._nb_validation_empty

    def _is_confirmed_line(self, up: bool, current_line: bool, current_confidence: float) -> bool:
        """
        Follow the approximate direction of the line to valid, or not, the line.

        :param up: If True, follow the line in top direction, if False follow in bottom direction?
        :param current_line: The line classification inference for the current position.
        :param current_confidence: The line classification confidence for the inference of the current position.
        :return: True if it was possible to follow the line the required number of time in a row.
        """
        self._step_descr = 'checking line'

        max_to_confirm = self._nb_validation_line_forward if up else self._nb_validation_line_backward
        best_guess, best_confidence = current_line, current_confidence
        line_detected, confidence = current_line, current_confidence

        while max_to_confirm > 0 and not self.is_max_up():
            max_to_confirm -= 1

            if self.model.is_above_confident_threshold(line_detected, confidence):
                # Enough confidence to confirm or not
                return line_detected

            # Not enough information to validate, but keep the best inference
            if confidence > best_confidence:
                best_guess = line_detected

            if up:
                self.move_left(self._shift_size_follow_line)
                self.move_up()
            else:
                self.move_right(self._shift_size_follow_line)
                self.move_down()

            line_detected, confidence = self.is_transition_line()

        # Max limit or border reach, return best guess
        return best_guess
