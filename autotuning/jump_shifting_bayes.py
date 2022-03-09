from autotuning.jump_shifting import Direction, JumpShifting


class JumpShiftingBayes(JumpShifting):
    # Number of exploration steps before to give up the line checking
    _max_steps_checking_line: int = 6

    # Threshold to consider the model inference good enough
    _confidence_valid: float = 0.90

    def _is_confirmed_line(self) -> bool:
        """
        Check if the current position should be considered as a line, according to the current model and the
        validation logic.
        If a line is validated update the leftmost line.

        :return: True if a line is detected and considered as valid.
        """

        # Infer with the model at the current position
        line_detected, confidence = self.is_transition_line()

        if confidence < self._confidence_valid:
            # Confidence too low, need checking
            x, y = self.x, self.y
            line_detected = self._checking_line(line_detected, confidence)
            self.move_to_coord(x, y)  # Back to the position we were before checking

        # If this is the leftmost line detected so far, save it
        if line_detected and (self._leftmost_line_coord is None or self.x < self._leftmost_line_coord[1]):
            self._leftmost_line_coord = self.x, self.y

        return line_detected

    def _checking_line(self, current_line: bool, current_confidence: float) -> bool:
        """
        Follow the supposed direction of a line until a high confidence inference is reached.

        :param current_line: The line classification inference for the current position.
        :param current_confidence: The line classification confidence for the inference of the current position.
        :return: True if it was possible to follow the line the required number of time in a row.
        """

        nb_search_steps = 0

        up = Direction(last_x=self.x, last_y=self.y, move=self._move_up_follow_line, check_stuck=self.is_max_up)
        down = Direction(last_x=self.x, last_y=self.y, move=self._move_down_follow_line, check_stuck=self.is_max_down)
        directions = [up, down]

        best_guess, best_confidence = current_line, current_confidence

        while nb_search_steps < self._max_steps_checking_line and not Direction.all_stuck(directions):
            for direction in (d for d in directions if not d.is_stuck):
                nb_search_steps += 1
                self._step_descr = f'checking line ({nb_search_steps}/{self._max_steps_checking_line})'

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_detected, confidence = self.is_transition_line()

                if confidence > self._confidence_valid:
                    # Enough confidence to confirm or not
                    self._step_descr = ''
                    return line_detected

                # Not enough information to validate, but keep the best inference
                if confidence > best_confidence:
                    best_guess = line_detected

        self._step_descr = ''
        return best_guess
