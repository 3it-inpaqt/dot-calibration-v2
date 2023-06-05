import numpy as np

from autotuning.jump import Jump
from utils.logger import logger
from utils.settings import settings


class ParasitDotProcedure(Jump):
    """
    Procedure to detect if the line is a parasit dot or not
    """

    def _is_confirmed_line(self) -> bool:
        """
        Check if the current position should be considered as a line, according to the current model and the
        validation logic.
        If a line is validated update the leftmost line.

        :return: True if a line is detected and considered as valid.
        """

        # Infer with the model at the current position
        line_detected, _ = self.is_transition_line()

        # Check parasit dot
        if settings.parasit_dot_procedure and line_detected:
            line_detected = self._confirm_line()

        # If this is the leftmost line detected so far, save it
        if line_detected and (self._leftmost_line_coord is None or self._is_left_relative_to_line()):
            self._leftmost_line_coord = self.x, self.y

        return line_detected

    def _confirm_line(self) -> bool:
        delta = 5
        epsilon = 1
        self._line_distances.append()
        logger.debug(f'Step Check Parasit Dot')
        if len(self._line_distances) > 2:
            last_average_distance = np.mean(self._line_distances[-2:])
            average_distance = np.mean(self._line_distances[:-1])
            if average_distance - epsilon < last_average_distance < average_distance + epsilon:
                logger.debug(f'Result: True line')
                return True
            else:
                logger.debug(f'Result: True parasite dot')
                return False
        elif len(self._line_distances) == 1:
            previous_angle = self._line_slope
            self._search_line_slope()
            if previous_angle - delta < self._line_slope < previous_angle + delta:
                return True
            else:
                return False
        else:
            return True
