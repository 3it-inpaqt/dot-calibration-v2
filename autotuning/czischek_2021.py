from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


class Czischek2021(AutotuningProcedure):
    """ Autotuning procedure from https://arxiv.org/abs/2101.03181 """

    _search_first_line_step_limit: int = 20
    _search_zero_electron_limit: int = 40
    _search_one_electron_limit: int = 50

    _nb_validation_line_forward: int = 10  # Number of steps
    _nb_validation_line_backward: int = 4  # Number of steps

    _shift_size_follow_line: int = 1  # Number of pixels

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        self.x, self.y = start_coord

        self.search_line(diagram)
        self.search_zero_electron(diagram)
        self.search_one_electron(diagram)

        return self.get_patch_center()

    def search_line(self, diagram: Diagram) -> bool:
        """
        Search any line from the tuning starting point.

        :param diagram: The diagram to explore.
        :return: True if we found the a line, False if we reach the step limit without detecting a line.
        """
        for _ in range(self._search_first_line_step_limit):
            line_detected, _ = self.is_transition_line(diagram)

            if line_detected:
                # Follow line up to validate the line detection
                line_validated = True
                for _ in range(self._nb_validation_line_forward):
                    self.move_left(self._shift_size_follow_line)
                    self.move_up()
                    line_detected, _ = self.is_transition_line(diagram)
                    if not line_detected:
                        line_validated = False
                        break

                if line_validated:
                    return True  # First line found and validated

            else:
                # No line detected, move top left
                self.move_left()
                self.move_up()

        return False  # At this point we reached the step limit, we assume we passed the first line

    def search_zero_electron(self, diagram: Diagram) -> None:
        """
        Search the 0 electron regime.

        :param diagram: The diagram to explore.
        """
        # FIXME with no boundaries this one could run forever
        no_line_in_a_row = 0
        while no_line_in_a_row < self._search_zero_electron_limit:
            line_detected, _ = self.is_transition_line(diagram)

            if line_detected:
                # Follow line up
                self.move_left(self._shift_size_follow_line)
                self.move_up()
            else:
                no_line_in_a_row += 1
                self.move_left()

    def search_one_electron(self, diagram: Diagram) -> bool:
        """
        Search the first line starting from the 0 electron regime.

        :param diagram: The diagram to explore.
        :return: True if we found the first line, False if we reach the step limit without detecting a line.
        """
        for _ in range(self._search_one_electron_limit):
            line_detected, _ = self.is_transition_line(diagram)

            if line_detected:
                # Follow line up to validate the line detection
                line_validated = True
                for _ in range(self._nb_validation_line_backward):
                    self.move_right(self._shift_size_follow_line)
                    self.move_down()
                    line_detected, _ = self.is_transition_line(diagram)
                    if not line_detected:
                        line_validated = False
                        break

                if line_validated:
                    # We assume we are on the first transition line
                    self.move_right()
                    self.move_down()
                    return True  # First line found and validated

            else:
                # No line detected, keep moving right
                self.move_right()

        return False  # At this point we reached the step limit, we assume we passed the first line
