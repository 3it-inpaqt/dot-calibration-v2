from dataclasses import dataclass
from typing import Callable, Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


@dataclass
class Direction:
    """ Data class to factorise code in exploration phase. """
    is_stuck: bool = False
    last_x: int = 0
    last_y: int = 0
    move: Callable = None
    check_stuck: Callable = None


class JumpShifting(AutotuningProcedure):
    _max_exploration_steps: int = 400  # Number of exploration steps before to give up (sum all directions)

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        self.x, self.y = start_coord

        self._search_first_line(diagram)
        self.search_line_direction(diagram)
        self._search_zero_electron(diagram)

        return 0, 0

    def _search_first_line(self, diagram):
        """
        Search any line from the tuning starting point by exploring 4 directions.

        :param diagram: The diagram to explore.
        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """

        nb_exploration_steps = 0

        # First scan at the start position
        line_detected, confidence = self.is_transition_line(diagram)

        if line_detected:
            return True

        directions = [
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_left, check_stuck=self.is_max_down_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_left, check_stuck=self.is_max_up_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_right, check_stuck=self.is_max_up_right),
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_right, check_stuck=self.is_max_down_right),
        ]

        # Stop if max exploration steps reach or all directions are stuck (reach corners)
        while nb_exploration_steps < self._max_exploration_steps and not (all(d.is_stuck for d in directions)):

            # Move and search line in every not stuck directions
            for direction in (d for d in directions if not d.is_stuck):
                nb_exploration_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)
                direction.move()
                direction.last_x, direction.last_y = self.x, self.y
                direction.is_stuck = direction.check_stuck(diagram)

                line_detected, confidence = self.is_transition_line(diagram)
                if line_detected:
                    return True

    def search_line_direction(self, diagram):
        pass

    def _search_zero_electron(self, diagram):
        pass
