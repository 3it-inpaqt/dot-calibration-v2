from dataclasses import dataclass
from math import cos, pi, sin
from typing import Callable, Iterable, List, Optional, Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.diagram import Diagram


@dataclass
class Direction:
    """ Data class to factorise code. """
    is_stuck: bool = False
    last_x: int = 0
    last_y: int = 0
    move: Callable = None
    check_stuck: Callable = None

    @staticmethod
    def all_stuck(directions: Iterable["Direction"]):
        return all(d.is_stuck for d in directions)


class JumpShifting(AutotuningProcedure):
    # Number of exploration steps before to give up each phase
    _max_steps_exploration: int = 400
    _max_steps_search_empty: int = 60

    # Line angle degree (0 = horizontal - 90 = vertical)
    _line_direction: int = 90
    # List of distance between lines in pixel
    _line_distances: List[int] = None
    # Coordinate of the leftmost line found so far
    _leftmost_line_coord: Optional[Tuple[int, int]] = None

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        self.x, self.y = start_coord

        self._search_first_line(diagram)
        self._line_direction = self._search_line_direction(diagram)
        self._search_empty(diagram)
        self._guess_one_electron(diagram)

        return self.get_patch_center()

    def _search_first_line(self, diagram: Diagram) -> bool:
        """
        Search any line from the tuning starting point by exploring 4 directions.

        :param diagram: The diagram to explore.
        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """

        # First scan at the start position
        if self._is_confirmed_line(diagram):
            return True

        directions = [
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_left, check_stuck=self.is_max_down_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_left, check_stuck=self.is_max_up_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_right, check_stuck=self.is_max_up_right),
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_right, check_stuck=self.is_max_down_right),
        ]

        # Stop if max exploration steps reach or all directions are stuck (reach corners)
        nb_exploration_steps = 0
        while nb_exploration_steps < self._max_steps_exploration and not Direction.all_stuck(directions):

            # Move and search line in every not stuck directions
            for direction in (d for d in directions if not d.is_stuck):
                nb_exploration_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck(diagram)  # Check if reach a corner

                if self._is_confirmed_line(diagram):
                    return True

        return False

    def _search_line_direction(self, diagram: Diagram) -> int:
        """
        Estimate the direction of the current line.

        :param diagram: The diagram to explore.
        :return: The estimated direction in degree.
        """
        return 75  # Hardcoded for now

    def _search_empty(self, diagram: Diagram) -> None:
        """
        Explore the diagram by scanning patch perpendicular to the estimated lines direction.
        :param diagram: The diagram to explore.
        """

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost by default.
        self._leftmost_line_coord = self.x, self.y

        nb_search_steps = 0

        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_max_left)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_max_right)
        directions = [left, right]

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions):
            for direction in (d for d in directions if not d.is_stuck):
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck(diagram)  # Check if reach a corner

                self._is_confirmed_line(diagram)  # Check line and save position if leftmost one

    def _guess_one_electron(self, diagram: Diagram) -> None:
        """
        According to the leftmost line validated, guess a good location for the 1 electron regime.
        Then move to this location.

        :param diagram: The diagram to explore.
        """

        # If no line found, desperately guess random position as last resort
        if self._leftmost_line_coord is None:
            self.x, self.y = self.get_random_coordinates_in_diagram(diagram)
            return

        x, y = self._leftmost_line_coord
        self.move_to_coord(x, y)
        self._move_right_perpendicular_to_line()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(diagram, force=True)

    def _is_confirmed_line(self, diagram: Diagram) -> bool:
        """
        Check if the current position should be considered as a line, according to the current model and the
        validation logic.
        If a line is validated update the leftmost line.

        :param diagram: The diagram to explore.
        :return: True if a line is detected and considered as valid.
        """
        # Infer with the model at the current position
        line_detected, _ = self.is_transition_line(diagram)

        # If this is the leftmost line detected so far, save it
        if line_detected and (self._leftmost_line_coord is None or self.y < self._leftmost_line_coord[1]):
            self._leftmost_line_coord = self.x, self.y

        return line_detected

    def _move_relative_to_line(self, angle: int, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates in a direction relative to the estimated lines directions.

        :param angle:
        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        # Compute distance
        distance_x = step_size if step_size is not None else self._default_step_x
        distance_y = step_size if step_size is not None else self._default_step_y

        # Convert angle from degree to radian
        angle = angle * (pi / 180)

        self.x = round(self.x + (distance_x * cos(angle)))
        self.y = round(self.y + (distance_y * sin(angle)))

    def _move_left_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_relative_to_line

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self._move_relative_to_line(270 - self._line_direction, step_size)

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self._move_relative_to_line(90 - self._line_direction, step_size)

    def _move_up_follow_line(self, step_size: Optional[int] = None) -> None:
        self._move_relative_to_line(180 - self._line_direction, step_size)

    def _move_down_follow_line(self, step_size: Optional[int] = None) -> None:
        self._move_relative_to_line(-self._line_direction, step_size)

    def reset_procedure(self):
        super().reset_procedure()
        self._line_direction = 90
        self._line_distances = []
        self._leftmost_line_coord = None