from math import atan2, cos, pi, sin
from typing import List, Optional, Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.data_structures import Direction, SearchLineSlope
from utils.settings import settings


class JumpShifting(AutotuningProcedure):
    # Number of exploration steps before to give up each phase
    _max_steps_exploration: int = 600
    _max_steps_search_empty: int = 100

    # Line angle degree (0 = horizontal - 90 = vertical)
    _line_slope: float = None
    # List of distance between lines in pixel
    _line_distances: List[int] = None
    # Coordinate of the leftmost line found so far
    _leftmost_line_coord: Optional[Tuple[int, int]] = None

    def _tune(self) -> Tuple[int, int]:
        self._search_first_line()
        if settings.auto_detect_slope:
            self._search_line_slope()
        self._search_empty()
        self._guess_one_electron()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return self.get_patch_center()

    def _search_first_line(self) -> bool:
        """
        Search any line from the tuning starting point by exploring 4 directions.

        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """
        self._step_name = '1. Search first line'

        # First scan at the start position
        if self._is_confirmed_line():
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
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                if self._is_confirmed_line():
                    return True

        return False

    def _search_line_slope(self) -> None:
        """
        Estimate the direction of the current line and update the _line_slope attribut if the measurement looks valid.
        """
        self._step_name = '2. Search line slope'

        start_x, start_y = self.x, self.y

        # Step distance relative to the line distance to reduce the risk to reach another line
        step_distance = round(self._default_step_y * self._get_avg_line_distance())
        # Start angle base on prior knowledge
        start_angle = round(self._line_slope)

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        # Scan top and bottom with a specific angle range
        # TODO could be smarter
        for delta, search in zip((180, 0), searches):
            for angle in range(start_angle - 25, start_angle + 25, 4):
                self.move_to_coord(start_x, start_y)
                self._move_relative_to_line(delta - angle, step_distance)
                line_detected, _ = self.is_transition_line()  # No confidence validation here
                search.scans_results.append(line_detected)
                search.scans_positions.append(self.get_patch_center())

        # Valid coherent results, if not, do not save slope
        if all(search.is_valid_sequence() for search in searches):
            line_coords_estimations = []  # For Top and Bottom
            for search in searches:
                # Get coordinates of first and last lines detected
                x_1, y_1 = search.get_line_boundary(first=True)
                x_2, y_2 = search.get_line_boundary(first=False)

                line_coords_estimations.append((((x_1 + x_2) / 2), ((y_1 + y_2) / 2)))

            x_top, y_top = line_coords_estimations[0]
            x_bot, y_bot = line_coords_estimations[1]
            # X and Y inverted because my angle setup is wierd
            slope_estimation = atan2(x_top - x_bot, y_top - y_bot) * 180 / pi + 90

            print(f'{self.diagram.file_basename} - Top: {x_top, y_top} - Bot: {x_bot, y_bot} - {slope_estimation = }')
            self._line_slope = slope_estimation

        # Reset to starting point
        self.move_to_coord(start_x, start_y)

    def _search_empty(self) -> None:
        """
        Explore the diagram by scanning patch perpendicular to the estimated lines direction.
        """
        self._step_name = f'{3 if settings.auto_detect_slope else 2}. Search 0 e-'

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost by default.
        self._leftmost_line_coord = self.x, self.y

        nb_search_steps = 0

        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_max_left)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_max_right)
        directions = (left, right)
        left.no_line_count = 0
        right.no_line_count = 0

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions):
            for direction in (d for d in directions if not d.is_stuck):
                avg_line_distance = self._get_avg_line_distance()
                self._step_descr = f'line slope: {self._line_slope:.0f}°\navg line dist: {avg_line_distance:.1f}'
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_detected = self._is_confirmed_line()  # Check line and save position if leftmost one

                # If new line detected, save distance and reset counter
                if line_detected:
                    if direction.no_line_count >= 1:
                        self._line_distances.append(direction.no_line_count)
                    direction.no_line_count = 0
                # If no line detected since long time compare to the average distance between line, stop to go in this
                # direction
                else:
                    direction.no_line_count += 1
                    if direction.no_line_count > 3 * avg_line_distance:  # TODO could use x2 mean + x2 std
                        direction.is_stuck = True

    def _guess_one_electron(self) -> None:
        """
        According to the leftmost line validated, guess a good location for the 1 electron regime.
        Then move to this location.
        """

        # If no line found, desperately guess random position as last resort
        if self._leftmost_line_coord is None:
            self.x, self.y = self.get_random_coordinates_in_diagram()
            return

        x, y = self._leftmost_line_coord
        self.move_to_coord(x, y)
        self._move_right_perpendicular_to_line(round(self._default_step_x * (self._get_avg_line_distance() / 2 + 1)))

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)

    def _is_confirmed_line(self) -> bool:
        """
        Check if the current position should be considered as a line, according to the current model and the
        validation logic.
        If a line is validated update the leftmost line.

        :return: True if a line is detected and considered as valid.
        """

        # Infer with the model at the current position
        line_detected, _ = self.is_transition_line()

        # If this is the leftmost line detected so far, save it
        if line_detected and (self._leftmost_line_coord is None or self.x < self._leftmost_line_coord[0]):
            self._leftmost_line_coord = self.x, self.y

        return line_detected

    def _move_relative_to_line(self, angle: float, step_size: Optional[int] = None) -> None:
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
        self._move_relative_to_line(270 - self._line_slope, step_size)

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self._move_relative_to_line(90 - self._line_slope, step_size)

    def _move_up_follow_line(self, step_size: Optional[int] = None) -> None:
        self._move_relative_to_line(180 - self._line_slope, step_size)

    def _move_down_follow_line(self, step_size: Optional[int] = None) -> None:
        self._move_relative_to_line(-self._line_slope, step_size)

    def _get_avg_line_distance(self) -> float:
        """ Get the mean line distance. """
        return sum(self._line_distances) / len(self._line_distances)

    def reset_procedure(self):
        super().reset_procedure()

        if settings.research_group == 'michel_pioro_ladriere':
            self._line_slope = 75  # Prior assumption about line direction
            self._line_distances = [5]  # Prior assumption about distance between lines

        elif settings.research_group == 'louis_gaudreau':
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances = [3]  # Prior assumption about distance between lines

        self._leftmost_line_coord = None
