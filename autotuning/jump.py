from collections import deque
from math import atan2, ceil, cos, pi, radians, sin, tan, sqrt
from typing import List, Optional, Tuple, Deque

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.data_structures import Direction, SearchLineSlope
from utils.logger import logger
from utils.settings import settings


class Jump(AutotuningProcedure):
    # Exploration limits
    _max_steps_exploration: int = 1000  # Nb of steps
    _max_steps_search_empty: int = 250  # Nb of steps
    _max_line_explore_right: int = 5  # Nb detected lines
    _max_steps_validate_left_line: int = 250  # Nb steps
    _max_nb_leftmost_checking: int = 6

    # Line angle degree (0 = horizontal | 90 = vertical | 45 = slope -1 | 135 = slope 1)
    _line_slope: float = None
    # Line distance prior knowledge as number of pixels
    _line_distances_prior: int = None
    # List of lines found
    _line_sequence: List[Tuple[int, int]] = None
    # List of lines found with potential duplicates and None for no line found
    _line_sequence_raw: Deque[Tuple[int, int] | None] = None

    @property
    def _nb_line_found(self) -> int:
        """ Number of lines found so far. """
        return len(self._line_sequence)

    @property
    def _leftmost_line_coord(self) -> Tuple[int, int] | None:
        """ Coordinates of the leftmost line found so far. """
        return self._line_sequence[0] if self._nb_line_found > 0 else None

    @property
    def _line_avg_distance(self) -> float:
        """
        Compute the average distance between lines found.
        The prior knowledge is considered as an initial measurement.

        :return: Average distance between lines found so far in pixels.
        """
        # We start with the prior knowledge
        distance_sequence = [self._line_distances_prior]

        for i in range(1, self._nb_line_found):
            # Compute distance between the current line and the previous one
            x_1, y_1 = self._line_sequence[i - 1]
            x_2, y_2 = self._line_sequence[i]

            # Special condition for 90° (vertical line) because tan(90) is undefined
            if self._line_slope == 90:
                distance_sequence.append(abs(x_1 - x_2))
            else:
                # Reconstruct line equation (ax + by + c = 0)
                a = tan(radians(- self._line_slope))  # Inverted angle because the setup is wierd
                b = -1
                c = y_1 - (x_1 * a)

                # Calculate the distance between the line and the point
                distance_sequence.append(abs(a * x_2 + b * y_2 + c) / sqrt(a ** 2 + b ** 2))

        # Return the average distance between lines
        return sum(distance_sequence) / len(distance_sequence)

    def _tune(self) -> Tuple[int, int]:
        # Search the first line, if none found return a random position
        if not self._search_first_line():
            return self.diagram.get_random_starting_point()

        # Optional step: detect line slope
        if settings.auto_detect_slope:
            self._search_line_slope()

        # Search empty regime
        self._search_empty()

        # Optional step: make sure we found the leftmost line
        if settings.validate_left_line:
            self.validate_left_line()

        # Move to one electron coordinates based on the leftmost line found
        self._guess_one_electron()

        return self.get_patch_center()

    def _search_first_line(self) -> bool:
        """
        Search any line from the tuning starting point by exploring 4 directions.

        :return: True if we found a line, False if we reach the step limit without detecting a line.
        """
        logger.debug('Stage (1) - Search first line')
        self._step_name = '1. Search first line'

        # First scan at the start position
        if self._is_confirmed_line():
            return True

        directions = [
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_left, check_stuck=self.is_corner),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_left, check_stuck=self.is_corner),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_right, check_stuck=self.is_corner),
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_right, check_stuck=self.is_corner),
        ]

        # Stop if max exploration steps reach or all directions are stuck (reach corners)
        nb_exploration_steps = 0
        while nb_exploration_steps < self._max_steps_exploration and not Direction.all_stuck(directions):

            # Move and search line in every not stuck direction
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
        logger.debug('Stage (2) - Search line slope')
        self._step_name = '2. Search line slope'

        start_x, start_y = self.x, self.y

        # Step distance relative to the line distance to reduce the risk to reach another line
        step_distance = round(self._line_avg_distance)
        # Start angle base on prior knowledge
        start_angle = round(self._line_slope)

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        max_angle_search = 65  # Max search angle on both sides of the prior knowledge
        search_step = 8
        # Scan top and bottom with a specific angle range
        for side, search in zip((180, 0), searches):
            init_line = False
            init_no_line = False
            delta = 0
            while abs(delta) < max_angle_search:
                self.move_to_coord(start_x, start_y)
                self._move_relative_to_line(side - start_angle + delta, step_distance)
                self._step_descr = f'delta: {delta}°\ninit line: {init_line}\ninit no line: {init_no_line}'
                line_detected, _ = self.is_transition_line()  # No confidence validation here

                if delta >= 0:
                    search.scans_results.append(line_detected)
                    search.scans_positions.append(self.get_patch_center())
                else:
                    search.scans_results.appendleft(line_detected)
                    search.scans_positions.appendleft(self.get_patch_center())

                # Stop when we found a line then no line twice
                if init_no_line and init_line and not line_detected:
                    break  # We are on the other side of the line

                if line_detected and not init_line:
                    init_line = True
                elif not line_detected and not init_no_line:
                    init_no_line = True
                    if init_line:
                        delta = -delta  # Change the orientation if we found a line then no line

                if init_line and init_no_line:
                    # Stop alternate, iterate delta depending on the direction
                    delta += search_step if delta >= 0 else -search_step
                else:
                    if delta >= 0:
                        delta += search_step  # Increment delta on one direction only because alternate
                    delta = -delta  # Alternate directions as long as we didn't find one line and one no line

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

            self._line_slope = slope_estimation

        # Reset to starting point
        self.move_to_coord(start_x, start_y)

    def _search_empty(self) -> None:
        """
        Explore the diagram by scanning patch perpendicular to the estimated lines direction.
        """
        logger.debug(f'Stage ({3 if settings.auto_detect_slope else 2}) - Search empty')
        self._step_name = f'{3 if settings.auto_detect_slope else 2}. Search 0 e-'

        # At the beginning of this function, we should be on a line.
        self._record_line(self.get_patch_center())

        nb_search_steps = 0

        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_corner)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_corner)
        directions = (left, right)

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions):
            for direction in (d for d in directions if not d.is_stuck):
                avg_line_distance = self._line_avg_distance
                self._step_descr = (f'line slope: {self._line_slope:.0f}°\n'
                                    f'avg line dist: {self._get_line_distance_str()}\n'
                                    f'nb line found: {self._nb_line_found}\n'
                                    f'leftmost line: {self._get_leftmost_line_coord_str()}')
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to the last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_detected = self._is_confirmed_line()  # Check line and save position if leftmost one

                if line_detected:
                    self._record_line(self.get_patch_center(), is_before=direction == left)
                    # Stop exploring right if we found enough lines
                    right.is_stuck = right.is_stuck or self._nb_line_found >= self._max_line_explore_right
                # If no line detected since long time compare to the average distance between line, stop to go in this
                # direction
                else:
                    self._record_line(None, is_before=direction == left)
                    # Stop to explore this direction if we found more than 1 line, and we found no line in 3x the
                    # average line distance in this direction.
                    last_line_distance = self._distance_from_last_line(is_before=direction == left)
                    if self._nb_line_found > 1 and last_line_distance > 3 * avg_line_distance:
                        direction.is_stuck = True

    def validate_left_line(self) -> None:
        """
        Validate that the current leftmost line detected is really the leftmost one.
        Try to find a line left by scanning area at regular interval on the left, where we could find a line.
        If a new line is found that way, do the validation again.
        """
        logger.debug(f'Stage ({4 if settings.auto_detect_slope else 3}) - Validate left line')
        self._step_name = f'{4 if settings.auto_detect_slope else 3}. Validate leftmost line'
        line_distance = self._line_avg_distance
        half_patch_x, half_patch_y = settings.patch_size_x / 2, settings.patch_size_y / 2

        # Go up and down following the line
        up = Direction(last_x=0, last_y=0, move=self._move_up_follow_line, check_stuck=self.is_max_up_or_left)
        down = Direction(last_x=0, last_y=0, move=self._move_down_follow_line, check_stuck=self.is_max_down_or_right)

        nb_steps = 0
        new_line_found = True
        start_point = self._leftmost_line_coord
        while new_line_found:
            nb_line_search = 0
            new_line_found = False
            # Both directions start at the leftmost point
            up.last_x, up.last_y = start_point
            down.last_x, down.last_y = start_point
            # Move the patch coordinates (bottom left corner) have the starting point at the center of the patch
            up.last_x, up.last_y = up.last_x - half_patch_x, up.last_y - half_patch_y
            down.last_x, down.last_y = down.last_x - half_patch_x, down.last_y - half_patch_y
            up.is_stuck = down.is_stuck = False  # Unstuck since we are stating at a new location
            while not new_line_found and not Direction.all_stuck((up, down)):
                for direction in (d for d in (up, down) if not d.is_stuck):
                    # Check if we reached the maximum number of scans for the current line
                    nb_line_search += 1
                    if nb_line_search > self._max_nb_leftmost_checking:
                        return
                    self.move_to_coord(direction.last_x, direction.last_y)  # Go to the last position of this direction
                    # Step distance relative to the line distance
                    direction.move(round(line_distance * 2), False)
                    direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                    direction.is_stuck = direction.check_stuck()
                    if direction.is_stuck:
                        break  # We don't scan if we have reached the border

                    # Skip half-line distance left
                    self._move_left_perpendicular_to_line(ceil(line_distance / 2), False)
                    self._record_line(None)  # We assume there is no line in this skipped section

                    # Go left for 2x the line distance (total 2.5x the line distance)
                    for i in range(ceil(line_distance * 2 / self._default_step_x)):
                        nb_steps += 1
                        # If new line found and this is the new leftmost one, start again the checking loop
                        if self._is_confirmed_line():
                            self._record_line(self.get_patch_center())
                            self._step_descr = f'line slope: {self._line_slope:.0f}°\n' \
                                               f'avg line dist: {self._get_line_distance_str()}\n' \
                                               f'nb line found: {self._nb_line_found}\n' \
                                               f'leftmost line: {self._get_leftmost_line_coord_str()}'
                            if start_point != self._leftmost_line_coord:
                                new_line_found = True
                                start_point = self._leftmost_line_coord
                                break
                        else:
                            self._record_line(None)

                        self._move_left_perpendicular_to_line()
                        if self.is_max_left() or self.is_max_down():
                            break  # Nothing else to see here

                    if nb_steps > self._max_steps_validate_left_line:
                        return  # Hard break to avoid infinite search in case of bad slope detection (>90°)

                    if new_line_found:
                        break

    def _guess_one_electron(self) -> None:
        """
        According to the leftmost line validated, guess a good location for the 1 electron regime.
        Then move to this location.
        """

        # If no line found, desperately guess random position as last resort
        if self._leftmost_line_coord is None:
            self.x, self.y = self.diagram.get_random_starting_point()
            return

        x, y = self._leftmost_line_coord
        self.move_to_coord_center(x, y)
        self._move_right_perpendicular_to_line(ceil(self._line_avg_distance / 2 + 1),
                                               avoid_small_steps=False)

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

        return line_detected

    def _is_left_relative_to_line(self) -> bool:
        """
        Check if the current position is at the left of the leftmost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new leftmost point.
        """
        x, y = self._leftmost_line_coord
        # Error margin to avoid unnecessary updates
        x -= self._default_step_x
        y -= self._default_step_y

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if self._line_slope == 90:
            return self.x < x

        # Reconstruct line equation (y = m*x + b)
        m = tan(radians(- self._line_slope))  # Inverted angle because the setup is wierd
        b = y - (x * m)

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = m * self.x + b
        # Invert y delta if oriented to the right
        y_delta = y_line - self.y if 0 < self._line_slope < 90 else self.y - y_line
        return (y_delta > 0 and m < 0) or (y_delta < 0 and m > 0)

    def _move_relative_to_line(self, angle: float, step_size: Optional[int] = None,
                               avoid_small_steps: bool = False) -> None:
        """
        Shift the current coordinates in a direction relative to the estimated lines directions.

        :param angle: The direction relative to the line (in degree).
        :param step_size: The step size for the shifting (number of pixels).
            If None, the procedure default value is used, which is the (patch size - offset) if None is specified
            neither at the initialization.
        :param avoid_small_steps: If True, the procedure will try to avoid steps smaller than the step size in every
            axis when reaching a border.
            If False, the normal border policy is applied.
        """

        init_x, init_y = self.x, self.y

        # Compute distance
        distance_x = step_size if step_size is not None else self._default_step_x
        distance_y = step_size if step_size is not None else self._default_step_y

        # Convert the angle from degree to radian
        angle = angle * (pi / 180)

        self.x = round(self.x + (distance_x * cos(angle)))
        self.y = round(self.y + (distance_y * sin(angle)))

        if avoid_small_steps:
            # Enforce boundary now to avoid tiny steps in some cases
            if self.is_max_left() or self.is_max_right():
                if angle < pi:
                    self.y = init_y + distance_y  # Go up
                else:
                    self.y = init_y - distance_y  # Go down
            elif self.is_max_up() or self.is_max_down():
                if angle < pi / 2 or angle > (3 * pi) / 2:
                    self.x = init_x + distance_x  # Go right
                else:
                    self.x = init_x - distance_x  # Go left

        self._enforce_boundary_policy()

    def _move_left_perpendicular_to_line(self, step_size: Optional[int] = None, avoid_small_steps=True) -> None:
        """
        Alias of _move_relative_to_line

        :param step_size: The step size for the shifting (number of pixels).
            If None, the procedure default value is used, which is the (patch size - offset) if None is specified
            neither at the initialization.
        """
        self._move_relative_to_line(270 - self._line_slope, step_size, avoid_small_steps)

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None, avoid_small_steps=True) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self._move_relative_to_line(90 - self._line_slope, step_size, avoid_small_steps)

    def _move_up_follow_line(self, step_size: Optional[int] = None, avoid_small_steps=True) -> None:
        self._move_relative_to_line(180 - self._line_slope, step_size, avoid_small_steps)

    def _move_down_follow_line(self, step_size: Optional[int] = None, avoid_small_steps=True) -> None:
        self._move_relative_to_line(-self._line_slope, step_size, avoid_small_steps)

    def _record_line(self, coord: Tuple[int, int] | None, is_before: bool = True) -> None:
        """
        Record a line in the line sequence.
        If is_before is true, we record it in the direction of the empty regime.
        Otherwise, we record it while going away from the empty regime.

        :param coord: Coordinates of the line found or None if no line found.
        """
        if is_before:
            self._line_sequence_raw.appendleft(coord)
        else:
            self._line_sequence_raw.append(coord)
        # Recalculate the line clean sequence (remove duplicates measurements of the same line)
        self._calculate_line_sequence()

    def _calculate_line_sequence(self) -> None:
        """
        Recalculate the line sequence by removing duplicates measurements of the same line.
        And remove the None values that represent no line found.
        """
        self._line_sequence.clear()
        consecutive_lines = []
        for record in list(self._line_sequence_raw) + [None]:
            # If no line found, or we reach the end of the sequence, add the previous line coordinates
            if record is None:
                if len(consecutive_lines) == 1:
                    # Add the previous line coordinates based on the only measurement
                    self._line_sequence.append(consecutive_lines[0])
                elif len(consecutive_lines) > 1:
                    # Add the previous line coordinates, calculated the center between the first and last measurement
                    x_1, y_1 = consecutive_lines[0]
                    x_2, y_2 = consecutive_lines[-1]
                    self._line_sequence.append((round((x_1 + x_2) / 2), round((y_1 + y_2) / 2)))
                consecutive_lines.clear()
            else:
                consecutive_lines.append(record)

    def _distance_from_last_line(self, is_before: bool = True) -> float:
        """
        Compute the distance between the current position and the last line found in a specific direction.
        :param is_before: If True, compute the distance from the last line found in the direction of the empty regime.
            Otherwise, compute the distance from the last line found while going away from the empty regime.
        :return: Distance in pixels.
        """
        last_line_coord = self._line_sequence[0 if is_before else -1]
        # TODO make it more accurate by using the distance between current position and the line instead of the distance
        #  between the two points
        return sqrt((self.x - last_line_coord[0]) ** 2 + (self.y - last_line_coord[1]) ** 2)

    def _get_leftmost_line_coord_str(self) -> str:
        """
        :return: Leftmost coordinates with volt conversion.
        """
        if len(self._line_sequence) == 0:
            return 'None'

        x, y = self._line_sequence[0]
        x_volt, y_volt = self.diagram.coord_to_voltage(x, y)

        return f'{x_volt:.2f}V,{y_volt:.2f}V'

    def _get_line_distance_str(self) -> str:
        """
        :return: Average distance between lines as a string with volt conversion.
        """
        if len(self._line_sequence) == 0:
            return 'None'

        return f'{self._line_avg_distance * settings.pixel_size:.3f}V'

    def reset_procedure(self):
        super().reset_procedure()

        # TODO move in settings
        if settings.research_group == 'michel_pioro_ladriere':
            self._line_slope = 75  # Prior assumption about line direction
            self._line_distances_prior = 30  # Prior assumption about distance between lines (5 patches)

        elif settings.research_group == 'louis_gaudreau':
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances_prior = 16  # Prior assumption about distance between lines (4 patches)

        elif settings.research_group == 'eva_dupont_ferrier':
            self._line_slope = -10  # Prior assumption about line direction
            self._line_distances_prior = 30  # Prior assumption about distance between lines (5 patches)

        else:
            logger.warning(f'No prior knowledge defined for the dataset: {settings.research_group}')
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances_prior = 4  # Prior assumption about distance between lines

        self._line_sequence = []
        self._line_sequence_raw = deque()
