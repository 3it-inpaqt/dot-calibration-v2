from typing import List, Optional, Tuple

from math import atan2, ceil, cos, pi, radians, sin, tan

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.data_structures import Direction, SearchLineSlope
from utils.logger import logger
from utils.settings import settings


class Jump(AutotuningProcedure):
    ##########################
    ##    Case single dot   ##
    ##########################

    # Exploration limits
    _max_steps_exploration: int = 1000  # Nb of step
    _max_steps_search_empty: int = 100  # Nb of step
    _max_line_explore_right: int = 5  # Nb detected lines
    _max_steps_validate_left_line: int = 250  # Nb steps
    _max_nb_line_leftmost: int = 4
    _nb_line_found: int = 0
    _max_steps_align: int = 20
    # Line angle degree (0 = horizontal | 90 = vertical | 45 = slope -1 | 135 = slope 1)
    _line_slope: float = None
    # List of distance between lines in pixel
    _line_distances: List[int] = None
    # Coordinate of the leftmost  and bottommost line found so far
    _leftmost_line_coord: Optional[Tuple[int, int]] = None
    _bottommost_line_coord: Optional[Tuple[int, int]] = None

    ######################
    ##   Case N dots    ##
    ######################

    # Exploration limits
    _nb_line_found_1: int = 0
    _nb_line_found_2: int = 0
    # Line angle degree (0 = horizontal | 90 = vertical | 45 = slope -1 | 135 = slope 1)
    _line_slope_1: float = None
    _line_slope_2: float = None
    # List of distance between lines in pixel
    _line_distances_1: List[int] = None
    _line_distances_2: List[int] = None
    # Line state
    _previous_line_state: int = 0

    def _tune(self) -> Tuple[int, int]:
        # Search first line, if none found return a random position
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

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
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
            self._nb_line_found = 1
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
                    self._nb_line_found = 1
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
        step_distance = round(self._default_step_y * self._get_avg_line_step_distance())
        # Start angle base on prior knowledge
        start_angle = round(self._line_slope)

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        max_angle_search = 65  # Max search angle on both side of the prior knowledge
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

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost by default.
        self._leftmost_line_coord = self.x, self.y

        nb_search_steps = 0

        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_max_down_left)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_max_up_right)
        directions = (left, right)
        left.no_line_count = 0
        right.no_line_count = 0

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions):
            for direction in (d for d in directions if not d.is_stuck):
                avg_line_distance = self._get_avg_line_step_distance()
                self._step_descr = f'line slope: {self._line_slope:.0f}°\navg line dist: {avg_line_distance:.1f}\n' \
                                   f'nb line found: {self._nb_line_found}\n' \
                                   f'leftmost line: {self._get_leftmost_line_coord_str()}'
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_detected = self._is_confirmed_line()  # Check line and save position if leftmost one

                # If new line detected, save distance and reset counter
                if line_detected:
                    if direction.no_line_count >= 1:
                        self._nb_line_found += 1
                        self._line_distances.append(direction.no_line_count)
                        # Stop exploring right if we found enough lines
                        right.is_stuck = right.is_stuck or self._nb_line_found >= self._max_line_explore_right
                    direction.no_line_count = 0
                # If no line detected since long time compare to the average distance between line, stop to go in this
                # direction
                else:
                    direction.no_line_count += 1
                    # Stop to explore this direction if we found more than 1 line and we found no line in 3x the average
                    # line distance in this direction.
                    # TODO could also use the line distance std
                    if self._nb_line_found > 1 and direction.no_line_count > 6 * avg_line_distance:
                        direction.is_stuck = True

    def validate_left_line(self) -> None:
        """
        Validate that the current leftmost line detected is really the leftmost one.
        Try to find a line left by scanning area at regular interval on the left, where we could find a line.
        If a new line is found that way, do the validation again.
        """
        logger.debug(f'Stage ({4 if settings.auto_detect_slope else 3}) - Validate left line')
        self._step_name = f'{4 if settings.auto_detect_slope else 3}. Validate leftmost line'
        line_step_distance = self._get_avg_line_step_distance()

        # Go up and down following the line
        up = Direction(last_x=0, last_y=0, move=self._move_up_follow_line, check_stuck=self.is_max_up_or_left)
        down = Direction(last_x=0, last_y=0, move=self._move_down_follow_line, check_stuck=self.is_max_down_or_right)

        nb_steps = 0
        new_line_found = True
        start_point = self._leftmost_line_coord
        while new_line_found:
            nb_line = 0
            new_line_found = False
            # Both direction start at the leftmost point
            up.last_x, up.last_y = start_point
            down.last_x, down.last_y = start_point
            up.is_stuck = down.is_stuck = False  # Unstuck since we are stating at a new location
            while not new_line_found and not Direction.all_stuck((up, down)):
                for direction in (d for d in (up, down) if not d.is_stuck):
                    nb_line += 1
                    # Check the number of repetition needed to find a line on the left or on the right
                    if nb_line > self._max_nb_line_leftmost:
                        return
                    self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                    # Step distance relative to the line distance
                    direction.move(round(self._default_step_y * line_step_distance * 2.5))
                    direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                    direction.is_stuck = direction.check_stuck()
                    if direction.is_stuck:
                        break  # We don't scan if we have reached the border

                    # Skip 1.25 line distance left
                    self._move_left_perpendicular_to_line(ceil(self._default_step_x * line_step_distance * 0.5))

                    # Go left for 2x the line distance (total 2.5x the line distance)
                    for i in range(ceil(line_step_distance * 3.5)):
                        nb_steps += 1
                        # If new line found and this is the new leftmost one, start again the checking loop
                        if self._is_confirmed_line() and start_point != self._leftmost_line_coord:
                            self._nb_line_found += 1
                            new_line_found = True
                            start_point = self._leftmost_line_coord
                            self._step_descr = f'line slope: {self._line_slope:.0f}°\n' \
                                               f'avg line dist: {line_step_distance:.1f}\n' \
                                               f'nb line found: {self._nb_line_found}\n' \
                                               f'leftmost line: {self._get_leftmost_line_coord_str()}'
                            break
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
        self.move_to_coord(x, y)
        self._move_right_perpendicular_to_line(ceil(self._default_step_x *
                                                    (self._get_avg_line_step_distance() / 4 + 1)))

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
        if line_detected and (self._leftmost_line_coord is None or self._is_left_relative_to_line()):
            self._leftmost_line_coord = self.x, self.y

        return line_detected

    def _is_left_relative_to_line(self) -> bool:
        """
        Check if the current position is at the left of the leftmost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new leftmost point.
        """
        # Select the angle of the corresponding line
        if not self._previous_line_state:  # Case for single dot
            line_slope = self._line_slope
        else:  # Case N dots
            line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2

        x, y = self._leftmost_line_coord
        # Error margin to avoid unnecessary updates
        x -= self._default_step_x
        y -= self._default_step_y

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if line_slope == 90:
            return self.x < x

        # Reconstruct line equation (y = m*x + b)
        m = tan(radians(-line_slope))  # Inverted angle because the setup is wierd
        b = y - (x * m)

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = m * self.x + b
        y_delta = y_line - self.y
        return (y_delta > 0 > m) or (y_delta < 0 < m)

    def _is_bottom_relative_to_line(self) -> bool:
        """
        Check if the current position is at the bottom of the bottommost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new bottommost point.
        """
        # Select the angle of the corresponding line
        if not self._previous_line_state:  # Case for single dot
            line_slope = self._line_slope
        else:  # Case N dots
            line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2

        x, y = self._bottommost_line_coord
        # Error margin to avoid unnecessary updates
        x -= self._default_step_x
        y -= self._default_step_y

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if line_slope == 90:
            return self.x < x

        # Reconstruct line equation (y = m*x + b)
        m = tan(radians(-line_slope))  # Inverted angle because the setup is wierd
        b = y - (x * m)

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = m * self.x + b
        y_delta = y_line - self.y
        return y_delta > 0

    def _move_relative_to_line(self, angle: float, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates in a direction relative to the estimated lines directions.

        :param angle: The direction relative to the line (in degree).
        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        init_x, init_y = self.x, self.y

        # Compute distance
        distance_x = step_size if step_size is not None else self._default_step_x
        distance_y = step_size if step_size is not None else self._default_step_y

        # Convert angle from degree to radian
        angle = angle * (pi / 180)

        self.x = round(self.x + (distance_x * cos(angle)))
        self.y = round(self.y + (distance_y * sin(angle)))

        # Enforce boundary now to avoid very small steps in some cases
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

    def _move_left_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_relative_to_line

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        if not self._previous_line_state:  # Case single dot
            self._move_relative_to_line(270 - self._line_slope, step_size)
        else:  # Case N dots
            self._move_relative_to_line(270 - self._line_slope_1 if self._previous_line_state == 1
                                        else 270 - self._line_slope_2, step_size)

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        if not self._previous_line_state:  # Case single dot
            self._move_relative_to_line(90 - self._line_slope, step_size)
        else:  # Case N dots
            self._move_relative_to_line(90 - self._line_slope_1 if self._previous_line_state == 1
                                        else 90 - self._line_slope_2, step_size)

    def _move_up_follow_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        if not self._previous_line_state:  # Case single dot
            self._move_relative_to_line(180 - self._line_slope, step_size)
        else:  # Case N dots
            self._move_relative_to_line(180 - self._line_slope_1 if self._previous_line_state == 1
                                        else 180 - self._line_slope_2, step_size)

    def _move_down_follow_line(self, step_size: Optional[int] = None, line_state: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        if not self._previous_line_state:  # Case single dot
            self._move_relative_to_line(-self._line_slope, step_size)
        else:  # Case N dots
            self._move_relative_to_line(-self._line_slope_1 if self._previous_line_state == 1
                                        else -self._line_slope_2, step_size)

    def _get_avg_line_step_distance(self, line_distances: Optional[list] = None) -> float:
        """ Get the mean line distance in number of steps. """
        if not line_distances:
            return sum(self._line_distances) / len(self._line_distances)
        else:
            return sum(line_distances) / len(line_distances)

    def _get_leftmost_line_coord_str(self) -> str:
        """
        :return: Leftmost coordinates with volt conversion.
        """
        if self._leftmost_line_coord is None:
            return 'None'

        return 'FIXME'

        # FIXME volt conversion
        # x, y = self._leftmost_line_coord
        # x_volt = self.diagram.x_axes[x]
        # y_volt = self.diagram.y_axes[y]
        #
        # return f'{x_volt:.2f}V,{y_volt:.2f}V'

    def _get_bottommost_line_coord_str(self) -> str:
        """
        :return: Leftmost coordinates with volt conversion.
        """
        if self._bottommost_line_coord is None:
            return 'None'

        return 'FIXME'

        # FIXME volt conversion
        # x, y = self._leftmost_line_coord
        # x_volt = self.diagram.x_axes[x]
        # y_volt = self.diagram.y_axes[y]
        #
        # return f'{x_volt:.2f}V,{y_volt:.2f}V'

    def reset_procedure(self):
        super().reset_procedure()

        if settings.research_group == 'michel_pioro_ladriere':
            self._line_slope = 75  # Prior assumption about line direction
            self._line_distances = [5]  # Prior assumption about distance between lines

        elif settings.research_group == 'louis_gaudreau':
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances = [3]  # Prior assumption about distance between lines

        elif settings.research_group == 'eva_dupont_ferrier_gen3':
            self._line_slope = 10  # Prior assumption about line direction
            self._line_distances = [3]  # Prior assumption about distance between lines
        elif settings.research_group == 'eva_dupont_ferrier':
            if settings.dot_number == 1:
                self._line_slope = 10
                self._line_distances = [3]
            else:
                # Prior assumption about line direction
                self._line_slope_1 = None
                self._line_slope_2 = None
                # Prior assumption about distance between lines
                self._line_distances_1 = [2]
                self._line_distances_2 = [2]
        else:
            logger.warning(f'No prior knowledge defined for the dataset: {settings.research_group}')
            if settings.dot_number == 1:
                # Prior assumption about line direction
                self._line_slope = 45
                # Prior assumption about distance between lines
                self._line_distances = [4]
            else:
                # Prior assumption about line direction
                self._line_slope_1 = 45
                self._line_slope_2 = 45
                # Prior assumption about distance between lines
                self._line_distances_1 = [4]
                self._line_distances_2 = [4]

        self._bottommost_line_coord = None
        self._leftmost_line_coord = None
        self._previous_line_state = 0

    ###################################################################################################################

    def _tune_Ndots(self) -> Tuple[int, int]:
        """
        Tuning for Ndots

        :return: Final state
        """

        # TODO complete program for Ndot, for now work with 2 dots

        # Stage (1) - First line detection
        # Search a line or a crosspoint, if none found return a random position
        logger.debug('Stage (1) - Search First line')
        self._step_name = 'Stage (1) - Search First line'
        line_state = self._search_first_line_Ndots()
        # Set _previous_line_state with line state

        if line_state == 0:
            return self.diagram.get_random_starting_point()

        # Stage (1bis) - Optional -  Case of first line is a crosspoint or slope calculation
        # If the first line is a crosspoint, we search around the crosspoint a line,
        # else we calculate the slope of the line

        elif line_state == settings.dot_number + 1:  # Crosspoint
            logger.debug('Stage (1bis) - Case of first line is a crosspoint')
            self._step_name = 'Stage (1bis) - Case of first line is a crosspoint'
            if not self._line_around_crosspoint():
                # If we are stuck inside the crosspoint, we return the coord of the crosspoint
                return self.x, self.y
            # Set _previous_line_state with line state
        # Line 1 or 2
        logger.debug(f'Stage (1bis) - Slope calculation for line {self._previous_line_state}')
        self._step_name = f'Stage (1bis) - Slope calculation for line {self._previous_line_state}'
        self._search_line_slope_Ndots()
        self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)

        # Stage (2) - Search empty area
        # With the slope of the first line, we will move perpendicular to the line to find the zero-electron regime
        logger.debug(f'Stage (2) - Search empty for {settings.dot_number} dots')
        self._step_name = '2. Search 0 e-'
        if not self._search_empty_Ndot():
            return self.x, self.y

        # Stage (Final) - Search 1 electron regime
        # Move to one electron coordinates based on the leftmost line found
        logger.debug(f'Final stage - Get {str(tuple([1] * settings.dot_number))} area')
        self._step_name = f'Final stage - Get {str(tuple([1] * settings.dot_number))} area'
        self._guess_one_electron_Ndots()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return self.get_patch_center()

    def _is_bottommost_or_leftmost_line(self, line_state: int) -> None:
        """
        Check if the current position should be considered the leftmost line or the bottommost line.
        """
        if line_state == 1 \
                and (self._bottommost_line_coord is None or self._is_bottom_relative_to_line()):
            self._bottommost_line_coord = self.x, self.y
        elif line_state == 2 \
                and (self._leftmost_line_coord is None or self._is_left_relative_to_line()):
            self._leftmost_line_coord = self.x, self.y

    def _search_first_line_Ndots(self) -> Tuple[int]:
        """
        Search any line from the tuning starting point by exploring 4 directions.

        :return: The line state, 0 if we reach the step limit without detecting a line or crosspoint,
        and 1,2 or 3 for line 1, line 2 and crosspoint.
        """

        # First scan at the start position
        # Infer with the model at the current position
        line_state, _ = self.is_transition_line()  # We don't need the confidence

        if line_state != 0:
            self._nb_line_found_1 += 1 if line_state == 1 else 0
            self._nb_line_found_2 += 1 if line_state == 2 else 0
            self._previous_line_state = line_state
            return line_state

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

                line_state, _ = self.is_transition_line()

                if line_state != 0:
                    self._nb_line_found_1 += 1 if line_state == 1 else 0
                    self._nb_line_found_2 += 1 if line_state == 2 else 0
                    self._previous_line_state = line_state
                    return line_state

        return 0

    def _line_around_crosspoint(self, target_line: Optional[int] = None) -> bool:
        """
        We search the nearest line around the crossbar
        :target_line: Optional - if defined, we search the corresponding line around the crossbar
        :return: True if we find a line, else False
        """
        start_x, start_y = self.x, self.y
        start_angle = 10

        # Step distance relative to the line distance to reduce the risk to reach another line too far away
        line_distance = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        step_distance = round(self._default_step_y * self._get_avg_line_step_distance(line_distance) / 2)

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        max_angle_search = 180 / settings.dot_number * 0.75  # Max search angle on both side of the prior knowledge
        search_step = 8
        # Scan top and bottom with a specific angle range
        delta = 0
        while abs(delta) < max_angle_search:
            for side, search in zip((180, 0), searches):
                self.move_to_coord(start_x, start_y)
                self._move_relative_to_line(side - start_angle + delta, step_distance)
                self._step_descr = f'delta: {delta}°'

                line_state, _ = self.is_transition_line()

                if line_state != 0 and line_state != settings.dot_number + 1:
                    if target_line and line_state != target_line:
                        continue
                        # We find a line
                    self._previous_line_state = line_state
                    return True
            delta += search_step
        # We don't find a line
        return False

    def _search_line_slope_Ndots(self) -> None:
        """
        Estimate the direction of the current line and update the _line_slope attribut if the measurement looks valid.
        """
        # Step distance relative to the line distance to reduce the risk to reach another line
        line_distance = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        step_distance = round(
            self._default_step_y * self._get_avg_line_step_distance(line_distances=line_distance) / 1.5)
        # Start angle base on prior knowledge
        start_angle = 10

        start_x, start_y = self.x, self.y

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        max_angle_search = 65  # Max search angle on both side of the prior knowledge
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

                line_state, _ = self.is_transition_line()  # No confidence validation here
                line_detected = True if line_state == self._previous_line_state else False

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

            self._line_slope_1 = slope_estimation if self._previous_line_state == 1 else self._line_slope_1
            self._line_slope_2 = slope_estimation if self._previous_line_state == 2 else self._line_slope_2
        else:
            self._line_slope_1 = 10 if self._previous_line_state == 1 else self._line_slope_1
            self._line_slope_2 = 100 if self._previous_line_state == 2 else self._line_slope_2
        # Reset to starting point
        self.move_to_coord(start_x, start_y)
        logger.debug(f'Calculation finish for line {self._previous_line_state}')

    def _search_empty_Ndot(self) -> bool:
        """
        Explore the diagram by scanning patch.
        We search the 0 electron regime for each line
        """

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost and the bottommost by default.

        self._bottommost_line_coord = self.x, self.y
        self._leftmost_line_coord = self.x, self.y
        stage = 1

        logger.debug(f'Start research with line {self._previous_line_state} at a coord ({self._leftmost_line_coord})')

        logger.debug(f'Stage 2.{stage} - Find empty area for line {self._previous_line_state}: '
                     f'{(1, 0) if self._previous_line_state == 1 else (0, 1)}')
        self._step_name = f'Stage 2.{stage} - Find empty area for line {self._previous_line_state}'

        # We are on a line, so we search the empty charge area of this line
        directions = self._get_direction(line_state=self._previous_line_state)
        self._get_empty_area(directions=directions)  # Get 0 electron regime for the first line

        # Optional step: make sure we found the leftmost or bottommost line
        if settings.validate_line:
            stage += 1
            logger.debug(
                f'Stage 2.{stage} - Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line')
            self._step_name = f'2.1 Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line'
            self.validate_line_Ndots(directions=directions)

        # We change the target empty area
        stage += 1
        logger.debug(f'Stage 2.{stage} - Target line change '
                     f'{self._previous_line_state} -> {[2, 1][self._previous_line_state - 1]}')
        self._step_name = f'Stage 2.{stage} - Target line change'
        # Change line: 1->2 and 2->1
        if not self._find_other_line():
            # We fail to find the other line, so we return the bottommost y and the leftmost x
            logger.debug(f'Fail to find the line {[2, 1][self._previous_line_state - 1]}')
            self.x = self._leftmost_line_coord[0]
            self.y = self._bottommost_line_coord[1]
            return False

        stage += 1
        logger.debug(f'Stage 2.{stage} - Find empty area for line {self._previous_line_state}: '
                     f'{(1, 0) if self._previous_line_state == 1 else (0, 1)}')
        self._step_name = f'Stage 2.{stage} - Find empty area for line {self._previous_line_state}'
        # We are on a line, so we search the empty charge area of this line
        directions = self._get_direction(line_state=self._previous_line_state)
        self._get_empty_area(directions=directions)  # Get 0 electron regime for the first line

        # Optional step: make sure we found the leftmost or bottommost line
        if settings.validate_line:
            stage += 1
            logger.debug(
                f'Stage 2.{stage} - Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line')
            self._step_name = f'2.{stage} Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line'
            self.validate_line_Ndots(directions=directions)
        return True

    def _get_empty_area(self, directions: tuple) -> None:
        """
        Search the area of empty charge for the horizontal or vertical state (0,:) or (:,0)
        :param directions: direction left, right, up, down of the line for a specific line (line 1 or line 2)
        """

        nb_line_found = self._nb_line_found_1 if self._previous_line_state == 1 else self._nb_line_found_2
        line_distances = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        # The line slope of the target empty area
        target_line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
        # The other slope
        other_line_slope = self._line_slope_2 if self._previous_line_state == 1 else self._line_slope_1

        directions[0].no_line_count = 0  # Direction Left
        directions[1].no_line_count = 0  # Direction right
        directions[2].no_line_count = 0  # Direction up
        directions[3].no_line_count = 0  # Direction down

        # Check if we already have the slope of the line
        if not target_line_slope and \
                not (self._previous_line_state == settings.dot_number + 1 or self._previous_line_state == 0):
            # We search the slope of the other line
            logger.debug(f'Stage (bis) - Slope calculation for line {self._previous_line_state}')
            self._step_name = f'Stage (bis) - Slope calculation for line {self._previous_line_state}'
            self._search_line_slope_Ndots()
            target_line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2

        nb_search_steps = 0

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions[:2]):
            for direction in (d for d in directions[:2] if not d.is_stuck):
                avg_line_distance = self._get_avg_line_step_distance(line_distances)
                a = 1
                self._step_descr = f'init line: line {self._previous_line_state}\n' \
                                   f'line slope: {str(target_line_slope)}°\n' \
                                   f'avg line dist: {avg_line_distance:.1f}\n' \
                                   f'nb line found: {nb_line_found}\n' \
                                   f'leftmost line: {str(self._get_leftmost_line_coord_str())}\n' \
                                   f'bottommost line: {str(self._get_bottommost_line_coord_str())}'
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                # Check line
                line_state, _ = self.is_transition_line()

                # Case of no line or the other line
                # If no line detected since long time compare to the average distance between line,
                # stop to go in this direction
                if line_state not in (self._previous_line_state, settings.dot_number + 1):
                    if line_state != 0:  # Case other line
                        # Check if we already have the slope of the line
                        if not other_line_slope and line_state != settings.dot_number + 1:
                            # We search the slope of the other line
                            self._previous_line_state = [2, 1][self._previous_line_state - 1]
                            logger.debug(f'Stage (1bis) - Slope calculation for line {self._previous_line_state}')
                            self._step_name = f'Stage (1bis) - Slope calculation for line {self._previous_line_state}'
                            self._search_line_slope_Ndots()
                            other_line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                            # Reset previous line state
                            self._previous_line_state = [2, 1][self._previous_line_state - 1]

                        self._is_bottommost_or_leftmost_line(line_state=line_state)

                    direction.no_line_count += 1
                    # Stop to explore this direction if we found more than 1 line and
                    # we found no line in 2x the average line distance in this direction.
                    # TODO could also use the line distance std
                    if nb_line_found > 1 and direction.no_line_count > 2 * avg_line_distance:
                        direction.is_stuck = True

                else:  # Case target line or crosspoint

                    if line_state == settings.dot_number + 1:  # Case of a crosspoint

                        # Check if we already have the slope of the line
                        if not other_line_slope and line_state != settings.dot_number + 1:
                            # We search the slope of the other line
                            self._previous_line_state = [2, 1][self._previous_line_state - 1]
                            logger.debug(f'Stage (1bis) - Slope calculation for line {self._previous_line_state}')
                            self._step_name = f'Stage (1bis) - Slope calculation for line {self._previous_line_state}'
                            self._search_line_slope_Ndots()
                            other_line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                            # Reset previous line state
                            self._previous_line_state = [2, 1][self._previous_line_state - 1]

                        if not self._slide_around_crosspoint(directions=directions):
                            # Stuck
                            self._is_bottommost_or_leftmost_line(line_state=1)
                            self._is_bottommost_or_leftmost_line(line_state=2)
                            return

                    # We are on a line
                    # If new specific line detected, save distance and reset counter
                    if direction.no_line_count >= 1:
                        nb_line_found += 1
                        line_distances.append(direction.no_line_count)
                        # Stop exploring right if we found enough lines
                        directions[1].is_stuck = directions[1].is_stuck \
                                                 or nb_line_found >= self._max_line_explore_right

                    self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)
                    direction.no_line_count = 0

        if self._previous_line_state == 1:
            self._line_distances_1 = line_distances
            self._nb_line_found_1 = nb_line_found
        else:
            self._line_distances_2 = line_distances
            self._nb_line_found_2 = nb_line_found

    def _slide_around_crosspoint(self, directions=tuple) -> bool:
        """
        Search the line around the crosspoint by sliding on the target line

        :param directions: direction left, right, up, down of the target line
        :return: True if we succed to find the target line
        """

        step, line_state = 0, 3

        while line_state != self._previous_line_state and step < self._max_steps_align:
            step += 1
            for vect in (d for d in directions[2:] if not d.is_stuck):
                vect.move()
                vect.last_x, vect.last_y = self.x, self.y
                vect.is_stuck = vect.check_stuck()  # Check if reach a corner

                # Check line and save position
                line_state, _ = self.is_transition_line()

                if line_state == self._previous_line_state:
                    return True
                elif line_state == [2, 1][self._previous_line_state - 1]:
                    self._is_bottommost_or_leftmost_line(line_state=line_state)

        # We are stuck
        return False

    def validate_line_Ndots(self, directions: tuple) -> None:
        """
        Validate that the current leftmost line detected is really the leftmost one.
        Try to find a line left by scanning area at regular interval on the left, where we could find a line.
        If a new line is found that way, do the validation again.
        """

        line_distance = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        line_step_distance = self._get_avg_line_step_distance(line_distances=line_distance) * 2
        line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2

        new_line_found = True
        start_point = self._leftmost_line_coord if self._previous_line_state == 2 else self._bottommost_line_coord
        target = self._leftmost_line_coord if self._previous_line_state == 2 else self._bottommost_line_coord
        nb_steps = 0
        while new_line_found:
            nb_line = 0
            new_line_found = False
            # Both direction start at the leftmost point
            directions[2].last_x, directions[2].last_y = start_point
            directions[3].last_x, directions[3].last_y = start_point
            # Unstuck since we are stating at a new location
            directions[2].is_stuck = directions[3].is_stuck = False
            while not new_line_found and not Direction.all_stuck((directions[2], directions[3])):
                for direction in (d for d in (directions[2], directions[3]) if not d.is_stuck):
                    nb_line += 1
                    # Check the number of repetition needed to find a line on the left or on the right
                    if nb_line > self._max_nb_line_leftmost:
                        return

                    self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                    # Step distance relative to the line distance
                    direction.move(round(self._default_step_y * line_step_distance))
                    # Save current position for next time
                    direction.last_x, direction.last_y = self.x, self.y
                    direction.is_stuck = direction.check_stuck()
                    if direction.is_stuck:
                        break  # We don't scan if we have reached the border

                    # Skip 1.25 line distance left
                    self._move_left_perpendicular_to_line(ceil(self._default_step_x * line_step_distance * 0.5))

                    # Go left for 2x the line distance (total 2.5x the line distance)
                    for i in range(ceil(line_step_distance * 1.5)):
                        nb_steps += 1
                        # If new line found and this is the new leftmost one, start again the checking loop
                        line_state, _ = self.is_transition_line()

                        if line_state == self._previous_line_state and start_point != target:
                            self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)

                            self._nb_line_found_1 += 1 if self._previous_line_state == 1 else 0
                            self._nb_line_found_2 += 1 if self._previous_line_state == 2 else 0

                            new_line_found = True
                            line_found = self._nb_line_found_1 if self._previous_line_state == 1 \
                                else self._nb_line_found_2
                            start_point = self._leftmost_line_coord if self._previous_line_state == 2 \
                                else self._bottommost_line_coord

                            self._step_descr = f'init line: line {self._previous_line_state}' \
                                               f'line slope: {line_slope:.0f}°\n' \
                                               f'avg line dist: {line_step_distance:.1f}\n' \
                                               f'nb line found: {line_found}\n' \
                                               f'leftmost line: {self._get_leftmost_line_coord_str()}' \
                                               f'bottommost line: {self._get_bottommost_line_coord_str()}'
                            break

                        self._move_left_perpendicular_to_line()

                        if self.is_max_left() or self.is_max_down():
                            break  # Nothing else to see here

                    if nb_steps > self._max_steps_validate_left_line:
                        return  # Hard break to avoid infinite search in case of bad slope detection (>90°)

                    if new_line_found:
                        break

    def _find_other_line(self) -> bool:
        """
        Search the other line

        :line:
        :return: True if we find the other line, else False
        """

        target_coord = [self._leftmost_line_coord, self._bottommost_line_coord][self._previous_line_state - 1]
        target_line = [2, 1][self._previous_line_state - 1]

        # We already find the other line
        if target_coord:
            start_x, start_y = target_coord
            self.move_to_coord(start_x, start_y)
            self._previous_line_state = [2, 1][self._previous_line_state - 1]
            line_slope = self._line_slope_1 if self._previous_line_state == 2 else self._line_slope_1
            if not line_slope:
                self._search_line_slope_Ndots()
            return True

        # Start the research
        start_x, start_y = self._bottommost_line_coord if self._previous_line_state == 1 else self._leftmost_line_coord

        self.move_to_coord(start_x, start_y)

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

                line_state, _ = self.is_transition_line()

                if line_state == target_line:
                    self._nb_line_found_1 += 1 if line_state == 1 else 0
                    self._nb_line_found_2 += 1 if line_state == 2 else 0
                    self._previous_line_state = [2, 1][self._previous_line_state - 1]
                    line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                    if not line_slope:
                        self._search_line_slope_Ndots()

                    return True
                # Crosspoint
                if line_state == settings.dot_number + 1 and self._line_around_crosspoint(target_line=target_line):
                    line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                    if not line_slope:
                        self._search_line_slope_Ndots()
                    self._nb_line_found_1 += 1 if line_state == 1 else 0
                    self._nb_line_found_2 += 1 if line_state == 2 else 0
                    return True

        return False

    def _get_direction(self, line_state: Optional[int] = None) -> tuple:
        """
        Get the direction left, right, up, down of the state line

        :line_state: The status of the line from which we will extract the directions
        :return: the directions of the lines
        """
        if line_state:
            previous_line_state = self._previous_line_state
            self._previous_line_state = line_state

        # Define direction for horizontal line
        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_max_down_left)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_max_up_right)
        up = Direction(last_x=self.x, last_y=self.y, move=self._move_up_follow_line,
                       check_stuck=self.is_max_up_right)
        down = Direction(last_x=self.x, last_y=self.y, move=self._move_down_follow_line,
                         check_stuck=self.is_max_up_right)
        if line_state:
            self._previous_line_state = previous_line_state

        return left, right, up, down

    def _guess_one_electron_Ndots(self) -> None:
        """
        According to the leftmost line validated and to the bottommost line validated,
         guess a good location for the 1 electron regime.
        Then move to this location.
        """

        # If no line found, desperately guess random position as last resort
        x, y = None, None
        if self._leftmost_line_coord is None:
            x = self.diagram.get_random_starting_point()[0]

        if self._bottommost_line_coord is None:
            y = self.diagram.get_random_starting_point()[1]

        if x or y:
            self.move_to_coord(x, y)
            self._enforce_boundary_policy(force=True)
            return

        # Search 1e area
        x_l, y_l = self._leftmost_line_coord
        x_b, y_b = self._bottommost_line_coord

        x, y = x_l, y_b

        self.move_to_coord(x, y)

        self._previous_line_state = 1
        self._move_right_perpendicular_to_line(ceil(self._default_step_x *
                                                    (self._get_avg_line_step_distance(self._line_distances_1) / 4 + 1)))
        self._previous_line_state = 2
        self._move_right_perpendicular_to_line(ceil(self._default_step_x *
                                                    (self._get_avg_line_step_distance(self._line_distances_2) / 4 + 1)))

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
