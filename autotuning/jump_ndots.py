from math import atan2, ceil, pi, radians, tan
from time import perf_counter
from typing import List, Optional, Tuple

import torch

from autotuning.jump import Jump
from classes.data_structures import BoundaryPolicy, StepHistoryEntry
from classes.data_structures import Direction, SearchLineSlope
from datasets.diagram_online import DiagramOnline
from datasets.qdsd import QDSDLines
from utils.logger import logger
from utils.settings import settings


class JumpNDots(Jump):
    """
    Same as Jump but for N Dots
    #TODO Adapt for N class (work for 2 dots for now)
    """
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
    # Parameters
    _max_nb_line_leftmost: int = 4
    _max_nb_line_bottommost: int = 4
    _max_steps_validate_line: int = 100  # Nb steps
    # Class
    _class = QDSDLines.classes

    _bottommost_line_coord = None

    def default_slope(self, line_state: int) -> int:
        if line_state == 1:
            return self._line_slope_default_1
        else:
            return self._line_slope_default_2

    def reset_procedure(self):
        super().reset_procedure()

        self._bottommost_line_coord = None
        self._leftmost_line_coord = None
        self._previous_line_state = 0

        if settings.research_group == 'eva_dupont_ferrier':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 2
            self._line_slope_default_2 = 92
            # Prior assumption about distance between lines
            self._line_distances_1 = [3]
            self._line_distances_2 = [6]
        elif settings.research_group == 'louis_gaudreau':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 45
            self._line_slope_default_2 = 80
            # Prior assumption about distance between lines
            self._line_distances_1 = [14]
            self._line_distances_2 = [12]
        else:
            logger.warning(f'No prior knowledge defined for the dataset: {settings.research_group}')
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 45
            self._line_slope_default_2 = 135
            # Prior assumption about distance between lines
            self._line_distances_1 = [5]
            self._line_distances_2 = [5]

    #### Rewrite function ####

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        self._move_relative_to_line(90 - self._line_slope_1
                                    if self._previous_line_state == 1 else 90 - self._line_slope_2, step_size)

    def _move_left_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_relative_to_line

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        self._move_relative_to_line(270 - self._line_slope_1
                                    if self._previous_line_state == 1 else 270 - self._line_slope_2, step_size)

    def _move_up_follow_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        self._move_relative_to_line(180 - self._line_slope_1
                                    if self._previous_line_state == 1 else 180 - self._line_slope_2, step_size)

    def _move_down_follow_line(self, step_size: Optional[int] = None, line_state: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """

        self._move_relative_to_line(-self._line_slope_1
                                    if self._previous_line_state == 1 else -self._line_slope_2, step_size)

    def _get_avg_line_step_distance(self, line_distances: list = None) -> float:
        """ Get the mean line distance in number of steps. """

        return sum(line_distances) / len(line_distances)

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

    def _is_left_relative_to_line(self) -> bool:
        """
        Check if the current position is at the left of the leftmost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new leftmost point.
        """
        # Select the angle of the corresponding line
        line_slope = self._line_slope_2

        x, y = self._leftmost_line_coord
        # Error margin to avoid unnecessary updates
        x -= self._default_step_x
        y -= self._default_step_y

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if line_slope == 90:
            return self.x < x

        # Reconstruct line equation (y = m*x + b)
        m = radians(-line_slope)  # Inverted angle because the setup is wierd
        slope = tan(m)
        b = y - (x * slope)

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = slope * self.x + b
        y_delta = y_line - self.y
        return (y_delta > 0 > slope) or (y_delta < 0 < slope)

    def _is_bottom_relative_to_line(self) -> bool:
        """
        Check if the current position is at the bottom of the bottommost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new bottommost point.
        """
        # Select the angle of the corresponding line
        line_slope = self._line_slope_1
        x, y = self._bottommost_line_coord
        # Error margin to avoid unnecessary updates
        x -= self._default_step_x
        y -= self._default_step_y

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if line_slope == 0 or line_slope == 90:
            return self.y < y

        # Reconstruct line equation (y = m*x + b)
        m = radians(-line_slope)
        slope = tan(m)
        b = y - (x * slope)

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = slope * self.x + b
        y_delta = y_line - self.y
        return y_delta > 0

    def is_transition_line(self) -> Tuple[int, float]:
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """
        time_start = perf_counter()

        # Check coordinates according to the current policy.
        # They could be changed to fit inside the diagram if necessary
        self._enforce_boundary_policy()

        # Fetch ground truth from labels if available (all set to None if Online diagram)
        ground_truth, soft_truth_larger, soft_truth_smaller = self.get_ground_truths(self.x, self.y)

        result: Tuple[bool, float]
        if self.is_oracle_enable:
            # Oracle use ground truth with full confidence
            prediction = ground_truth
            confidence = 1
            time_data_processed = time_data_fetched = perf_counter()
            is_above_confidence_threshold = True
        else:
            with torch.no_grad():
                # Cut the patch area and send it to the model for inference
                patch = self.diagram.get_patch((self.x, self.y), self.patch_size)
                time_data_fetched = perf_counter()
                # Reshape as valid input for the model (batch size, patch x, patch y)
                size_x, size_y = self.patch_size
                patch = patch.view((1, size_x, size_y))
                # Send to the model for inference
                prediction, confidence = self.model.infer(patch, settings.bayesian_nb_sample_test)
                # Extract data from pytorch tensor
                prediction = QDSDLines.class_mapping(prediction)
                confidence = QDSDLines.conf_mapping(confidence, prediction)
                time_data_processed = perf_counter()

            is_above_confidence_threshold = self.model.is_above_confident_threshold(prediction, confidence)

        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr
        self._scan_history.append(StepHistoryEntry(
            (self.x, self.y), prediction, confidence, ground_truth, soft_truth_larger, soft_truth_smaller,
            is_above_confidence_threshold, step_description, time_start, time_data_fetched, time_data_processed,
            isinstance(self.diagram, DiagramOnline)
        ))
        x, y = self.diagram.coord_to_voltage(self.x, self.y)
        logger.debug(f'Patch {self.get_nb_steps():03} classified as {QDSDLines.classes[prediction]} with confidence '
                     f'{confidence:.2%} at coord ({x:.2f}, {y:.2f})')

        return prediction, confidence

    def _get_bottommost_line_coord_str(self) -> str:
        """
        :return: Bottommost coordinates with volt conversion.
        """
        if self._bottommost_line_coord is None:
            return 'None'

        x, y = self._bottommost_line_coord
        x_volt, y_volt = self.diagram.coord_to_voltage(x, y)

        return f'{x_volt:.2f}V,{y_volt:.2f}V'

    ###########################

    def _tune(self) -> Tuple[int, int]:
        """
        Tuning for Ndots

        :return: Final state
        """

        # TODO complete program for Ndot, for now work with 2 dots

        # Stage (1) - First line detection
        # Search a line or a crosspoint, if none found return a random position

        stage = 1
        substage = 1

        logger.debug(f'Stage ({stage}.{substage}) - Search First line')
        self._step_name = f'Stage ({stage}.{substage}) - Search First line'

        self._search_first_line()  # previous_line_state set line state

        if self._previous_line_state == 0:
            return self.diagram.get_random_starting_point()

        # If the first line is a crosspoint, we search around the crosspoint a line,
        # else we calculate the slope of the line

        elif self._previous_line_state == settings.dot_number + 1:  # Crosspoint
            substage += 1
            logger.debug(f'Stage ({stage}.{substage}) - Case of first line is a crosspoint')
            self._step_name = f'Stage ({stage}.{substage}) - Case of first line is a crosspoint'
            if not self._line_around_crosspoint():
                # If we are stuck inside the crosspoint, we return the coord of the crosspoint
                return self.x, self.y
            # Set _previous_line_state with line state

        # Line 1 or 2
        substage += 1

        logger.debug(f'Stage ({stage}.{substage}) - Slope calculation for {self._class[self._previous_line_state]}')
        self._step_name = f'Stage ({stage}.{substage}) - Slope calculation for {self._class[self._previous_line_state]}'
        self._search_line_slope()
        self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)

        # Stage (2) - Search empty area
        stage += 1
        substage = 1
        # With the slope of the first line, we will move perpendicular to the line to find the zero-electron regime

        logger.debug(f'Stage ({stage}.{substage}) - Search empty area for {settings.dot_number} dots')
        self._step_name = f'Stage ({stage}.{substage}) - Search {str(tuple([0] * settings.dot_number))} electron area'
        if not self._search_empty():
            return self.x, self.y

        # Stage (Final) - Search 1 electron regime
        # Move to one electron coordinates based on the leftmost line found
        logger.debug(f'Final stage - Get {str(tuple([1] * settings.dot_number))} area')
        self._step_name = f'Final stage - Get {str(tuple([1] * settings.dot_number))} area'
        self._guess_one_electron()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return self.get_patch_center()

    def _search_first_line(self) -> None:
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
            return

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
                    return None

        return None

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

                if line_state not in [0, settings.dot_number + 1]:
                    # We find a line
                    if target_line and line_state != target_line:
                        continue
                    self._previous_line_state = line_state
                    return True
            delta += search_step
        # We don't find a line
        return False

    def _search_line_slope(self) -> None:
        """
        Estimate the direction of the current line and update the _line_slope attribut if the measurement looks valid.
        """
        # Step distance relative to the line distance to reduce the risk to reach another line
        line_distance = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        step_distance = round(
            self._default_step_y * self._get_avg_line_step_distance(line_distances=line_distance) / 2)
        # Start angle base on prior knowledge
        start_angle = 5

        start_x, start_y = self.x, self.y

        # (Top search, Bottom search)
        searches = (SearchLineSlope(), SearchLineSlope())

        max_angle_search = 70  # Max search angle on both side of the prior knowledge
        search_step = 10
        # Scan top and bottom with a specific angle range
        for side, search in zip((180, 0), searches):
            init_line = False
            init_no_line = False
            delta = 0
            while abs(delta) < max_angle_search:
                self.move_to_coord(start_x, start_y)
                self._move_relative_to_line(side - start_angle + delta, step_distance)
                self._step_descr = f'Target line: {self._class[self._previous_line_state]}' \
                                   f'delta: {delta}°\n' \
                                   f'init line: {init_line}\n' \
                                   f'init no line: {init_no_line}'

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

            if self._previous_line_state == 1:
                self._line_slope_1 = slope_estimation
            else:
                self._line_slope_2 = slope_estimation

        else:
            # Doesn't find the slope
            if self._previous_line_state == 1:
                self._line_slope_1 = self._line_slope_default_1
            else:
                self._line_slope_2 = self._line_slope_default_2

        # Reset to starting point
        self.move_to_coord(start_x, start_y)
        logger.debug(f'Calculation finish for {self._class[self._previous_line_state]}')

    def _search_empty(self) -> bool:
        """
        Explore the diagram by scanning patch.
        We search the 0 electron regime for each line
        """

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost and the bottommost by default.

        if self._previous_line_state == 1:
            self._bottommost_line_coord = self.x, self.y
        else:
            self._leftmost_line_coord = self.x, self.y

        substage = 0

        logger.debug(f'Start research with {self._class[self._previous_line_state]} '
                     f'at a coord ({self._leftmost_line_coord})')

        # We are on a line, so we search the empty charge area of this line
        directions = self._get_direction(line_state=self._previous_line_state)

        line_found = True
        max_step = 100
        step = 0

        while line_found and step < max_step:
            substage += 1
            step += 1
            logger.debug(f'Stage 2.{substage} - Find empty area for {self._class[self._previous_line_state]}: '
                         f'{(1, 0) if self._previous_line_state == 1 else (0, 1)}')
            self._step_name = f'Stage 2.{substage} - Find empty area for {self._class[self._previous_line_state]}'
            # Get 0 electron regime for the first line
            self._get_empty_area(directions=directions)

            # Optional step: make sure we found the leftmost or bottommost line
            fine_tuning = [settings.validate_bottom_line, settings.validate_left_line][self._previous_line_state - 1]
            if fine_tuning:
                substage += 1
                logger.debug(
                    f'Stage 2.{substage} - Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"}'
                    f' line')
                self._step_name = f'2.1 Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line'
                line_found = self._validate_line(directions=directions)
            else:
                line_found = False

        # We change the target empty area
        logger.debug(f'Stage 3.1 - Target line change '
                     f'{self._class[self._previous_line_state]} into '
                     f'{self._class[[2, 1][self._previous_line_state - 1]]}')
        self._step_name = f'Stage 3.1 - Target line change ' \
                          f'{self._class[self._previous_line_state]} into ' \
                          f'{self._class[[2, 1][self._previous_line_state - 1]]}'

        # Change line: 1->2 or 2->1
        if not self._find_other_line():
            # We fail to find the other line, so we return the bottommost y and the leftmost x
            logger.debug(f'Fail to find {self._class[[2, 1][self._previous_line_state - 1]]}')
            self.x = self._leftmost_line_coord[0] \
                if self._leftmost_line_coord[0] else self.diagram.get_random_starting_point()[0]
            self.y = self._bottommost_line_coord[1] \
                if self._bottommost_line_coord[1] else self.diagram.get_random_starting_point()[1]
            return False

        line_found = True
        step = 0
        directions = self._get_direction(line_state=self._previous_line_state)
        substage = 0
        logger.debug(f'Stage 4 - Start research with {self._class[self._previous_line_state]} '
                     f'at a coord ({self._leftmost_line_coord})')
        while line_found and step < max_step:
            substage += 1
            step += 1
            logger.debug(f'Stage 4.{substage} - Find empty area for {self._class[self._previous_line_state]}: '
                         f'{(1, 0) if self._previous_line_state == 1 else (0, 1)}')
            self._step_name = f'Stage 4.{substage} - Find empty area for {self._class[self._previous_line_state]}'
            # Get 0 electron regime for the second line
            self._get_empty_area(directions=directions)

            # Optional step: make sure we found the leftmost or bottommost line
            fine_tuning = [settings.validate_bottom_line, settings.validate_left_line][self._previous_line_state - 1]
            if fine_tuning:
                substage += 1
                logger.debug(
                    f'Stage 4.{substage} - Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line')
                self._step_name = f'4.{substage} Validate {"leftmost" if self._previous_line_state == 2 else "bottommost"} line'
                line_found = self._validate_line(directions=directions)
            else:
                line_found = False

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

        directions[0].no_line_count = 0  # Direction Left
        directions[1].no_line_count = 0  # Direction right
        directions[2].no_line_count = 0  # Direction up
        directions[3].no_line_count = 0  # Direction down

        # Reset direction
        directions[0].is_stuck = False

        nb_search_steps = 0

        while nb_search_steps < self._max_steps_search_empty and not Direction.all_stuck(directions[:2]):
            for direction in (d for d in directions[:2] if not d.is_stuck):
                avg_line_distance = self._get_avg_line_step_distance(line_distances)
                self._step_descr = f'Target line: {self._class[self._previous_line_state]}\n' \
                                   f'Line slope: {str(target_line_slope)}°\n' \
                                   f'Avg line dist: {avg_line_distance:.1f}\n' \
                                   f'Nb line found: {nb_line_found}\n' \
                                   f'Leftmost line: {str(self._get_leftmost_line_coord_str())}\n' \
                                   f'Bottommost line: {str(self._get_bottommost_line_coord_str())}'
                nb_search_steps += 1

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                # Check line
                line_state, _ = self.is_transition_line()

                # Case of no line

                if line_state == 0:
                    direction.no_line_count += 1
                    # Stop to explore this direction if we found more than 1 line and
                    # we found no line in 2x the average line distance in this direction.
                    # TODO could also use the line distance std
                    if nb_line_found > 1 and direction.no_line_count > 2 * avg_line_distance:
                        direction.is_stuck = True

                # Case of Crosspoint

                elif line_state == settings.dot_number + 1:

                    # Check if we have the second slope
                    other_line_slope = self._line_slope_2 if self._previous_line_state == 1 else self._line_slope_1

                    if not other_line_slope:
                        self._previous_line_state = [2, 1][self._previous_line_state - 1]
                        logger.debug(
                            f'Stage (bis) - Slope calculation for {self._class[self._previous_line_state]}')
                        self._step_name = f'Stage (bis) - Slope calculation for' \
                                          f' {self._class[self._previous_line_state]}'
                        self._search_line_slope()
                        other_line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                        # Reset previous line state
                        self._previous_line_state = [2, 1][self._previous_line_state - 1]

                    self._is_bottommost_or_leftmost_line(line_state=1)
                    self._is_bottommost_or_leftmost_line(line_state=2)

                    if direction.no_line_count >= 1:
                        nb_line_found += 1
                        line_distances.append(direction.no_line_count)
                        # Stop exploring right if we found enough lines
                        directions[1].is_stuck = directions[1].is_stuck \
                                                 or nb_line_found >= self._max_line_explore_right

                    direction.no_line_count = 0

                # Case of the target line

                elif line_state == self._previous_line_state:
                    if direction.no_line_count >= 1:
                        nb_line_found += 1
                        line_distances.append(direction.no_line_count)
                        # Stop exploring right if we found enough lines
                        directions[1].is_stuck = directions[1].is_stuck \
                                                 or nb_line_found >= self._max_line_explore_right
                    self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)
                    direction.no_line_count = 0

                # Case of line but not the target line

                else:
                    # Check if we have the second slope
                    other_line_slope = self._line_slope_2 if self._previous_line_state == 1 else self._line_slope_1

                    if not other_line_slope:
                        self._previous_line_state = [2, 1][self._previous_line_state - 1]
                        logger.debug(
                            f'Stage (bis) - Slope calculation for {self._class[self._previous_line_state]}')
                        self._step_name = f'Stage (bis) - Slope calculation for' \
                                          f' {self._class[self._previous_line_state]}'
                        self._search_line_slope()
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

        step, init_line = 0, 3
        logger.debug(f'Step: slide_around_crosspoint')
        while init_line != self._previous_line_state and step < self._max_steps_align:
            step += 1
            for vect in (d for d in directions[2:] if not d.is_stuck):
                self.move_to_coord(vect.last_x, vect.last_y)
                vect.move()
                vect.last_x, vect.last_y = self.x, self.y
                vect.is_stuck = vect.check_stuck()  # Check if reach a corner

                # Check line and save position
                line_state, _ = self.is_transition_line()

                if line_state == self._previous_line_state:
                    logger.debug(f'Finish: slide_around_crosspoint')
                    return True
                elif line_state == [2, 1][self._previous_line_state - 1]:
                    self._is_bottommost_or_leftmost_line(line_state=line_state)

        # We are stuck
        logger.debug(f'Finish: slide_around_crosspoint')
        return False

    def _validate_line(self, directions: tuple) -> bool:
        """
        Validate that the current leftmost or the bottommost line detected is really the leftmost or the bottommost one.
        Try to find a line left by scanning area at regular interval on the left, where we could find a line.
        If a new line is found that way, do the validation again.
        """

        line_distance = self._line_distances_1 if self._previous_line_state == 1 else self._line_distances_2
        line_step_distance = self._get_avg_line_step_distance(line_distances=line_distance) * 2
        line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
        max_nb_line = [self._max_nb_line_bottommost, self._max_nb_line_leftmost][self._previous_line_state - 1]
        default_step = [self._default_step_y, self._default_step_x][self._previous_line_state - 1]
        default_step_inv = [self._default_step_x, self._default_step_y][self._previous_line_state - 1]
        new_line_found = True
        start_point = self._leftmost_line_coord \
            if self._previous_line_state == 2 else self._bottommost_line_coord

        # Case of start point is None
        if not start_point:
            start_point = self.diagram.get_random_starting_point()
            self._leftmost_line_coord = start_point if self._previous_line_state == 2 else self._leftmost_line_coord
            self._bottommost_line_coord = start_point if self._previous_line_state == 1 else self._bottommost_line_coord

        nb_steps = 0
        while new_line_found:
            nb_line = 1
            new_line_found = False
            # Both direction start at the leftmost/bottommost point
            directions[2].last_x, directions[2].last_y = start_point
            directions[3].last_x, directions[3].last_y = start_point
            # Unstuck since we are starting at a new location
            directions[2].is_stuck = directions[3].is_stuck = False
            while not new_line_found and not Direction.all_stuck((directions[2], directions[3])):
                for direction in (d for d in (directions[2], directions[3]) if not d.is_stuck):
                    self._step_descr = f'Target line: {self._class[self._previous_line_state]}\n' \
                                       f'Coord: {(direction.last_x, direction.last_y)}\n' \
                                       f'Leftmost line: {self._get_leftmost_line_coord_str()}\n' \
                                       f'Bottommost line: {self._get_bottommost_line_coord_str()}\n' \
                                       f'Line slope: {line_slope:.0f}°\n' \
                                       f'Avg line dist: {line_step_distance:.1f}'
                    self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                    # Step distance relative to the line distance

                    direction.move(round(default_step * line_step_distance))
                    # Save current position for next time
                    direction.last_x, direction.last_y = self.x, self.y
                    direction.is_stuck = direction.check_stuck()
                    if direction.is_stuck:
                        break  # We don't scan if we have reached the border

                    # Skip 30% line distance
                    self._move_left_perpendicular_to_line(ceil(default_step_inv * line_step_distance * 0.3))

                    # Go left for 2x the line distance (total 1.5x the line distance)
                    for i in range(ceil(line_step_distance * 1.5)):
                        nb_steps += 1
                        # If new line found and this is the new leftmost one, start again the checking loop
                        line_state, _ = self.is_transition_line()

                        if line_state == self._previous_line_state or line_state == settings.dot_number + 1:
                            coord = self._bottommost_line_coord \
                                if self._previous_line_state == 1 else self._leftmost_line_coord

                            self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)

                            new_coord = self._bottommost_line_coord \
                                if self._previous_line_state == 1 else self._leftmost_line_coord
                            # If line isn't the leftmost or bottommost: ignore
                            if coord == new_coord:
                                continue
                            else:
                                x, y = self.diagram.coord_to_voltage(coord[0], coord[1])
                                xbis, ybis = self.diagram.coord_to_voltage(self.x, self.y)
                                logger.debug(
                                    f'Previous {"leftmost" if self._previous_line_state == 2 else "bottommost"} '
                                    f'line: ({x:.2f}, {y:.2f}), After verification: ({xbis:.2f}, {ybis:.2f})')
                                direction.last_x, direction.last_y = self.x, self.y
                                self._nb_line_found_1 += 1 if self._previous_line_state == 1 else 0
                                self._nb_line_found_2 += 1 if self._previous_line_state == 2 else 0
                                new_line_found = True
                                break

                        self._move_left_perpendicular_to_line()

                        if self.is_max_left() or self.is_max_down():
                            # Nothing else to see here
                            break

                    if nb_steps > self._max_steps_validate_line:
                        # Hard break to avoid infinite search in case of bad slope detection (>90°)
                        return False

                    nb_line += 1
                    # Check the number of repetition needed to find a line on the left or on the right
                    if nb_line > max_nb_line:
                        return False
        return False

    def _find_other_line(self) -> bool:
        """
        Search the other line

        :line:
        :return: True if we find the other line, else False
        """
        substage = 1
        start_coord = [self._leftmost_line_coord, self._bottommost_line_coord][self._previous_line_state - 1]

        # We already find the other line
        if start_coord:
            start_x, start_y = start_coord
            self.move_to_coord(start_x, start_y)
            self._previous_line_state = [2, 1][self._previous_line_state - 1]
            line_slope = self._line_slope_1 if self._previous_line_state == 2 else self._line_slope_1
            if not line_slope:
                substage += 1
                logger.debug(
                    f'Stage (3.{substage}) - Slope calculation for {self._class[self._previous_line_state]}')
                self._step_name = f'Stage (3.{substage}) - ' \
                                  f'Slope calculation for {self._class[self._previous_line_state]}'
                self._search_line_slope()
            x, y = self.diagram.coord_to_voltage(start_coord[0], start_coord[1])
            logger.debug(f'{self._class[self._previous_line_state]} already found at coord ({x:.2f},{y:.2f})')
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
        target_line = [2, 1][self._previous_line_state - 1]

        while nb_exploration_steps < self._max_steps_exploration and not Direction.all_stuck(directions):

            # Move and search line in every not stuck directions
            for direction in (d for d in directions if not d.is_stuck):
                nb_exploration_steps += 1
                self._step_descr = f'Init line: {self._class[self._previous_line_state]}' \
                                   f'Target line: {self._class[target_line]}' \
                                   f'Leftmost line: {self._get_leftmost_line_coord_str()}' \
                                   f'Bottommost line: {self._get_bottommost_line_coord_str()}'

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_state, _ = self.is_transition_line()

                if line_state == target_line:
                    self._nb_line_found_1 += 1 if line_state == 1 else 0
                    self._nb_line_found_2 += 1 if line_state == 2 else 0
                    self._previous_line_state = target_line
                    line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                    if not line_slope:
                        substage += 1
                        logger.debug(
                            f'Stage (3.{substage}) - Slope calculation for {self._class[self._previous_line_state]}')
                        self._step_name = f'Stage (3.{substage}) - ' \
                                          f'Slope calculation for {self._class[self._previous_line_state]}'
                        self._search_line_slope()
                        self._is_bottommost_or_leftmost_line(line_state=self._previous_line_state)
                    return True

                # Crosspoint
                if line_state == settings.dot_number + 1:
                    substage += 1
                    logger.debug(
                        f'Stage (3.{substage}) - Check around crosspoint')
                    self._step_name = f'Stage (3.{substage}) - Check around crosspoint'

                    if self._line_around_crosspoint(target_line=target_line):
                        line_slope = self._line_slope_1 if self._previous_line_state == 1 else self._line_slope_2
                        if not line_slope:
                            substage += 1
                            logger.debug(
                                f'Stage (3.{substage}) - Slope calculation for {self._class[self._previous_line_state]}')
                            self._step_name = f'Stage (3.{substage}) - ' \
                                              f'Slope calculation for {self._class[self._previous_line_state]}'

                            self._search_line_slope()
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

    def _guess_one_electron(self) -> None:
        """
        According to the leftmost line validated and to the bottommost line validated,
         guess a good location for the 1 electron regime.
        Then move to this location.
        """

        time_start = perf_counter()

        # If no line found, desperately guess random position as last resort
        if not (self._leftmost_line_coord and self._bottommost_line_coord):
            if not self._leftmost_line_coord:
                x = self.diagram.get_random_starting_point()[0]
                self.move_to_coord(x=x)
            else:
                self.move_to_coord(x=self._leftmost_line_coord[0])
            if not self._bottommost_line_coord:
                y = self.diagram.get_random_starting_point()[1]
                self.move_to_coord(y=y)
            else:
                self.move_to_coord(x=self._bottommost_line_coord[1])
            self._enforce_boundary_policy(force=True)
            return

        # Search 1e area
        x_l, y_l = self._leftmost_line_coord
        state, x_l, y_l = self._enforce_boundary(True, x_l, y_l)
        x_b, y_b = self._bottommost_line_coord
        state, x_b, y_b = self._enforce_boundary(True, x_b, y_b)

        # Reconstruct Line 1 equation (y = m*x + b)
        m1 = radians(
            -self._line_slope_1)  # tan(radians(-self._line_slope_1))  # Inverted angle because the setup is wierd
        slope_1 = tan(m1)
        b1 = self.diagram.x_axes[y_b] - (self.diagram.x_axes[x_b] * slope_1)
        print(f'Y : {self.diagram.x_axes[y_b]}, X : {self.diagram.x_axes[x_b]}, B : {b1}')

        # Reconstruct Line 2 equation (y = m*x + b)
        m2 = radians(
            -self._line_slope_2)  # tan(radians(-self._line_slope_2))  # Inverted angle because the setup is wierd
        slope_2 = tan(m2)
        b2 = self.diagram.x_axes[y_l] - (self.diagram.x_axes[x_l] * slope_2)
        print(f'Y : {self.diagram.x_axes[y_l]}, X : {self.diagram.x_axes[x_l]}, B : {b2}')

        x = int((b2 - b1) / (slope_1 - slope_2))
        y = int(slope_1 * x + b1)
        state, x, y = self._enforce_boundary(True, x, y)

        x_volt_l = self.diagram.x_axes[x_l]
        y_volt_b = self.diagram.y_axes[y_b]
        x_volt = self.diagram.x_axes[x]
        y_volt = self.diagram.y_axes[y]
        import numpy as np
        ang1 = m1 * 180 / np.pi
        ang2 = m2 * 180 / np.pi
        logger.debug(f'- Stage Final - \nIntersection point: ({x_volt:.2f}V,{y_volt:.2f}V)\n'
                     f'Leftmost coord: {(x_l, y_l)} ->  {(self.diagram.x_axes[x_l], self.diagram.y_axes[y_l])}\n'
                     f'Bottommost coord: {(x_b, y_b)} -> {(self.diagram.x_axes[x_b], self.diagram.y_axes[y_b])}\n'
                     f'Angle Line 1: {m1} ou {ang1 if not ang1 < 0 else ang1 + 360}\n'
                     f'Angle Line 2: {m2} ou {ang2 if not ang2 < 0 else ang2 + 360}\n'
                     f'Y1 = {slope_1}.x + {b1}\n'
                     f'Y2 = {slope_2}.x + {b2}')

        self.move_to_coord(x, y)

        # Record Intersection
        self._step_descr = f'---- Intersection point -------\n' \
                           f'Coord: ({x_volt:.2f}V,{y_volt:.2f}V)\n' \
                           f'Leftmost Line: {str(self._get_leftmost_line_coord_str())}\n' \
                           f'Bottommost Line: {str(self._get_bottommost_line_coord_str())}\n' \
 \
        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr

        if self.is_oracle_enable:
            # Oracle use ground truth with full confidence
            prediction = -1
            confidence = 1
            time_data_processed = time_data_fetched = perf_counter()
            is_above_confidence_threshold = True
        else:
            prediction = -1
            confidence = 1
            is_above_confidence_threshold = self.model.is_above_confident_threshold(True, 1)
            time_data_fetched = time_data_processed = perf_counter()

        self._scan_history.append(StepHistoryEntry(
            (self.x, self.y), prediction, confidence, False, False, False,
            is_above_confidence_threshold, step_description, time_start, time_data_fetched, time_data_processed,
            isinstance(self.diagram, DiagramOnline)
        ))

        self._previous_line_state = 1

        ratio = 1 / 2

        self._move_down_follow_line(
            ceil(self._get_avg_line_step_distance(line_distances=self._line_distances_2) * ratio)
        )

        self.move_to_coord(x, y)

        self._previous_line_state = 2

        self._move_up_follow_line(
            ceil(self._get_avg_line_step_distance(line_distances=self._line_distances_1) * ratio)
        )

        x, y = self.x, self.y
        state, x, y = self._enforce_boundary(True, x, y)
        self.move_to_coord(x, y)

        x = self.diagram.x_axes[x]
        y = self.diagram.y_axes[y]

        self._step_descr = f' ---- Final Coord ----\n' \
                           f'Coord: ({x:.2f}V,{y:.2f}V)\n' \
                           f'Leftmost Line: {str(self._get_leftmost_line_coord_str())}\n' \
                           f'Bottommost Line: {str(self._get_bottommost_line_coord_str())}\n' \
                           f'{tuple([0] * settings.dot_number)} area: ({x_volt_l:.2f}V,{y_volt_b:.2f}V)\n' \
                           f'{tuple([1] * settings.dot_number)} area: ({x:.2f}V,{y:.2f}V)'

        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr

        if self.is_oracle_enable:
            # Oracle use ground truth with full confidence
            prediction = True
            confidence = 1
            time_data_processed = time_data_fetched = perf_counter()
            is_above_confidence_threshold = True
        else:
            prediction = True
            confidence = 1
            is_above_confidence_threshold = self.model.is_above_confident_threshold(True, 1)
            time_data_fetched = time_data_processed = perf_counter()

        self._scan_history.append(StepHistoryEntry(
            (self.x, self.y), prediction, confidence, False, False, False,
            is_above_confidence_threshold, step_description, time_start, time_data_fetched, time_data_processed,
            isinstance(self.diagram, DiagramOnline)
        ))
        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)

        return

    def _enforce_boundary(self, force: bool = False, x: float = None, y: float = None) -> [bool, float, float]:
        """
        Check if the coordinates violate the boundary policy. If they do, move the coordinates according to the policy.
        :param force: If True the boundaries are forced, no matter the currant policy.
        :return: True if the coordinates are acceptable in the current policy, False if not.
        """

        # Always good for soft policies
        if not force and self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return True

        if force or self.boundary_policy is BoundaryPolicy.HARD:
            max_x, max_y = self.diagram.get_max_patch_coordinates()

            match_policy = True
            if x < 0:
                x = 0
                match_policy = False
            elif x > max_x:
                x = max_x
                match_policy = False
            if y < 0:
                y = 0
                match_policy = False
            elif y > max_y:
                y = max_y
                match_policy = False

            return match_policy, x, y

        raise ValueError(f'Unknown or invalid policy "{self.boundary_policy}" for diagram "{self.diagram}"')
