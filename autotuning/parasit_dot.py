from math import ceil, atan2, pi
from typing import Tuple

import torch

from autotuning.jump import Jump
from autotuning.jump_ndots import JumpNDots
from classes.data_structures import Direction, SearchLineSlope, StepHistoryEntry
from datasets.qdsd import QDSDLines
from utils.logger import logger
from utils.settings import settings


def selection_parasitdot_procedure(model, patch_size, label_offsets, autotuning_use_oracle):
    if settings.parasit_dot_procedure:
        # Append classification parasite dot
        QDSDLines.classes.append('Parasite dot')

    if settings.dot_number == 1:
        return ParasitDotProcedure(model, patch_size, label_offsets, autotuning_use_oracle)
    else:
        raise 'Not Implemented'
        return ParasitsDotsProcedure(model, patch_size, label_offsets, autotuning_use_oracle)


class ParasitDotProcedure(Jump):
    """
    Procedure to detect if the line is a parasit dot or not
    """

    _delta: int = 10  # Uncertainty on the angle of the slope
    _epsilon: int = 4  # Uncertainty on the inter-line spacing
    _line_coord: list = []  # List of the coord of all the detected line
    _slope_list: list = []  # List of the slope of some detected line
    _last_distance: int = None
    _default_line_distance: int = None

    # Settings to start the parasite dot procedure only if we are on stage (3)
    _parasite_procedure: bool = False

    # If the two first line doesn't have the same slope _hard_procedure = True, else _hard_procedure = False
    _hard_procedure: bool = False

    def reset_procedure(self):
        super().reset_procedure()

        # TODO move in settings
        if settings.research_group == 'michel_pioro_ladriere':
            self._line_slope = 75  # Prior assumption about line direction
            self._line_distances = self._default_line_distance = [5]  # Prior assumption about distance between lines

        elif settings.research_group == 'louis_gaudreau':
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances = self._default_line_distance = [3]  # Prior assumption about distance between lines

        elif settings.research_group == 'eva_dupont_ferrier':
            self._line_slope = 10  # Prior assumption about line direction
            self._line_distances = self._default_line_distance = [4]  # Prior assumption about distance between lines

        else:
            logger.warning(f'No prior knowledge defined for the dataset: {settings.research_group}')
            self._line_slope = 45  # Prior assumption about line direction
            self._line_distances = self._default_line_distance = [4]  # Prior assumption about distance between lines

        self._bottommost_line_coord = None
        self._leftmost_line_coord = None
        self._line_coord = []
        self._slope_list = []
        self._parasite_procedure = False
        self._hard_procedure = False

    def _confirm_line(self, confidence: float = None) -> bool:
        logger.debug(f'Line detected, check if it is a parasite dot')
        init_angle = self._line_slope
        logger.debug(f'Init angle: {init_angle},\n'
                     f'List line ({len(self._line_coord)}): {self._line_coord},\n'
                     f'List slope ({len(self._slope_list)}): {self._slope_list},\n'
                     f'Average line distance: {self._get_avg_line_step_distance()},\n'
                     f'Actual line distance: {self._last_distance}')
        # Case of this is the First line detected
        if len(self._line_distances) == 0:
            logger.debug(f'Case: first Line detected')
            self._record_log(True, confidence, None, self.x, self.y)
            return True

        # Case of this is the second line detected
        elif len(self._line_distances) == 1:
            logger.debug(f'Case: second Line detected')
            self._parasite_procedure = False
            self._search_line_slope()
            self._parasite_procedure = True
            if init_angle - self._delta < self._line_slope < init_angle + self._delta:
                # Same angle -> True line
                self._line_slope = init_angle
                self._slope_list = [self._line_slope]
                self._record_log(True, confidence, None, self.x, self.y)
                return True
            else:
                # Case Parasite dot on the first or the second line
                self._slope_list = [init_angle] + [self._line_slope]
                self._line_slope = init_angle
                self._hard_procedure = True
                self._record_log(True, confidence, -1, self.x, self.y)
                return True

        # Check witch line is a parasite dot with the angle
        # From the beginning, the detected line don't have the same slope
        elif self._hard_procedure:
            logger.debug(f'Case: hard procedure on')
            self._parasite_procedure = False
            self._search_line_slope()
            self._parasite_procedure = True
            for nb, slope in enumerate(self._slope_list):
                if self._line_slope - self._delta < slope < self._line_slope + self._delta:
                    self._line_coord = [self._line_coord[nb]]
                    self._slope_list = [self._slope_list[nb]]
                    self._leftmost_line_coord = self._line_coord[0][0], self._line_coord[0][1]

                    # Correction Label
                    logger.debug(f'Correction label on {(self._line_coord[0][0], self._line_coord[0][1])} as Line'
                                 f'as reference {(self.x, self.y)}')
                    self._hard_procedure = False
                    self._record_log(True, confidence, None, self._line_coord[0][0], self._line_coord[0][1])
                    return True
                else:
                    self._record_log(True, confidence, -1, self._line_coord[0][0], self._line_coord[0][1])
            # We don't have two lines with the same slope
            self._slope_list = self._slope_list + [self._line_slope]
            self._record_log(True, confidence, -1, self.x, self.y)
            return True

        # Check witch line is a parasite dot with the distance
        else:
            logger.debug(f'Case: hard procedure off')
            average_distance = self._get_avg_line_step_distance()
            if average_distance - self._epsilon < self._last_distance < average_distance + self._epsilon:
                self._record_log(True, confidence, None, self.x, self.y)
                return True
            else:
                self._record_log(True, confidence, -1, self.x, self.y)
                return False

    def _search_empty(self) -> None:
        """
        Explore the diagram by scanning patch perpendicular to the estimated lines direction.
        """
        logger.debug(f'Stage ({3 if settings.auto_detect_slope else 2}) - Search empty')
        self._step_name = f'{3 if settings.auto_detect_slope else 2}. Search 0 e-'
        self._parasite_procedure = True

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

                # Check line and save position if leftmost one
                self._last_distance = direction.no_line_count + 1
                line_detected = self._is_confirmed_line()

                # If new line detected, save distance and reset counter
                if line_detected:
                    self._line_distances.append(direction.no_line_count)
                    self._line_coord.append((self.x, self.y))
                    if direction.no_line_count >= 1:
                        self._nb_line_found += 1
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
                    if self._nb_line_found > 1 and direction.no_line_count > 3 * avg_line_distance:
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
            nb_line_search = 0
            new_line_found = False
            # Both direction start at the leftmost point
            up.last_x, up.last_y = start_point
            down.last_x, down.last_y = start_point
            up.no_line_count = round(self._default_step_y * line_step_distance * 2)
            down.no_line_count = round(self._default_step_y * line_step_distance * 2)
            up.is_stuck = down.is_stuck = False  # Unstuck since we are stating at a new location
            while not new_line_found and not Direction.all_stuck((up, down)):
                for direction in (d for d in (up, down) if not d.is_stuck):
                    # Check if we reached the maximum number of leftmost search for the current line
                    nb_line_search += 1
                    if nb_line_search > self._max_nb_leftmost_checking:
                        return
                    self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                    # Step distance relative to the line distance
                    direction.move(round(self._default_step_y * line_step_distance * 2))
                    direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                    direction.is_stuck = direction.check_stuck()
                    if direction.is_stuck:
                        break  # We don't scan if we have reached the border

                    # Skip half line distance left
                    self._move_left_perpendicular_to_line(ceil(self._default_step_x * line_step_distance / 2))

                    # Go left for 2x the line distance (total 2x the line distance)
                    line_distance = 1
                    for i in range(ceil(line_step_distance * 2)):
                        nb_steps += 1
                        # If new line found and this is the new leftmost one, start again the checking loop
                        self._last_distance = direction.no_line_count + 1
                        if self._is_confirmed_line() and start_point != self._leftmost_line_coord:
                            self._nb_line_found += 1
                            new_line_found = True
                            start_point = self._leftmost_line_coord
                            self._step_descr = f'line slope: {self._line_slope:.0f}°\n' \
                                               f'avg line dist: {line_step_distance:.1f}\n' \
                                               f'nb line found: {self._nb_line_found}\n' \
                                               f'leftmost line: {self._get_leftmost_line_coord_str()}'
                            break
                        direction.no_line_count += 1
                        self._move_left_perpendicular_to_line()
                        if self.is_max_left() or self.is_max_down():
                            break  # Nothing else to see here
                        line_distance += 1

                    if nb_steps > self._max_steps_validate_left_line:
                        return  # Hard break to avoid infinite search in case of bad slope detection (>90°)

                    if new_line_found:
                        break

    def is_transition_line(self) -> Tuple[bool, float]:
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """

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
        else:
            with torch.no_grad():
                # Cut the patch area and send it to the model for inference
                patch = self.diagram.get_patch((self.x, self.y), self.patch_size)
                # Reshape as valid input for the model (batch size, patch x, patch y)
                size_x, size_y = self.patch_size
                patch = patch.view((1, size_x, size_y))
                # Send to the model for inference
                prediction, confidence = self.model.infer(patch, settings.bayesian_nb_sample_test)
                # Extract data from pytorch tensor
                prediction = prediction.item()
                confidence = confidence.item()

        # Check parasit dot
        if settings.parasit_dot_procedure and self._parasite_procedure and prediction:
            prediction = self._confirm_line(confidence)
        else:
            # Record the diagram scanning activity.
            pred = 1 if (self._parasite_procedure and prediction) else 0 if not prediction else -1
            self._record_log(pred, confidence, None, self.x, self.y)
        return prediction, confidence

    def _record_log(self, prediction, confidence, prediction_corrected, x: int, y: int):
        ground_truth, soft_truth_larger, soft_truth_smaller = self.get_ground_truths(x, y)
        _class = prediction_corrected if prediction_corrected else prediction
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr
        self._scan_history.append(StepHistoryEntry((x, y), _class, confidence, ground_truth,
                                                   soft_truth_larger, soft_truth_smaller, step_description))

        logger.debug(f'Patch {self.get_nb_steps():03}, coord {[x, y]}, classified as {QDSDLines.classes[prediction]} '
                     f'with confidence 'f'{confidence:.2%}'
                     f'{f" corrected as {QDSDLines.classes[prediction_corrected]}" if prediction_corrected else ""}')

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
            self._line_coord.append([self.x, self.y])
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
                    self._line_coord.append([self.x, self.y])
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
        step_distance = round(self._default_step_y * self._default_line_distance[0])
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
        logger.debug(f'End search line slope')


class ParasitsDotsProcedure(JumpNDots):
    """
    Not implemented yet
    """
