from typing import Tuple

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.data_structures import Direction
from plots.data import plot_diagram_step_animation
from utils.logger import logger


class SanityCheck(AutotuningProcedure):
    _sequence_size = 3

    def _tune(self) -> Tuple[int, int]:
        """
        Run some simple patterns to check if the autotuning is working as expected (especially the bloody coordinates).
        """

        expected_number_of_steps = 1 + (self._sequence_size * 4) + (self._sequence_size * 8 + 4)
        x_v, y_v = self.diagram.coord_to_voltage(self.x, self.y)
        logger.info(f'Auto-tuning sanity check. Start from ({x_v:.2f}V, {y_v:.2f}V). '
                    f'Expected number of steps: {expected_number_of_steps}')

        # First scan the starting position
        logger.debug(f'Debug stage (0) - Scan the starting position')
        self._step_name = '0. Starting position'
        self.is_transition_line()

        # Check the four directions around the starting position
        self._clockwise_sanity_check()

        # Check the four corners
        self._border_sanity_check()

        if len(self._scan_history) == expected_number_of_steps:
            logger.info(f'Expected number of steps ({expected_number_of_steps}) reached.')
        else:
            logger.error(f'Expected number of steps ({expected_number_of_steps}) not reached. '
                         f'Number of steps: {len(self._scan_history)}')

        return self.get_patch_center()

    def _clockwise_sanity_check(self) -> None:
        """
        Navigate clockwise around the starting position.
        Go up, right, down, left for a fixed number of steps.
        """

        directions = {
            'up': Direction(last_x=self.x, last_y=self.y, move=self.move_up, check_stuck=self.is_max_up),
            'right': Direction(last_x=self.x, last_y=self.y, move=self.move_right, check_stuck=self.is_max_right),
            'down': Direction(last_x=self.x, last_y=self.y, move=self.move_down, check_stuck=self.is_max_down),
            'left': Direction(last_x=self.x, last_y=self.y, move=self.move_left, check_stuck=self.is_max_left)
        }

        # Navigate clockwise around the starting position
        for stage_num, (label, direction) in enumerate(directions.items(), start=1):
            logger.debug(f'Clockwise sanity check ({stage_num}) - Go {label} for {self._sequence_size} steps')
            self._step_name = f'{stage_num}. Go {label}'
            # Go back at the starting position
            self.move_to_coord(direction.last_x, direction.last_y)
            for i in range(self._sequence_size):
                if direction.check_stuck():
                    logger.warning(f'Direction stuck because the border is reached, stop going {label}. '
                                   f'Step {i}/{self._sequence_size}.')
                    break
                direction.move()
                self.is_transition_line()

    def _border_sanity_check(self) -> None:
        """
        Navigate to each corner of the diagram, then navigate clockwise around the corner.
        We expect to be able to go in 2/4 directions at each corner.
        """

        # Get the coordinates of the corners
        max_x, max_y = self.diagram.get_max_patch_coordinates()
        corners = {
            'up right': (max_x, max_y),
            'down right': (max_x, 0),
            'down left': (0, 0),
            'up left': (0, max_y),
        }

        # Navigate to each corner of the diagram
        for stage_num, (corner_label, (x, y)) in enumerate(corners.items(), start=5):
            logger.debug(f'Border sanity check ({stage_num}) - Go to {corner_label}')
            self._step_name = f'{stage_num}. Go to {corner_label}'
            self.move_to_coord(x=x, y=y)
            self.is_transition_line()

            # We expect to find 2 valid directions at each corner
            directions = {
                'up': Direction(last_x=x, last_y=y, move=self.move_up, check_stuck=self.is_max_up),
                'right': Direction(last_x=x, last_y=y, move=self.move_right, check_stuck=self.is_max_right),
                'down': Direction(last_x=x, last_y=y, move=self.move_down, check_stuck=self.is_max_down),
                'left': Direction(last_x=x, last_y=y, move=self.move_left, check_stuck=self.is_max_left)
            }

            # Navigate clockwise around the starting position
            for direction_label, direction in directions.items():

                # Go back at the corner position
                self.move_to_coord(direction.last_x, direction.last_y)

                # Since we are at the corner, we expect to skip 2 directions
                if direction.check_stuck():
                    continue

                logger.debug(f'Border sanity check ({stage_num}) - Corner {corner_label} - Go {direction_label}')
                self._step_name = f'{stage_num}. Corner {corner_label} - Go {direction_label}'
                for i in range(self._sequence_size):
                    if direction.check_stuck():
                        logger.warning(f'Direction stuck because the border is reached, stop going {direction_label} '
                                       f'from {corner_label}. Step {i}/{self._sequence_size}.')
                        break
                    direction.move()
                    self.is_transition_line()

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        """
        Plot the animated diagram with the tuning steps of the current procedure.
        Hide the crosses during the step by step animation for easier debugging.

        :param final_coord: The final coordinate of the tuning procedure
        :param success_tuning: Result of the tuning (True = Success)
        """

        name = f'{self.diagram.file_basename} steps'
        # Generate a gif image
        plot_diagram_step_animation(self.diagram, name, self._scan_history, final_coord, False)
