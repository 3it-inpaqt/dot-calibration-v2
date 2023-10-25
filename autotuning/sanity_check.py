from typing import Tuple

import torch

from autotuning.autotuning_procedure import AutotuningProcedure
from classes.data_structures import Direction
from datasets.diagram_online import DiagramOnline
from plots.data import plot_diagram_step_animation
from utils.logger import logger
from utils.settings import settings


class SanityCheck(AutotuningProcedure):
    _sequence_size = 3

    def _tune(self) -> Tuple[int, int]:
        """
        Run some simple patterns to check if the autotuning is working as expected (especially the bloody coordinates).
        """
        # The number of steps expected for this sanity-check procedure
        nb_steps_expected = 1 + (self._sequence_size * 4) + (self._sequence_size * 8 + 4)
        # The total number of measurements (pixel) expected for this sanity-check procedure
        nb_measurements_expected = nb_steps_expected * settings.patch_size_x * settings.patch_size_y
        # Because of label offset measurement could, therefore some datapoint are measured twice
        surface_overlap_x = settings.label_offset_x * 2 * settings.patch_size_y
        surface_overlap_y = settings.label_offset_y * 2 * settings.patch_size_x
        nb_double_measurements_expected = self._sequence_size * 6 * (surface_overlap_x + surface_overlap_y)
        # The number of unique measurements expected
        nb_unique_measurements_expected = nb_measurements_expected - nb_double_measurements_expected

        if isinstance(self.diagram, DiagramOnline):
            # Get the voltage range
            min_x_v, max_x_v = settings.start_range_voltage_x
            min_y_v, max_y_v = settings.start_range_voltage_y

            # Convert the voltage to coordinates
            min_x, min_y = self.diagram.voltage_to_coord(min_x_v, min_y_v)
            max_x, max_y = self.diagram.voltage_to_coord(max_x_v, max_y_v)
        else:
            # Directly get the coordinate range
            min_x, min_y = 0, 0
            max_x, max_y = len(self.diagram.x_axes), len(self.diagram.y_axes)

        # Start from the middle of the starting area
        start_x = self.x = min_x + ((max_x - min_x) // 2) - (settings.patch_size_x // 2)
        start_y = self.y = min_y + ((max_y - min_y) // 2) - (settings.patch_size_y // 2)

        x_v, y_v = self.diagram.coord_to_voltage(self.x, self.y)
        logger.info(f'Auto-tuning sanity check. Start from ({x_v:.2f}V, {y_v:.2f}V). '
                    f'Expected number of steps: {nb_steps_expected}. '
                    f'Expected number of measurements: {nb_measurements_expected} '
                    f'(unique: {nb_unique_measurements_expected}).')

        # First scan the starting position
        logger.debug(f'Debug stage (0) - Scan the starting position')
        self._step_name = '0. Starting position'
        self.is_transition_line()

        # Check the four directions around the starting position
        self._clockwise_sanity_check()

        # Check the four corners
        self._border_sanity_check()

        assert len(self._scan_history) == nb_steps_expected, \
            f"The number of steps ({len(self._scan_history)}) doesn't match with the expectation ({nb_steps_expected})."

        if isinstance(self.diagram, DiagramOnline):
            nb_unique_measurements = self.diagram.values.isnan().logical_not().sum()
            assert nb_unique_measurements == nb_unique_measurements_expected, \
                f"The number of unique measurements ({nb_unique_measurements}) doesn't match with the " \
                f"expectation ({nb_unique_measurements_expected})."

            assert not torch.isnan(self.diagram.values[0][0]).item(), 'Corner (0, 0) is not measured.'
            assert not torch.isnan(self.diagram.values[-1][0]).item(), 'Corner (0, -1) is not measured.'
            assert not torch.isnan(self.diagram.values[0][-1]).item(), 'Corner (-1, 0) is not measured.'
            assert not torch.isnan(self.diagram.values[-1][-1]).item(), 'Corner (-1, -1) is not measured.'
            assert not torch.isnan(self.diagram.values[start_y][start_x]).item(), \
                f'Starting point ({start_x}, {start_y}) is not measured.'

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

            # We expect to find 2 valid directions in each corner
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

                # Since we are in the corner, we expect to skip 2 directions
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
        file_name = f'tuning_{self}_{self.diagram.name}'
        title = f'Tuning {self}: {self.diagram.name}'
        # Generate a gif and / or video
        plot_diagram_step_animation(self.diagram, title, file_name, self._scan_history, final_coord)
