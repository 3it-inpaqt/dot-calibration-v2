import random
from math import atan2, ceil, pi, radians, tan
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from autotuning.jump import Jump
from classes.classifier_nn import ClassifierNN
from classes.data_structures import BoundaryPolicy, StepHistoryEntry
from classes.data_structures import Direction, SearchLineSlope
from datasets.diagram_online import DiagramOnline
from datasets.qdsd import QDSDLines
from models.slope_network import init_slope_model, load_slope_network, norm_value
from utils.logger import logger
from utils.output import get_save_path, OUT_DIR
from utils.settings import settings


def Jump_selector(model, patch_size, label_offsets, autotuning_use_oracle):
    if settings.dot_number == 1:
        return Jump(model, patch_size, label_offsets, autotuning_use_oracle)
    else:
        if settings.slope_network:
            return JumpNDots_slope(model, patch_size, label_offsets, autotuning_use_oracle)
        else:
            return JumpNDots(model, patch_size, label_offsets, autotuning_use_oracle)


class JumpNDots(Jump):
    """
    Same as Jump but for N Dots
    TODO Adapt for N dots

    Warning: the horizontal line is set as Line 1 (Bottommost Line) and
                the vertical line is set as Line 2 (Leftmost line)
    """

    # -- Exploration limits -- #

    _nb_line_found_1: int = 0
    _nb_line_found_2: int = 0
    _nb_line_found_cross: int = 0
    _max_nb_line_leftmost: int = 4
    _max_nb_line_bottommost: int = 4
    _max_steps_validate_line: int = 75
    _class = QDSDLines.classes
    _bottommost_line_coord = [None, None]
    _leftmost_line_coord = [None, None]

    # -- Line  -- #

    # Line state
    _line_state: int = 0

    # List of distance between lines in pixel
    _line_distances_1: List[int] = None
    _line_distances_2: List[int] = None

    # Line slope (0 = 0° | inf = 90° | -1 = 45° | 1 = 135°)
    _line_slope_default_1: float = 45
    _line_slope_default_2: float = 80
    _line_slope_1: float = None
    _line_slope_2: float = None
    _error_1: float = 0
    _error_2: float = 0
    _sigma: list = []

    # -- Plot parameters -- #
    _nb_plot_bad: int = settings.nb_error_to_plot
    _nb_plot_good: int = settings.nb_good_to_plot
    _run_nb: int = 0

    def reset_procedure(self):
        super().reset_procedure()

        self._bottommost_line_coord = [None, None]
        self._leftmost_line_coord = [None, None]
        self._line_state = 0

        if settings.research_group == 'eva_dupont_ferrier':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 10
            self._line_slope_default_2 = 100
            # Prior assumption about distance between lines
            self._line_distances_1 = [10]
            self._line_distances_2 = [20]
            # Prior assumption about the estimation slope error
            self._error_1 = -3
            self._error_2 = 24
        elif settings.research_group == 'louis_gaudreau':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 149
            self._line_slope_default_2 = 119
            # Prior assumption about distance between lines
            self._line_distances_1 = [14]
            self._line_distances_2 = [12]
            # Prior assumption about the estimation slope error
            self._error_1 = 0  # 14
            self._error_2 = 0  # 18
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
            # Prior assumption about the estimation slope error
            self._error_1 = 0
            self._error_2 = 0

    def default_slope(self, line_state: int) -> float:
        if line_state == 1:
            return self._line_slope_default_1
        else:
            return self._line_slope_default_2

    # ========================== #
    # --   Utility function   -- #
    # ========================== #

    def _setup_direction_X(self) -> list:
        """
        Return all the diagonal direction
        :return: list
        """
        direction = [
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_left, check_stuck=self.is_max_down_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_left, check_stuck=self.is_max_up_left),
            Direction(last_x=self.x, last_y=self.y, move=self.move_up_right, check_stuck=self.is_max_up_right),
            Direction(last_x=self.x, last_y=self.y, move=self.move_down_right, check_stuck=self.is_max_down_right),
        ]
        return direction

    def _get_direction(self, line_state: Optional[int] = None) -> tuple:
        """
        Get the direction left, right, up, down of the state line

        :line_state: The status of the line from which we will extract the directions
        :return: Directions of the lines
        """

        if line_state:
            previous_line_state = self._line_state
            self._line_state = line_state

        # Define direction for horizontal line
        left = Direction(last_x=self.x, last_y=self.y, move=self._move_left_perpendicular_to_line,
                         check_stuck=self.is_inside)
        right = Direction(last_x=self.x, last_y=self.y, move=self._move_right_perpendicular_to_line,
                          check_stuck=self.is_inside)
        up = Direction(last_x=self.x, last_y=self.y, move=self._move_up_follow_line,
                       check_stuck=self.is_inside)
        down = Direction(last_x=self.x, last_y=self.y, move=self._move_down_follow_line,
                         check_stuck=self.is_inside)
        if line_state:
            self._line_state = previous_line_state

        return left, right, up, down

    def is_inside(self) -> bool:
        return self.is_max_down() and self.is_max_up() and self.is_max_left() and self.is_max_right()

    def _convert_pixel_to_volt(self, pixel: float = None) -> float:
        return pixel * settings.pixel_size

    def _is_bottommost_or_leftmost_line(self, line_state: int) -> None:
        """
        Check if the current position should be considered the leftmost line or the bottommost line.
        """
        if line_state == 1 \
                and (self._bottommost_line_coord == [None, None] or self._is_bottom_relative_to_line()):
            self._bottommost_line_coord = self.x, self.y
        elif line_state == 2 \
                and (self._leftmost_line_coord == [None, None] or self._is_left_relative_to_line()):
            self._leftmost_line_coord = self.x, self.y

    def _is_left_relative_to_line(self) -> bool:
        """
        Check if the current position is at the left of the leftmost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new leftmost point.
        """

        # Reconstruct line equation (y = m*x + b)
        slope, offset = self._line_interpolation(
            self._line_slope_2, 2, [self._leftmost_line_coord[0] - self._default_step_x,
                                    self._leftmost_line_coord[1] - self._default_step_y])

        # Special condition for 90° (vertical line) because tan(90) is undefined
        if slope == 0:
            return self.x < self._leftmost_line_coord[0] - self._default_step_x

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = slope * self.x + offset
        y_delta = y_line - self.y

        return (y_delta > 0 > slope) or (y_delta < 0 < slope)

    def _is_bottom_relative_to_line(self) -> bool:
        """
        Check if the current position is at the bottom of the bottommost line found so far, considering the line angle.

        :return: True if the current position should be considered as the new bottommost point.
        """

        # Reconstruct line equation (y = m*x + b)
        slope, offset = self._line_interpolation(
            self._line_slope_1, 1, [self._bottommost_line_coord[0] - self._default_step_x,
                                    self._bottommost_line_coord[1] - self._default_step_y])

        # Check if the current position is at the left (https://math.stackexchange.com/a/1896651/1053890)
        y_line = slope * self.x + offset
        y_delta = y_line - self.y
        return y_delta > 0

    def _get_bottommost_line_coord_str(self) -> str:
        """
        :return: Bottommost coordinates with volt conversion.
        """
        if self._bottommost_line_coord == [None, None]:
            return 'None'

        x, y = self._bottommost_line_coord
        x_volt, y_volt = self.diagram.coord_to_voltage(x, y)

        return f'{x_volt:.2f} V, {y_volt:.2f} V'

    def _get_leftmost_line_coord_str(self) -> str:
        """
        :return: Leftmost coordinates with volt conversion.
        """
        if self._leftmost_line_coord == [None, None]:
            return 'None'

        x, y = self._leftmost_line_coord
        x_volt, y_volt = self.diagram.coord_to_voltage(x, y)

        return f'{x_volt:.2f} V, {y_volt:.2f} V'

    def _get_bottom_left_coord(self) -> tuple:
        """
        Get the coord of the bottom left point thank to the leftmost point and the bottommost point
        :return: (float, float)
        """
        x = self._leftmost_line_coord[0] \
            if self._leftmost_line_coord[0] else self.diagram.get_random_starting_point()[0]
        y = self._bottommost_line_coord[1] \
            if self._bottommost_line_coord[1] else self.diagram.get_random_starting_point()[1]

        return x, y

    # == Rework move function == #

    def _move_up_follow_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        angle = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
        self._move_relative_to_line(angle, step_size)

    def _move_down_follow_line(self, step_size: Optional[int] = None, line_state: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        angle = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
        self._move_relative_to_line(angle + 180, step_size)

    def _move_right_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_perpendicular_to_line(False)

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        angle = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
        self._move_relative_to_line(angle - 90, step_size)

    def _move_left_perpendicular_to_line(self, step_size: Optional[int] = None) -> None:
        """
        Alias of _move_relative_to_line

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        angle = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
        self._move_relative_to_line(angle + 90, step_size)

    # ========================== #

    def _get_avg_line_step_distance(self, line_distances: list = None) -> float:
        """ Get the mean line distance in number of steps. """

        return sum(line_distances) / len(line_distances)

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

    def _plot_intersection(self, result: str = None, l1: list = [None, None], l2: list = [None, None],
                           l_point_volt: list = [None, None], b_point_volt: list = [None, None],
                           i_point: list = [None, None], f_point: list = [None, None]) -> None:
        """
        Plot the final step (intersection point, interpolated line, true line, final point)

        :param result: If the autotuning is a success or not
        :param l1: parameter of line 1 (slope, offset)
        :param l2: parameter of line 2 (slope, offset)
        :param l_point_volt: Leftmost point
        :param b_point_volt: Bottommost point
        :param i_point: Intersection point
        :param f_point: Final point
        """

        logger.debug(f'Plot Intersection point for a {result}')

        x_start, y_start = self.x, self.y
        # Parameter of Line
        slope_1, b1 = l1
        slope_2, b2 = l2
        x_b_volt, y_b_volt = b_point_volt
        x_l_volt, y_l_volt = l_point_volt
        x_intersec, y_intersec = self.diagram.coord_to_voltage(i_point[0], i_point[1])
        x_final, y_final = self.diagram.coord_to_voltage(f_point[0], f_point[1])

        self.x, self.y = x_start, y_start

        # Save Path

        number = settings.nb_good_to_plot - self._nb_plot_good if result == 'Success' else \
            settings.nb_error_to_plot - self._nb_plot_bad
        name = f'Intersection_{self.diagram.name}_{result}_{number}'
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), name, 'png', False)

        values, x_axes_volt, y_axes_volt = self.diagram.get_values()

        line_volt = []
        slope1_true, offset1_true = self._line_interpolation(
            self._line_slope_default_1 - self._error_1, 1, [x_b_volt, y_b_volt])
        slope2_true, offset2_true = self._line_interpolation(
            self._line_slope_default_2 - self._error_2, 2, [x_l_volt, y_l_volt])

        for i_volt in x_axes_volt:
            j1_volt = slope_1 * i_volt + b1
            j1_true = slope1_true * i_volt + offset1_true
            j2_volt = slope_2 * i_volt + b2
            j2_true = slope2_true * i_volt + offset2_true
            line_volt.append([j1_volt, j2_volt, j1_true, j2_true])

        line_volt = np.array(line_volt)

        plt.figure(name, figsize=(12.8, 9.6), dpi=200)
        extent = [x_axes_volt[0], x_axes_volt[-1],
                  y_axes_volt[0], y_axes_volt[-1]]
        plt.imshow(values, cmap='copper', extent=extent, origin='lower')
        cbar = plt.colorbar()
        cbar.set_label('$I_{QPC}$ (A)', rotation=90)

        plt.plot(x_axes_volt, line_volt[:, 0],
                 label=f'Line bottommost: {slope_1:.2f}.x + {b1:.2f} ; angle: {self._line_slope_1:0.1f}°', color='blue')
        plt.plot(x_axes_volt, line_volt[:, 2],
                 label=f'Line true bottommost: angle: {self._line_slope_default_1}°, '
                       f'$\\Delta$ = {self._line_slope_default_1 - self._line_slope_1: 0.2f}°', color='skyblue')
        plt.scatter(x_b_volt, y_b_volt, label='Bottommost point', color='blue')

        plt.plot(x_axes_volt, line_volt[:, 1],
                 label=f'Line leftmost: {slope_2:.2f}.x + {b2:.2f} ; angle: {self._line_slope_2:0.1f}°', color='orange')
        plt.plot(x_axes_volt, line_volt[:, 3],
                 label=f'Line true leftmost: angle: {self._line_slope_default_2}°, '
                       f'$\\Delta$ = {self._line_slope_default_2 - self._line_slope_2: 0.2f}°', color='bisque')
        plt.scatter(x_l_volt, y_l_volt, label='Leftmost point', color='orange')

        plt.scatter(x_intersec, y_intersec, label='Intersection point', color='purple', marker='x', s=300)
        plt.scatter(x_final, y_final, label='Final coord', color='red', marker='x', s=300)
        plt.xlabel('Gate 1 (Volt)')
        plt.ylabel('Gate 2 (Volt)')
        plt.legend()
        plt.grid(False)
        plt.xlim(x_axes_volt[0], x_axes_volt[-1])  # x goes from -7 to 0
        plt.ylim(y_axes_volt[0], y_axes_volt[-1])  # y goes from -7 to 2
        # The tight bbox will remove white space around the image, the image "figsize" won't be respected.
        plt.savefig(save_path, dpi=200)

    def _line_interpolation(self, angle: float, line: int, coord: list) -> tuple:
        """
        Reconstruct Line equation (y = a * x + b) ou (y = tan(angle) * x)
        :param angle: angle of the line in deg
        :param line: type of line (line 1 or line 2)
        :return: parameters of the line (slope and offset)
        """
        if angle > 0:
            angle = -angle

        # Validity domain
        if angle == 90 or angle == -90:
            slope = 0
        else:
            error = self._error_1 if line == 1 else self._error_2
            slope = tan(radians(-angle + error))

        offset = coord[1] - (coord[0] * slope)

        return slope, offset

    def verif_patch(self, data, angle, target_line, debug: bool = False) -> None:
        """
        Plot the patch with the angle estimated by the NN and the true angle (mostly for debug)
        :param data: value of the patch
        :param angle: estimated angle of the patch
        :param target_line: corresponding line
        :param debug: If we plot or not the patch
        """
        if not debug:
            return
        _, self.x, self.y = self._enforce_boundary(force=True, x=self.x, y=self.y)
        x, y = self.get_patch_center()
        x_volt = self.diagram.x_axes[x]
        y_volt = self.diagram.y_axes[y]

        slope, offset = self._line_interpolation(angle, target_line, [x_volt, y_volt])
        i = self.diagram.x_axes[self.x: self.x + settings.patch_size_x]
        j = slope * np.array(self.diagram.x_axes[self.x: self.x + settings.patch_size_x]) + offset

        angle_true = self._line_slope_default_1 if target_line == 1 else self._line_slope_default_2
        slope_true, offset_true = self._line_interpolation(angle_true, target_line, [x_volt, y_volt])
        x_true = self.diagram.x_axes[self.x: self.x + settings.patch_size_x]
        y_true = slope_true * np.array(self.diagram.x_axes[self.x: self.x + settings.patch_size_x]) + offset_true

        plt.figure(f'line_{target_line}', figsize=(12.8, 9.6), dpi=200)
        plt.title(f"Patch {self.get_nb_steps()} as {self._class[target_line]}: angle = {angle:0.2f},"
                  f" $\\Delta\\alpha$ = {angle - angle_true:0.2f}")

        extent = [self.diagram.x_axes[self.x], self.diagram.x_axes[self.x + settings.patch_size_x - 1],
                  self.diagram.y_axes[self.y], self.diagram.y_axes[self.y + settings.patch_size_y - 1]]

        plt.imshow(data, cmap='copper', extent=extent, origin='lower')
        cbar = plt.colorbar()

        plt.plot(i, j, label=f'NN line: {slope:0.2e}.x + {offset:0.2e}')
        plt.plot(x_true, y_true, label=f'True line: {slope_true:0.2e}.x + {offset_true:0.2e}')

        cbar.set_label('$I_{QPC}$ (A)', rotation=90)
        plt.xlabel('Gate 1 (Volt)')
        plt.ylabel('Gate 2 (Volt)')
        plt.legend()
        plt.grid(False)
        plt.xlim(self.diagram.x_axes[self.x], self.diagram.x_axes[self.x + settings.patch_size_x - 1])
        plt.ylim(self.diagram.y_axes[self.y], self.diagram.y_axes[self.y + settings.patch_size_y - 1])

        name = f'{self._class[target_line]}'
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), name, 'png', False)
        plt.savefig(save_path, dpi=200)

    def _plot_stat(self):
        """
        Plot stat of the angle NN
        """
        if not self._run_nb == settings.autotuning_nb_iteration:
            return

        angle_line_1 = np.array([self._sigma[i][0] for i in range(len(self._sigma))])
        metric_line_1 = np.array([self._sigma[i][1] for i in range(len(self._sigma))])
        angle_line_2 = np.array([self._sigma[i][2] for i in range(len(self._sigma))])
        metric_line_2 = np.array([self._sigma[i][3] for i in range(len(self._sigma))])
        x = [i for i in range(1, len(angle_line_1) + 1)]

        length = 10
        width = length * 2.5
        plt.figure('Stat angle', figsize=(width, length), dpi=400)
        plt.suptitle('Statistic')
        # Mean
        plt.subplot(121)
        plt.title('Angle')

        plt.plot(angle_line_1, label=self._class[1], color=(1, 0.5, 0.5), linestyle='--', marker='o', markersize=2)
        plt.plot([np.mean(angle_line_1)] * len(angle_line_2), label=f'Mean of {self._class[1]} angle',
                 color='red', linestyle='-')
        plt.plot([self._line_slope_default_1] * len(angle_line_2), label=f'Angle for {self._class[1]}', color='gray')

        plt.plot(angle_line_2, label=self._class[2], color=(0.5, 0.5, 1), linestyle='--', marker='s', markersize=2)
        plt.plot([np.mean(angle_line_2)] * len(angle_line_2), label=f'Mean of {self._class[2]} angle',
                 color='blue', linestyle='-')
        plt.plot([self._line_slope_default_2] * len(angle_line_2), label=f'Angle for {self._class[2]}', color='black')

        # plt.text(0.1, 0.5, f'Variance {self._class[1]}: {np.var(angle_line_1): 0.2f}\n'
        #                    f'Variance {self._class[2]}: {np.var(angle_line_2): 0.2f}',
        #          transform=plt.gca().transAxes, fontsize=14, verticalalignment='top',
        #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xlabel('Iteration')
        plt.ylabel('Angle (°)')
        plt.gca().xaxis.grid(False)
        plt.legend()

        # Variance
        plt.subplot(222)
        plt.title(f'Metric for {self._class[1]}')

        # Création du premier axe pour la variance
        ax1 = plt.gca()
        line1, = ax1.plot(metric_line_1[:, 0], linestyle='--', color='b')
        ax1.set_ylabel('Variance')
        ax1.tick_params('y', colors='b')

        # Création du deuxième axe pour le delta
        ax2 = ax1.twinx()
        line2, = ax2.plot(metric_line_1[:, 1], linestyle='--', color='r')
        ax2.set_ylabel('$\\Delta \\alpha$')
        ax2.tick_params('y', colors='r')

        ax1.set_xlabel('Iteration')

        # Création d'une seule légende pour les deux lignes
        lines = [line1, line2]
        labels = ['Variance', '$\\Delta \\alpha$']
        ax1.legend(lines, labels, loc='upper right')

        plt.subplot(224)
        plt.title(f'Metric for {self._class[2]}')

        # Création du premier axe pour la variance
        ax1 = plt.gca()
        line1, = ax1.plot(metric_line_2[:, 0], linestyle='--', color='b')
        ax1.set_ylabel('Variance')
        ax1.tick_params('y', colors='b')

        # Création du deuxième axe pour le delta
        ax2 = ax1.twinx()
        line2, = ax2.plot(metric_line_2[:, 1], linestyle='--', color='r')
        ax2.set_ylabel('$\\Delta \\alpha$')
        ax2.tick_params('y', colors='r')

        ax1.set_xlabel('Iteration')

        # Création d'une seule légende pour les deux lignes
        lines = [line1, line2]
        labels = ['Variance', '$\\Delta \\alpha$']
        ax1.legend(lines, labels, loc='upper right')

        name = f'Stat_slope_{self.diagram.name}'
        save_path = get_save_path(Path(OUT_DIR, settings.run_name, 'img'), name, 'png', False)
        plt.savefig(save_path, dpi=400)
        plt.close()

    def metrics(self, angle, true_angle):
        """
        Give MAE, RMSE and MEDAE of the estimation angle
        :param angle: angle predicted by the NN
        :param true_angle: true angle
        :return:
        """
        if not len(angle):
            return 0, 0, 0, 0

        errors = np.array(angle) - np.array(true_angle)

        var = np.var(angle)
        delta = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        medae = np.median(np.abs(errors))

        return var, delta, rmse, medae

    # ========================== #
    # --     Sub function     -- #
    # ========================== #

    def _search_line_angle(self, force_target_line: Optional[int] = None) -> None:
        """
        Estimate the direction of the current line and update the _line_slope attribut if the measurement looks valid.
        """
        # Step distance relative to the line distance to reduce the risk to reach another line
        line_distance = self._line_distances_1 if self._line_state == 1 else self._line_distances_2
        step_distance = round(
            self._default_step_y * self._get_avg_line_step_distance(line_distances=line_distance) / 2)
        target_line = force_target_line if force_target_line else self._line_state
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
                self._step_descr = f'Target line: {self._class[self._line_state]}\n' \
                                   f'Delta: {delta}°\n' \
                                   f'Init Line: {init_line}\n' \
                                   f'Init No Line: {init_no_line}'

                line_state, _ = self.is_transition_line()  # No confidence validation here
                line_detected = True if line_state == target_line else False

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

            if target_line == 1:
                self._line_slope_1 = - slope_estimation + 180
            else:
                self._line_slope_2 = - slope_estimation + 180

        else:
            # Doesn't find the angle
            if target_line == 1:
                self._line_slope_1 = self._line_slope_default_1
            else:
                self._line_slope_2 = self._line_slope_default_2

        slope_estimation = self._line_slope_1 if target_line == 1 else self._line_slope_2

        if settings.patch_verif_plot:
            self.move_to_coord(start_x, start_y)
            self.verif_patch(self.diagram.get_patch((self.x, self.y), self.patch_size, normalized=False),
                             slope_estimation, target_line, settings.patch_verif_plot)

        self.move_to_coord(start_x, start_y)
        logger.debug(f'Calculation finish for {self._class[target_line]}: angle = {slope_estimation}')

    def _verif_slope(self, stage: int = None, substage: int = None, subsubstage: int = None,
                     target_line: float = None, force_estimation: bool = False) -> None:
        """
        Estimation of the target line slope if needed
        :param stage: Main step
        :param substage: Sub step (First 0 electron area, Search other line, Second 0 electron area)
        :param subsubstage: Sub-Sub step (Slope calculation or else)
        :param target_line: the target line
        :param force_estimation: force the slope estimation
        :return: Estimation of the target line slope
        """
        slope = self._line_slope_1 if target_line == 1 else self._line_slope_2
        if force_estimation or not slope:
            logger.debug(
                f'Stage ({stage}.{substage}.{subsubstage}) - Slope calculation for {self._class[target_line]}')
            self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - ' \
                              f'Slope calculation for {self._class[target_line]}'
            self._search_line_angle(force_target_line=target_line)

    def _line_around_crosspoint(self, target_line: Optional[int] = None) -> bool:
        """
        We search the nearest line around the crossbar
        :target_line: Optional - if defined, we search the corresponding line around the crossbar
        :return: True if we find a line, else False
        """
        start_x, start_y = self.x, self.y
        start_angle = 10

        # Step distance relative to the line distance to reduce the risk to reach another line too far away
        line_distance = self._line_distances_1 if self._line_state == 1 else self._line_distances_2
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
                    self._line_state = line_state
                    return True
            delta += search_step
        # We don't find a line
        return False

    def _search_0_area(self, stage: int = None, substage: int = None, subsubstage: int = None) -> bool:
        """
        Search the 0 electron area
        :param stage: Main step
        :param substage: Sub step (First 0 electron area, Search other line, Second 0 electron area)
        :param subsubstage: Sub-Sub step (Slope calculation or else)
        :return: bool, if we succeed to find the area or not
        """
        # Line slope
        self._verif_slope(stage, substage, subsubstage, self._line_state)
        subsubstage += 1

        # With the slope of the line, we will move perpendicular to the line to find the zero-electron regime

        logger.debug(
            f'Stage ({stage}.{substage}.{subsubstage}) - Start research for {self._class[self._line_state]} at a coord ('
            f'{self._get_leftmost_line_coord_str() if self._line_state == 2 else self._get_bottommost_line_coord_str()})')
        self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - Start 0 area research for {self._class[self._line_state]}'
        directions = self._get_direction(line_state=self._line_state)

        # Get 0 electron regime for the first line
        self._get_empty_area(stage, substage, directions)

        # Optional step: make sure we found the leftmost or bottommost line
        fine_tuning = [settings.validate_bottom_line, settings.validate_left_line][self._line_state - 1]
        if fine_tuning:
            substage += 1
            logger.debug(
                f'Stage ({stage}.{substage}.{subsubstage}) - Validate {"leftmost" if self._line_state == 2 else "bottommost"}'
                f' line')
            self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - Validate {"leftmost" if self._line_state == 2 else "bottommost"} line'
            self._validate_line(stage, substage, directions)

        return True

    def _search_other_line(self, stage: int = None, substage: int = None, subsubstage: int = None,
                           target_line: int = None):
        """
        Search the other line
        :param stage: Main step
        :param substage: Sub step (First 0 electron area, Search other line, Second 0 electron area)
        :param subsubstage: Sub-Sub step (Slope calculation or else)
        :param target_line: line of the other dot
        :return: bool, if we succeed to find the other line or not
        """
        logger.debug(
            f'Stage ({stage}.{substage}.{subsubstage}) - Search the second line: {self._class[target_line]} at a coord ('
            f'{self._get_leftmost_line_coord_str() if target_line == 2 else self._get_bottommost_line_coord_str()})')
        self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - Start 0 area research for {self._class[target_line]}'

        start_coord = [self._bottommost_line_coord, self._leftmost_line_coord][target_line - 1]
        # We already find the other line
        if not start_coord == [None, None]:
            start_x, start_y = start_coord
            self.move_to_coord(start_x, start_y)
            self._line_state = target_line
            slope = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
            if not slope:
                logger.debug(f'Stage ({stage}.{substage}) - Slope calculation for {self._class[self._line_state]}')
                self._step_name = f'Stage ({stage}.{substage}) - Slope calculation for {self._class[self._line_state]}'
                self._search_line_angle()
            x, y = self.diagram.coord_to_voltage(start_coord[0], start_coord[1])
            logger.debug(f'{self._class[self._line_state]} already found at coord ({x:.2f},{y:.2f})')
            return True

        # Start the research
        start_x, start_y = self._bottommost_line_coord if target_line == 2 else self._leftmost_line_coord
        slope = self._line_slope_1 if target_line == 1 else self._line_slope_2
        self.move_to_coord(start_x, start_y)

        directions = self._setup_direction_X()

        # Stop if max exploration steps reach or all directions are stuck (reach corners)
        nb_exploration_steps = 0

        while nb_exploration_steps < self._max_steps_exploration and not Direction.all_stuck(directions):

            # Move and search line in every not stuck directions
            for direction in (d for d in directions if not d.is_stuck):
                nb_exploration_steps += 1
                self._step_descr = f'Init line: {self._class[self._line_state]}\n' \
                                   f'Target line: {self._class[target_line]}\n' \
                                   f'Leftmost line: {self._get_leftmost_line_coord_str()}\n' \
                                   f'Bottommost line: {self._get_bottommost_line_coord_str()}'

                self.move_to_coord(direction.last_x, direction.last_y)  # Go to last position of this direction
                direction.move()  # Move according to the current direction
                direction.last_x, direction.last_y = self.x, self.y  # Save current position for next time
                direction.is_stuck = direction.check_stuck()  # Check if reach a corner

                line_state, _ = self.is_transition_line()

                # Case the second line
                if line_state == target_line:
                    self._nb_line_found_1 += 1 if line_state == 1 else 0
                    self._nb_line_found_2 += 1 if line_state == 2 else 0
                    self._line_state = target_line
                    if not slope:
                        subsubstage += 1
                        logger.debug(
                            f'Stage ({stage}.{substage}.{subsubstage}) - Slope calculation for {self._class[self._line_state]}')
                        self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - Slope calculation for {self._class[self._line_state]}'
                        self._search_line_angle()
                        self._is_bottommost_or_leftmost_line(line_state=self._line_state)
                    return True

                # Case of a crosspoint
                elif line_state == settings.dot_number + 1:
                    subsubstage += 1
                    logger.debug(f'Stage ({stage}.{substage}.{subsubstage}) - Check around crosspoint')
                    self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - Check around crosspoint'
                    self._search_line_angle(force_target_line=target_line)
                    self._is_bottommost_or_leftmost_line(line_state=1)
                    self._is_bottommost_or_leftmost_line(line_state=2)

                    if self._line_around_crosspoint(target_line=target_line):
                        line_slope = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
                        if not line_slope:
                            subsubstage += 1
                            logger.debug(
                                f'Stage ({stage}.{substage}.{subsubstage}) - Slope calculation for {self._class[self._line_state]}')
                            self._step_name = f'Stage ({stage}.{substage}.{subsubstage}) - ' \
                                              f'Slope calculation for {self._class[self._line_state]}'
                            self._search_line_angle()

                        self._nb_line_found_1 += 1 if line_state == 1 else 0
                        self._nb_line_found_2 += 1 if line_state == 2 else 0
                        return True

                # Case of the first line
                elif line_state != 0:
                    self._is_bottommost_or_leftmost_line(line_state=self._line_state)
        logger.debug('Impossible to find the other line')
        return False

    def _search_first_line(self) -> bool:
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
            self._nb_line_found_cross += 1 if line_state == settings.dot_number + 1 else 0
            self._line_state = line_state
            return True

        directions = self._setup_direction_X()

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
                    self._line_state = line_state
                    return True

        return False

    def _get_empty_area(self, stage: int = None, substage: int = None, directions: tuple = None) -> None:
        """
        Search the area of empty charge for the horizontal or vertical state (0,:) or (:,0)
        :param directions: direction left, right, up, down of the line for a specific line (line 1 or line 2)
        """

        nb_line_found = self._nb_line_found_1 if self._line_state == 1 else self._nb_line_found_2
        line_distances = self._line_distances_1 if self._line_state == 1 else self._line_distances_2

        # The line slope of the target empty area
        target_line_slope = self._line_slope_1 if self._line_state == 1 else self._line_slope_2

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
                self._step_descr = f'Target line: {self._class[self._line_state]}\n' \
                                   f'Line slope: {str(target_line_slope)}°\n' \
                                   f'Average line distance: {self._convert_pixel_to_volt(avg_line_distance):.2e} V\n' \
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

                    self._verif_slope(stage, substage, 'bis', [2, 1][self._line_state - 1])

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
                elif line_state == self._line_state:
                    if direction.no_line_count >= 1:
                        nb_line_found += 1
                        line_distances.append(direction.no_line_count)
                        # Stop exploring right if we found enough lines
                        directions[1].is_stuck = directions[1].is_stuck \
                                                 or nb_line_found >= self._max_line_explore_right
                    self._is_bottommost_or_leftmost_line(line_state=self._line_state)
                    direction.no_line_count = 0

                # Case of line but not the target line
                else:
                    self._verif_slope(stage, substage, 'bis', [2, 1][self._line_state - 1])

                    self._is_bottommost_or_leftmost_line(line_state=line_state)

                    direction.no_line_count += 1
                    # Stop to explore this direction if we found more than 1 line and
                    # we found no line in 2x the average line distance in this direction.
                    # TODO could also use the line distance std
                    if nb_line_found > 1 and direction.no_line_count > 2 * avg_line_distance:
                        direction.is_stuck = True

        if self._line_state == 1:
            self._line_distances_1 = line_distances
            self._nb_line_found_1 = nb_line_found
        else:
            self._line_distances_2 = line_distances
            self._nb_line_found_2 = nb_line_found

    def _validate_line(self, stage: int = None, substage: int = None, directions: tuple = None) -> None:
        """
        Validate that the current leftmost or the bottommost line detected is really the leftmost or the bottommost one.
        Try to find a line left by scanning area at regular interval on the left, where we could find a line.
        If a new line is found that way, do the validation again.
        """

        # Setup parameter

        line_distance = self._line_distances_1 if self._line_state == 1 else self._line_distances_2
        line_step_distance = self._get_avg_line_step_distance(line_distances=line_distance) * 2
        line_slope = self._line_slope_1 if self._line_state == 1 else self._line_slope_2
        max_nb_line = [self._max_nb_line_bottommost, self._max_nb_line_leftmost][self._line_state - 1]
        default_step = [self._default_step_y, self._default_step_x][self._line_state - 1]
        default_step_inv = [self._default_step_x, self._default_step_y][self._line_state - 1]

        # Starting test
        nb_steps = 0
        new_line_found = True
        start_point = self._leftmost_line_coord if self._line_state == 2 else self._bottommost_line_coord

        # Case of start point is None
        if not start_point:
            start_point = self.diagram.get_random_starting_point()
            self._leftmost_line_coord = start_point if self._line_state == 2 else self._leftmost_line_coord
            self._bottommost_line_coord = start_point if self._line_state == 1 else self._bottommost_line_coord

        # Unstuck since we are starting at a new location
        directions[2].is_stuck = directions[3].is_stuck = False

        while new_line_found:
            nb_line = 0
            new_line_found = False
            # Both direction start at the leftmost/bottommost point
            directions[2].last_x, directions[2].last_y = start_point
            directions[3].last_x, directions[3].last_y = start_point
            while not new_line_found and not Direction.all_stuck((directions[2], directions[3])):
                for direction in (d for d in (directions[2], directions[3]) if not d.is_stuck):
                    # Check if we reached the maximum number of leftmost search for the current line
                    nb_line += 1

                    if nb_line > self._max_steps_validate_line:
                        return

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

                        # Case of the target line
                        if line_state == self._line_state or line_state == settings.dot_number + 1:

                            coord = self._bottommost_line_coord if self._line_state == 1 else self._leftmost_line_coord

                            if line_state == settings.dot_number + 1:
                                self._is_bottommost_or_leftmost_line(line_state=1)
                                self._is_bottommost_or_leftmost_line(line_state=2)
                            else:
                                self._is_bottommost_or_leftmost_line(line_state=self._line_state)

                            new_coord = self._bottommost_line_coord if self._line_state == 1 else self._leftmost_line_coord

                            # If line isn't the leftmost or bottommost: ignore
                            if coord == new_coord:
                                continue
                            else:
                                direction.last_x, direction.last_y = self.x, self.y
                                self._nb_line_found_1 += 1 if self._line_state == 1 else 0
                                self._nb_line_found_2 += 1 if self._line_state == 2 else 0
                                new_line_found = True
                                start_point = new_coord
                                self._step_descr = f'Target line: {self._class[self._line_state]}\n' \
                                                   f'Coord: {(self.diagram.x_axes[direction.last_x], self.diagram.y_axes[direction.last_y])}\n' \
                                                   f'Leftmost line: {self._get_leftmost_line_coord_str()}\n' \
                                                   f'Bottommost line: {self._get_bottommost_line_coord_str()}\n' \
                                                   f'Line slope: {line_slope:.0f}°\n' \
                                                   f'Avg line dist: {self._convert_pixel_to_volt(line_step_distance):.2e} V'

                                # Debug
                                x, y = self.diagram.coord_to_voltage(coord[0], coord[1])

                                xbis, ybis = self.diagram.coord_to_voltage(self.x, self.y)
                                logger.debug(f'Previous {"leftmost" if self._line_state == 2 else "bottommost"} '
                                             f'line: ({x:.2f}, {y:.2f}), After verification: ({xbis:.2f}, {ybis:.2f})')
                                break
                        # Case a line but not the target line
                        elif line_state == [2, 1][self._line_state - 1]:
                            self._verif_slope(stage, substage, 'bis', [2, 1][self._line_state - 1])
                            self._is_bottommost_or_leftmost_line(line_state=[2, 1][self._line_state - 1])

                        self._move_left_perpendicular_to_line()

                        if self.is_max_left() or self.is_max_down():
                            # Nothing else to see here
                            break

                    if nb_steps > max_nb_line:
                        # Hard break to avoid infinite search in case of bad slope detection (>90°)
                        return

                    if new_line_found:
                        break
        return

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
                     f'{confidence:.2%} at coord ({x:.2f} V, {y:.2f} V)')

        return prediction, confidence

    def _guess_one_electron(self) -> None:
        """
        According to the leftmost line validated and to the bottommost line validated,
         guess a good location for the 1 electron regime.
        Then move to this location.
        """

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

        # Search intersection point (1, 1) electron area

        intersec_point, bottom, left = self._intersection_point()

        # Located the final coordinate of the (1, 1) electron area

        self._final_position()

        # Record intersection point and final coord
        if settings.intersection_plot:
            result = self.diagram.get_charge(self.x, self.y) == ['1_1', '1_2']

            if result and self._nb_plot_good >= 0:
                result = 'Success'
                self._nb_plot_good -= 1
            elif not result and self._nb_plot_bad >= 0:
                result = 'Fail'
                self._nb_plot_bad -= 1
            else:
                return

            l1 = bottom[1]
            l2 = left[1]
            l_point_volt = left[0]
            b_point_volt = bottom[0]
            intersection_point = intersec_point
            final_point = [self.x, self.y]

            self._plot_intersection(result=result, l1=l1, l2=l2, l_point_volt=l_point_volt, b_point_volt=b_point_volt,
                                    i_point=intersection_point, f_point=final_point)

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return

    def _intersection_point(self) -> tuple:
        """
        Place on the intersection point
        """

        time_start = perf_counter()
        x_start, y_start = self.x, self.y
        # Line 1 and bottommost coord
        self.x, self.y = self._bottommost_line_coord
        x_b, y_b = self.get_patch_center()
        state, x_b, y_b = self._enforce_boundary(True, x_b, y_b)
        x_b_volt, y_b_volt = self.diagram.x_axes[x_b], self.diagram.y_axes[y_b]
        bottom_coord = [x_b_volt, y_b_volt]

        # Reconstruct Line 1: (y = a * x + b)
        slope_1, offset_1 = self._line_interpolation(self._line_slope_1, 1, [x_b_volt, y_b_volt])

        # Leftmost coord
        self.x, self.y = self._leftmost_line_coord
        x_l, y_l = self.get_patch_center()
        state, x_l, y_l = self._enforce_boundary(True, x_l, y_l)
        x_l_volt, y_l_volt = self.diagram.x_axes[x_l], self.diagram.y_axes[y_l]
        left_coord = [x_l_volt, y_l_volt]

        # Reconstruct Line 2
        slope_2, offset_2 = self._line_interpolation(self._line_slope_2, 2, [x_l_volt, y_l_volt])

        # Case line parallele
        if slope_1 == slope_2:
            x_volt = x_l
            y_volt = y_b
        else:
            x_volt = (offset_2 - offset_1) / (slope_1 - slope_2)
            y_volt = slope_1 * x_volt + offset_1

        self.x, self.y = self.diagram.voltage_to_coord(x_volt, y_volt)
        _, self.x, self.y = self._enforce_boundary(True, self.x, self.y)
        x, y = self.get_patch_center()
        x_volt, y_volt = self.diagram.x_axes[x], self.diagram.y_axes[y]

        # Reset X, Y
        self.x, self.y = x_start, y_start

        # Comment x, y

        state, x_intersec, y_intersec = self._enforce_boundary(True, x, y)

        logger.debug(f'- Stage Final - \n'
                     f'Intersection point: {x_volt:.2f} V, {y_volt:.2f} V\n'
                     f'Leftmost coord: {str(self._get_leftmost_line_coord_str())}\n'
                     f'Bottommost coord: {str(self._get_bottommost_line_coord_str())}\n'
                     f'Angle Line 1: {self._line_slope_1}° & slope: {slope_1}\n'
                     f'Y1 = {slope_1}.x + {offset_1}\n'
                     f'Angle Line 2: {self._line_slope_2}° & slope: {slope_2}\n'
                     f'Y2 = {slope_2}.x + {offset_2}')

        self.move_to_coord(x_intersec, y_intersec)

        # Record Intersection
        self._step_descr = f'---- Intersection point -------\n' \
                           f'Coord: {x_volt:.2f}V,{y_volt:.2f}V\n' \
                           f'Leftmost coord: {x_l_volt, y_l_volt}\n' \
                           f'Bottommost coord: {x_b_volt, y_b_volt}'

        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr

        time_data_fetched = time_data_processed = perf_counter()

        self._scan_history.append(StepHistoryEntry(
            (x_intersec, y_intersec), -1, 1, False, False, False,
            True, step_description, time_start, time_data_fetched, time_data_processed,
            isinstance(self.diagram, DiagramOnline)
        ))

        return [x_intersec, y_intersec], [bottom_coord, [slope_1, offset_1]], [left_coord, [slope_2, offset_2]]

    def _final_position(self) -> None:
        """
        Positioning in the (1, 1) electron zone
        """

        time_start = perf_counter()

        x_intersec, y_intersec = self.x, self.y

        ratio = 3 / 6

        # Positioning along the first row.
        self._line_state = 1

        self._move_down_follow_line(
            ceil(self._default_step_x * self._get_avg_line_step_distance(line_distances=self._line_distances_2) * ratio)
        )

        # Positioning along the second row.
        self._line_state = 2
        self._move_up_follow_line(
            ceil(self._default_step_x * self._get_avg_line_step_distance(line_distances=self._line_distances_1) * ratio)
        )

        x, y = self.x, self.y
        state, self.x, self.y = self._enforce_boundary(True, x, y)

        x_v = self.diagram.x_axes[self.x]
        y_v = self.diagram.y_axes[self.y]

        self._step_descr = f' ---- Final Coord ----\n' \
                           f'Coord: {x_v:.2f} V,{y_v:.2f} V\n' \
                           f'Leftmost Line: {str(self._get_leftmost_line_coord_str())}\n' \
                           f'Bottommost Line: {str(self._get_bottommost_line_coord_str())}\n' \
                           f'Intersection point: {x_intersec:.2f} V, {y_intersec:.2f} V'

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

        return

    # ========================== #
    # --     Main function    -- #
    # ========================== #

    def _tune(self) -> Tuple[int, int]:
        """
        Tuning for  2 dots

        :return: couple (x, y) => Final coord
        """

        # Stage (1) - First line detection
        # Search a line or a crosspoint, if none found return a random position
        self._run_nb += 1
        if not self._first_step_():
            if self._line_state == settings.dot_number + 1:
                return self.x, self.y
            else:
                return self.diagram.get_random_starting_point()

        # Stage (2) - Search empty area
        # Search empty area for each quantum dots
        if not self._second_step_():
            return self.x, self.y

        # Stage (Final) - Search (1, 1) electron regime
        # Move to (1, 1) electron coordinates based on the leftmost and bottommost line found
        self._third_step_()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)
        return self.get_patch_center()

    def _first_step_(self) -> bool:
        """
        The first step of the algorithm. The goal here, is to find a line
        :return: if the step succeed or not
        """
        stage, substage = 1, 1

        logger.debug(f'Stage ({stage}.{substage}) - Search First line')
        self._step_name = f'Stage ({stage}.{substage}) - Search First line'

        if not self._search_first_line():
            # No line found, we return a random point
            return False

        # If the first line is a crosspoint, we search around the crosspoint a line,
        # else we calculate the slope of the line

        elif self._line_state == settings.dot_number + 1:  # Crosspoint
            substage += 1
            logger.debug(f'Stage ({stage}.{substage}) - Case of first line is a crosspoint')
            self._step_name = f'Stage ({stage}.{substage}) - Case of first line is a crosspoint'
            if not self._line_around_crosspoint():
                # If we are stuck inside the crosspoint, we return the coord of the crosspoint
                return False

        # We have now a Line 1 or 2
        return True

    def _second_step_(self) -> bool:
        """
        The second step: Find the 0 electron area for each line
        :return: if the step succeed or not
        """

        substage = subsubstage = 1

        # At the beginning of this function we should be on a line.
        # Since this is the first we met, this is the leftmost and the bottommost by default.
        self._is_bottommost_or_leftmost_line(line_state=self._line_state)

        # (1) First 0 electron area
        logger.debug(f'Stage (2.{substage}) - Search empty area for {self._class[self._line_state]} dots')
        self._step_name = f'Stage (2.{substage}) - Search {str(tuple([0] * settings.dot_number))} electron area'
        if not self._search_0_area(stage=2, substage=substage, subsubstage=subsubstage):
            self.x, self.y = self._get_bottom_left_coord()
            return False

        # (2) Search second line
        # We change the target empty area, change line: 1->2 or 2->1
        substage += 1
        logger.debug(f'Stage (2.{substage}) - Target line change {self._class[self._line_state]} '
                     f'into {self._class[[2, 1][self._line_state - 1]]}')
        self._step_name = f'Stage (2.{substage}) - Target line change {self._class[self._line_state]} ' \
                          f'into {self._class[[2, 1][self._line_state - 1]]}'
        target_line = [2, 1][self._line_state - 1]
        if not self._search_other_line(stage=2, substage=substage, subsubstage=subsubstage, target_line=target_line):
            # We fail to find the other line, so we return the bottommost y and the leftmost x
            logger.debug(f'Fail to find {self._class[[2, 1][self._line_state - 1]]}')
            self.x, self.y = self._get_bottom_left_coord()
            return False

        # (3) Second 0 electron area
        substage += 1
        logger.debug(f'Stage (2.{substage}) - Search empty area for {self._class[self._line_state]} dots')
        self._step_name = f'Stage (2.{substage}) - Search {str(tuple([0] * settings.dot_number))} electron area'
        if not self._search_0_area(stage=2, substage=substage, subsubstage=subsubstage):
            self.x, self.y = self._get_bottom_left_coord()
            return False
        return True

    def _third_step_(self) -> None:
        """
        Third and final step: Interpolate the intersection point and find the (1, 1) electron area
        :return: None
        """
        # Stage (Final) - Search (1, 1) electron regime
        # Move to (1, 1) electron coordinates based on the leftmost and bottommost line found
        logger.debug(f'Final stage - Get {str(tuple([1] * settings.dot_number))} area')
        self._step_name = f'Final stage - Get {str(tuple([1] * settings.dot_number))} area'

        self._guess_one_electron()

        if settings.stat_angle_plot:
            if not len(self._sigma):
                self._sigma.append([self._line_slope_1, [0, 0, 0, 0], self._line_slope_2, [0, 0, 0, 0]])
            else:
                angle1 = [self._sigma[i][0] for i in range(len(self._sigma))]
                angle2 = [self._sigma[i][2] for i in range(len(self._sigma))]
                metric_1 = self.metrics(angle1, self._line_slope_default_1)
                metric_2 = self.metrics(angle2, self._line_slope_default_2)
                self._sigma.append([self._line_slope_1, metric_1, self._line_slope_2, metric_2])
            self._plot_stat()

        # Enforce the boundary policy to make sure the final guess is in the diagram area
        self._enforce_boundary_policy(force=True)


class JumpNDots_slope(JumpNDots):
    """
    Jump for 2 dots but with slope estimation thank to a network
    """
    _slope_model: ClassifierNN = None
    _nb_scan_estimation: int = 0
    debug_patch: bool = False

    def reset_procedure(self):
        super().reset_procedure()
        self._bottommost_line_coord = [None, None]
        self._leftmost_line_coord = [None, None]
        self._line_state = 0
        self._slope_model = self._init_slope_model() if not self._slope_model else self._slope_model
        self._nb_scan_estimation = 0

        if settings.research_group == 'eva_dupont_ferrier':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 10
            self._line_slope_default_2 = 100
            # Prior assumption about distance between lines
            self._line_distances_1 = [10]
            self._line_distances_2 = [20]
            # Prior assumption about the estimation slope error
            self._error_1 = -3
            self._error_2 = 24
        elif settings.research_group == 'louis_gaudreau':
            self._line_slope_1 = None
            self._line_slope_2 = None
            # Prior assumption about line direction
            self._line_slope_default_1 = 149 + settings.delta * random.choice([-1, 1])
            self._line_slope_default_2 = 119 + settings.delta * random.choice([-1, 1])
            # Prior assumption about distance between lines
            self._line_distances_1 = [14]
            self._line_distances_2 = [12]
            # Prior assumption about the estimation slope error
            self._error_1 = 0  # 14
            self._error_2 = 0  # 18
            if settings.logger_console_level == 'debug':
                logger.warning(f'# ========= New tuning ========= #')
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
            # Prior assumption about the estimation slope error
            self._error_1 = 0
            self._error_2 = 0

    def _search_line_angle(self, force_target_line: Optional[int] = None) -> None:
        """
        Estimate the direction of the current line and update the _line_slope attribut if the measurement looks valid
        :param force_target_line: Case of crosspoint
        """

        target_line = self._line_state if not force_target_line else force_target_line

        slope = []
        if target_line == settings.dot_number + 1:
            # TODO implement crosspoint problem
            if target_line == 1:
                self._line_slope_1 = self._line_slope_default_1
            else:
                self._line_slope_2 = self._line_slope_default_2
            return

        direction = 0
        steps_in_direction = 1
        # Initialize the number of steps taken in the current direction
        steps_taken = 0
        x_start, y_start = self.x, self.y
        # Iterate over each pixel
        for i in range(settings.nb_spiral ** 2):
            if not self.is_inside():
                logger.debug(f'Confirm patch {i}')
                line, _ = self.is_transition_line()
                if line == target_line:
                    angle, _ = self._angle_estimation(target_line)
                    slope.append(angle)
                elif line == settings.dot_number + 1:
                    # TODO implement crosspoint problem
                    angle = self._line_slope_default_1 if target_line == 1 else self._line_slope_default_2
                    slope.append(angle)
            # Move in the current direction
            if direction == 0:
                self.move_right()
            elif direction == 1:
                self.move_up()
            elif direction == 2:
                self.move_left()
            elif direction == 3:
                self.move_down()

            # Update the number of steps taken in the current direction
            steps_taken += 1
            # If we have taken the required number of steps in the current direction, change direction
            if steps_taken == steps_in_direction:
                # Change direction
                direction = (direction + 1) % 4

                # Reset the number of steps taken in the current direction
                steps_taken = 0

                # If we are now moving right or left, increase the number of steps in the current direction
                if direction % 2 == 0:
                    steps_in_direction += 1

        angle = np.mean(slope)
        if settings.logger_console_level == 'debug':
            text = f'Angle = {angle} for {self._class[target_line]}'
            if target_line == 1:
                text += f': $\\Deltat$ = {self._line_slope_default_1 - angle}'
            else:
                text += f': $\\Deltat$ = {self._line_slope_default_2 - angle}'
            logger.warning(text)
        self.move_to_coord(x_start, y_start)
        if settings.patch_verif_plot:
            self.verif_patch(self.diagram.get_patch((self.x, self.y), self.patch_size, normalized=False),
                             angle, target_line, 'Validity')
        if target_line == 1:
            self._line_slope_1 = angle
        else:
            self._line_slope_2 = angle

        return

    def _angle_estimation(self, target_line) -> Tuple[float, float]:
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        :return:
        """
        if not self._slope_model:
            logger.debug('Load slope network')
            self._slope_model = self._init_slope_model()

        time_start = perf_counter()

        result: Tuple[bool, float]
        if settings.slope_estimation_use_oracle:
            if target_line == 1:
                angle = self._line_slope_default_1
            else:
                angle = self._line_slope_default_2
        else:
            with torch.no_grad():
                # Cut the patch area and send it to the model for inference
                patch = self.diagram.get_patch((self.x, self.y), self.patch_size, normalized=False)
                patch_input_1 = patch.clone()
                patch_input = norm_value(patch_input_1, self.diagram.values.min(), self.diagram.values.max())
                time_data_fetched = perf_counter()
                size_x, size_y = self.patch_size
                # Reshape as valid input for the model (patch x . patch y)
                patch_input = patch_input.view(1, -1)
                # Send to the model for inference
                prediction = self._slope_model(patch_input)
                # Extract data from pytorch tensor

                angle = prediction.item() * 360 - 90
                angle = self._validated_angle(angle)
                if settings.patch_verif_plot and self.debug_patch:
                    self.verif_patch(norm_value(patch_input_1, self.diagram.values.min(),
                                                self.diagram.values.max()), angle, target_line,
                                     settings.patch_verif_plot)
                # confidence = confidence.item()
                time_data_processed = perf_counter()

            # is_above_confidence_threshold = self.model.is_above_confident_threshold(slope, confidence)
        confidence = 1
        # Record the diagram scanning activity.
        # TODO record slope estimation
        # self._record_slope_estimation(slope, confidence, time_start=time_start, time_data_fetched=time_data_fetched, time_data_processed=time_data_processed)

        x, y = self.diagram.coord_to_voltage(self.x, self.y)
        logger.debug(f'Patch {self.get_nb_steps():03}: '
                     f'Slope estimation = {angle}° (Network output: {angle / 360}) '
                     f'at coord ({self.x} , {self.y} V)')

        return angle, confidence

    def _init_slope_model(self) -> ClassifierNN:
        """
        Initialization of the neural network used for the slope estimation
        :return: slope model
        """
        # Automatically chooses if auto
        if settings.device is None or settings.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(settings.device)

        model = init_slope_model()

        if not load_slope_network(network=model, file_path=settings.slope_network_cache_path, device=device):
            raise f'Network cache not found in "{settings.slope_network_cache_path}"'

        return model

    def _record_angle_estimation(self, slope: float, confidence: float,
                                 time_start, time_data_fetched, time_data_processed,
                                 ground_truth, soft_truth_larger, soft_truth_smaller, is_above_confidence_threshold,
                                 ) -> None:
        raise f'Not implemented'

    def _validated_angle(self, angle: float) -> float:
        # ((angle * 360) % 180) / 360
        return angle % 180
