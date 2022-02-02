from dataclasses import dataclass
from enum import Enum
from random import randrange
from typing import Callable, Iterable, List, Optional, Tuple

import torch

from classes.classifier import Classifier
from classes.diagram import Diagram
from plots.data import plot_diagram, plot_diagram_step_animation
from utils.settings import settings


class BoundaryPolicy(Enum):
    """ Enumeration of policies to apply if a scan is requested outside the diagram borders. """
    HARD = 0  # Don't allow going outside the diagram
    SOFT_RANDOM = 1  # Allow going outside the diagram and fill unknown data with random values
    SOFT_VOID = 2  # Allow going outside the diagram and fill unknown data with 0


@dataclass(frozen=True)
class HistoryEntry:
    coordinates: Tuple[int, int]
    model_classification: bool
    model_confidence: bool
    ground_truth: bool


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


class AutotuningProcedure:
    """ Procedure for autotuning tuning of static stability diagrams. """

    def __init__(self, model: Optional[Classifier],
                 patch_size: Tuple[int, int],
                 label_offsets: Tuple[int, int] = (0, 0),
                 is_oracle_enable: bool = False,
                 default_step: Optional[Tuple[int, int]] = None,
                 boundary_policy: BoundaryPolicy = BoundaryPolicy.HARD):
        """
        Create a new procedure.

        :param model: The line detection model to use in this procedure.
        :param patch_size: The patch size to use in this procedure (should match with model expectation)
        :param is_oracle_enable: If true the line detection use directly on the labels instead of the model inference.
        :param default_step: The default move step. If None the (patch size - offset) is use.
        :param boundary_policy: The policy to apply if a scan is requested outside the diagram borders.
        """
        if model is None and not is_oracle_enable:
            raise ValueError('If no model is provided, the oracle should be explicitly enable')
        if model is not None and is_oracle_enable:
            raise ValueError('If a model is provided, the oracle should not be enable')

        self.model: Classifier = model
        self.patch_size: Tuple[int, int] = patch_size
        self.label_offsets: Tuple[int, int] = label_offsets
        self.is_oracle_enable: bool = is_oracle_enable
        self.boundary_policy: BoundaryPolicy = boundary_policy
        self.diagram: Optional[Diagram] = None
        self.x: Optional[int] = None
        self.y: Optional[int] = None

        # The default move step. If None the (patch size - offset) is use.
        if default_step is None:
            offset_x, offset_y = self.label_offsets
            self._default_step_x, self._default_step_y = patch_size
            self._default_step_x -= offset_x * 2
            self._default_step_y -= offset_y * 2
        else:
            self._default_step_x, self._default_step_y = default_step

        # Performance statistic (See HistoryEntry dataclass)
        self._scan_history: List[HistoryEntry] = []

    def __str__(self) -> str:
        return f'{type(self).__name__} ({"Oracle" if self.is_oracle_enable else self.model})'

    def reset_procedure(self) -> None:
        """
        Reset procedure statistics. Make it ready to start a new one.
        """
        self.diagram = None
        self.x = None
        self.y = None
        self._scan_history.clear()

    def is_transition_line(self) -> (bool, float):
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """

        # Check coordinates according to the current policy.
        # They could be changed to fit inside the diagram if necessary
        self._enforce_boundary_policy()

        # Fetch ground truth from labels
        ground_truth = self.diagram.is_line_in_patch((self.x, self.y), self.patch_size, self.label_offsets)

        result: Tuple[bool, float]
        if self.is_oracle_enable:
            # Oracle use ground truth with full confidence
            prediction = ground_truth
            confidence = 1
        else:
            with torch.no_grad():
                # Cut the patch area and send it to the model for inference
                patch = self.diagram.get_patch((self.x, self.y), self.patch_size)
                # Reshape as valid input for the model (batch size, chanel, patch x, patch y)
                size_x, size_y = self.patch_size
                patch = torch.Tensor(patch).view((1, 1, size_x, size_y))
                # Send to the model for inference
                prediction, confidence = self.model.infer(patch, settings.bayesian_nb_sample_test)
                # Extract data from pytorch tensor
                prediction = prediction.item()
                confidence = confidence.item()

        # Record the diagram scanning activity.
        self._scan_history.append(HistoryEntry((self.x, self.y), prediction, confidence, ground_truth))

        return prediction, confidence

    def get_patch_center(self) -> Tuple[int, int]:
        """
        :return: The center of the patch in the current position.
        """
        patch_size_x, patch_size_y = self.patch_size
        return self.x + (patch_size_x // 2), self.y + (patch_size_y // 2)

    def move_left(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the left.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.x -= step_size if step_size is not None else self._default_step_x

    def move_right(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the right.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.x += step_size if step_size is not None else self._default_step_x

    def move_up(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the top.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.y += step_size if step_size is not None else self._default_step_y

    def move_down(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the bottom.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.y -= step_size if step_size is not None else self._default_step_y

    def move_up_left(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the top left.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.move_up(step_size)
        self.move_left(step_size)

    def move_up_right(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the top right.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.move_up(step_size)
        self.move_right(step_size)

    def move_down_left(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the bottom left.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.move_down(step_size)
        self.move_left(step_size)

    def move_down_right(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the bottom right.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        self.move_down(step_size)
        self.move_right(step_size)

    def move_to_coord(self, x: int = None, y: int = None) -> None:
        """
        Move the current coordinate to a specific position.
        Could change x or y or both.

        :param x: The new x coordinate.
        :param y: The new y coordinate.
        """

        if x is None and y is None:
            raise ValueError('Move to coordinates called but no coordinates provided (need at least x or y)')

        if x is not None:
            self.x = x
        if y is not None:
            self.y = y

    def is_max_left(self) -> bool:
        """
        :return: True if the current coordinates have reach the left border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.x <= 0

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def is_max_right(self) -> bool:
        """
        :return: True if the current coordinates have reach the right border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.x >= len(self.diagram.x_axes) - self.patch_size[0] - 1

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def is_max_up(self) -> bool:
        """
        :return: True if the current coordinates have reach the top border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.y >= len(self.diagram.y_axes) - self.patch_size[1] - 1

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def is_max_down(self) -> bool:
        """
        :return: True if the current coordinates have reach the bottom border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.y <= 0

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def is_max_up_left(self):
        """
        :return: True if the current coordinates have reach the top left corner of the diagram. False if not.
        """
        return self.is_max_up() and self.is_max_left()

    def is_max_up_right(self):
        """
        :return: True if the current coordinates have reach the top right corner of the diagram. False if not.
        """
        return self.is_max_up() and self.is_max_right()

    def is_max_down_left(self):
        """
        :return: True if the current coordinates have reach the bottom left corner of the diagram. False if not.
        """
        return self.is_max_down() and self.is_max_left()

    def is_max_down_right(self):
        """
        :return: True if the current coordinates have reach the bottom right corner of the diagram. False if not.
        """
        return self.is_max_down() and self.is_max_right()

    def _enforce_boundary_policy(self, force: bool = False) -> bool:
        """
        Check if the coordinates violate the boundary policy. If they do, move the coordinates according to the policy.
        :param force: If True the boundaries are forced, no matter the currant policy.
        :return: True if the coordinates are acceptable in the current policy, False if not.
        """

        # Always good for soft policies
        if not force and self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return True

        if force or self.boundary_policy is BoundaryPolicy.HARD:
            patch_size_x, patch_size_y = self.patch_size
            max_x = len(self.diagram.x_axes) - patch_size_x - 1
            max_y = len(self.diagram.y_axes) - patch_size_y - 1

            match_policy = True
            if self.x < 0:
                self.x = 0
                match_policy = False
            elif self.x > max_x:
                self.x = max_x
                match_policy = False
            if self.y < 0:
                self.y = 0
                match_policy = False
            elif self.y > max_y:
                self.y = max_y
                match_policy = False

            return match_policy

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def get_random_coordinates_in_diagram(self) -> Tuple[int, int]:
        """
        Generate (pseudo) random coordinates for the top left corder of a patch inside the diagram.
        :return: The (pseudo) random coordinates.
        """
        patch_size_x, patch_size_y = self.patch_size
        max_x = len(self.diagram.x_axes) - patch_size_x - 1
        max_y = len(self.diagram.y_axes) - patch_size_y - 1
        return randrange(max_x), randrange(max_y)

    def get_nb_steps(self) -> int:
        """
        :return: The number of steps completed for the current procedure.
        """
        return len(self._scan_history)

    def get_area_scanned(self) -> int:
        """
        :return: The number of pixel scanned so far for the current procedure.
        """
        return self.get_nb_steps() * self.patch_size[0] * self.patch_size[1]

    def get_nb_line_detection_success(self) -> int:
        """ Return the number of successful line detection """
        return len([e for e in self._scan_history if e.model_classification == e.ground_truth])

    def plot_step_history(self, final_coord: Tuple[int, int], success_tuning: bool, plot_vanilla: bool = True) -> None:
        """
        Plot the diagram with the tuning steps of the current procedure.

        :param final_coord: The final coordinate of the tuning procedure
        :param success_tuning: Result of the tuning (True = Success)
        :param plot_vanilla: If True, also plot the diagram with no label and steps
        """
        d = self.diagram
        name = f'{self.diagram.file_basename} steps {"GOOD" if success_tuning else "FAIL"}'

        if plot_vanilla:
            # diagram
            plot_diagram(d.x_axes, d.y_axes, d.values, f'{d.file_basename}', 'nearest', d.x_axes[1] - d.x_axes[0])

        # diagram + label + step with classification color
        plot_diagram(d.x_axes, d.y_axes, d.values, name, 'nearest', d.x_axes[1] - d.x_axes[0],
                     transition_lines=d.transition_lines, scan_history=self._scan_history, final_coord=final_coord,
                     show_offset=False, history_uncertainty=False)
        # label + step with classification color and uncertainty
        plot_diagram(d.x_axes, d.y_axes, None, name + ' uncertainty', 'nearest',
                     d.x_axes[1] - d.x_axes[0], transition_lines=d.transition_lines, scan_history=self._scan_history,
                     final_coord=final_coord, show_offset=False, history_uncertainty=True)

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        """
        Plot the animated diagram with the tuning steps of the current procedure.

        :param final_coord: The final coordinate of the tuning procedure
        :param success_tuning: Result of the tuning (True = Success)
        """

        name = f'{self.diagram.file_basename} steps {"GOOD" if success_tuning else "FAIL"}'
        # Generate a gif image
        plot_diagram_step_animation(self.diagram, name, self._scan_history,
                                    final_coord)

    def setup_next_tuning(self, diagram: Diagram, start_coord: Optional[Tuple[int, int]] = None) -> None:
        """
        Set up the starting point and the diagram of the next tuning run.
        This action is revert by reset_procedure.

        :param diagram: The stability diagram to explore.
        :param start_coord: The starting coordinates (top right of the patch square). If None, random coordinates are
        set inside the diagram.
        """
        self.diagram = diagram
        self.x, self.y = self.get_random_coordinates_in_diagram() if start_coord is None else start_coord

    def tune(self) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError
