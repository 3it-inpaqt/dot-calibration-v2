from enum import Enum
from random import randrange
from typing import List, Optional, Tuple

import torch

from classes.classifier import Classifier
from classes.diagram import Diagram
from plots.data import plot_diagram


class BoundaryPolicy(Enum):
    """ Enumeration of policies to apply if a scan is requested outside the diagram borders. """
    HARD = 0  # Don't allow going outside the diagram
    SOFT_RANDOM = 1  # Allow going outside the diagram and fill unknown data with random values
    SOFT_VOID = 2  # Allow going outside the diagram and fill unknown data with 0


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
        self.patch_size = patch_size
        self.label_offsets = label_offsets
        self.is_oracle_enable = is_oracle_enable
        self.boundary_policy = boundary_policy
        self.x = None
        self.y = None

        # The default move step. If None the (patch size - offset) is use.
        if default_step is None:
            offset_x, offset_y = self.label_offsets
            self._default_step_x, self._default_step_y = patch_size
            self._default_step_x -= offset_x
            self._default_step_y -= offset_y
        else:
            self._default_step_x, self._default_step_y = default_step

        # Performance statistic
        # History: ((x, y), (line_detected, confidence))
        self._scan_history: List[Tuple[Tuple[int, int], Tuple[bool, float]]] = []

    def __str__(self) -> str:
        return f'{type(self).__name__} ({"Oracle" if self.is_oracle_enable else self.model})'

    def reset_procedure(self) -> None:
        """
        Reset procedure statistics. Make it ready to start a new one.
        """
        self.x = None
        self.y = None
        self._scan_history.clear()

    def is_transition_line(self, diagram: Diagram) -> (bool, float):
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :param diagram: The diagram to consider.
        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """

        # Check coordinates according to the current policy.
        # They could be changed to fit inside the diagram if necessary
        self._enforce_boundary_policy(diagram)

        result: Tuple[bool, float]
        if self.is_oracle_enable:
            # Check the diagram label and return the classification with full confidence
            result = diagram.is_line_in_patch((self.x, self.y), self.patch_size, self.label_offsets), 1
        else:
            # Cut the patch area and send it to the model for inference
            patch = diagram.get_patch((self.x, self.y), self.patch_size)
            # Reshape as valid input for the model (batch size, chanel, patch x, patch y)
            size_x, size_y = self.patch_size
            patch = torch.Tensor(patch).view((1, 1, size_x, size_y))
            # Send to the model for inference
            result = self.model.infer(patch)

        # Record the diagram scanning activity.
        self._scan_history.append(((self.x, self.y), result))

        return result

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

    def is_max_right(self, diagram: Diagram) -> bool:
        """
        :return: True if the current coordinates have reach the right border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.x >= len(diagram.x_axes) - self.patch_size[0] - 1

        raise ValueError(f'Unknown policy {self.boundary_policy}')

    def is_max_up(self, diagram: Diagram) -> bool:
        """
        :return: True if the current coordinates have reach the top border of the diagram. False if not.
        """

        # No max for soft policies
        if self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return False

        if self.boundary_policy is BoundaryPolicy.HARD:
            return self.y >= len(diagram.y_axes) - self.patch_size[1] - 1

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

    def _enforce_boundary_policy(self, diagram: Diagram, force: bool = False) -> bool:
        """
        Check if the coordinates violate the boundary policy. If they do, move the coordinates according to the policy.
        :param diagram: The current diagram.
        :param force: If True the boundaries are forced, no matter the currant policy.
        :return: True if the coordinates are acceptable in the current policy, False if not.
        """

        # Always good for soft policies
        if not force and self.boundary_policy in [BoundaryPolicy.SOFT_VOID, BoundaryPolicy.SOFT_RANDOM]:
            return True

        if force or self.boundary_policy is BoundaryPolicy.HARD:
            patch_size_x, patch_size_y = self.patch_size
            max_x = len(diagram.x_axes) - patch_size_x - 1
            max_y = len(diagram.y_axes) - patch_size_y - 1

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

    def get_random_coordinates_in_diagram(self, diagram: Diagram) -> Tuple[int, int]:
        """
        Generate (pseudo) random coordinates for the top left corder of a patch inside the diagram.
        :param diagram: The diagram to consider.
        :return: The (pseudo) random coordinates.
        """
        patch_size_x, patch_size_y = self.patch_size
        max_x = len(diagram.x_axes) - patch_size_x - 1
        max_y = len(diagram.y_axes) - patch_size_y - 1
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

    def plot_step_history(self, d: Diagram, final_coord: Tuple[int, int]) -> None:
        """
        Plot the diagram with the tuning steps of the current procedure.
        :param d: The diagram to plot.
        """
        plot_diagram(d.x_axes, d.y_axes, d.values, d.file_basename, 'nearest', d.x_axes[1] - d.x_axes[0],
                     transition_lines=d.transition_lines, steps_history=self._scan_history, final_coord=final_coord,
                     show_offset=False)

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :param diagram: The diagram to use for the training
        :param start_coord: The starting coordinates (top right of the patch square)
        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError
