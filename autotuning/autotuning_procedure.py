from multiprocessing import Pool
from random import randrange
from typing import List, Optional, Tuple

import torch

from classes.classifier import Classifier
from classes.data_structures import AutotuningResult, BoundaryPolicy, StepHistoryEntry
from datasets.diagram import Diagram
from plots.data import plot_diagram, plot_diagram_step_animation
from runs.run_line_task import get_cuda_device
from utils.misc import get_nb_loader_workers
from utils.settings import settings


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

        # Text description of the current activity for plotting
        self._step_name: str = 'Not started'
        self._step_descr: str = ''

        # The default move step. If None the (patch size - offset) is use.
        if default_step is None:
            offset_x, offset_y = self.label_offsets
            self._default_step_x, self._default_step_y = patch_size
            self._default_step_x -= offset_x * 2
            self._default_step_y -= offset_y * 2
        else:
            self._default_step_x, self._default_step_y = default_step

        # Performance statistic (See StepHistoryEntry dataclass)
        self._scan_history: List[StepHistoryEntry] = []
        self._batch_pending: List[Tuple[int, int]] = []

        # Initialise procedure parameters (could be useful for child classes)
        self.reset_procedure()

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
        self._batch_pending.clear()
        self._step_name = 'Not started'
        self._step_descr: str = ''

    def is_transition_line(self) -> Tuple[bool, float]:
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """

        # Check coordinates according to the current policy.
        # They could be changed to fit inside the diagram if necessary
        self._enforce_boundary_policy()

        # Fetch ground truth from labels
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

        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr
        self._scan_history.append(StepHistoryEntry((self.x, self.y), prediction, confidence, ground_truth,
                                                   soft_truth_larger, soft_truth_smaller, step_description))

        return prediction, confidence

    def add_to_inference_batch(self) -> None:
        """
        Add current coordinate to the pending inference. Useful to speed up processing with batch inference.
        Nothing is processed until "is_transition_line_batch" is called.
        """

        # Check coordinates according to the current policy.
        # They could be changed to fit inside the diagram if necessary
        self._enforce_boundary_policy()

        # Add to batch pending for grouped inference
        self._batch_pending.append((self.x, self.y))

    def is_transition_line_batch(self) -> List[Tuple[bool, float]]:
        """
        Try to detect a line in a batch of diagram sub-area using the current model or the oracle.
        Build batch using coordinates in _batch_pending.
        :return: A list of results with confidence. Empty list if no inference pending.
        """

        if len(self._batch_pending) == 0:
            return []

        # Fetch data and ground truths (ground_truth, soft_truth_larger, soft_truth_smaller)
        ground_truths = []
        size_x, size_y = self.patch_size
        patches = torch.zeros((len(self._batch_pending), size_x, size_y), device=get_cuda_device())
        for i, (x, y) in enumerate(self._batch_pending):
            ground_truths.append(self.get_ground_truths(x, y))
            patches[i] = self.diagram.get_patch((x, y), self.patch_size)

        if self.is_oracle_enable:
            # Oracle use ground truth with full confidence
            predictions = next(zip(*ground_truths))  # Get the ground truth section only (first item of each sublist)
            confidences = [1] * len(ground_truths)
        else:
            with torch.no_grad():
                # Send to the model for inference
                predictions, confidences = self.model.infer(patches, settings.bayesian_nb_sample_test)
                # Extract data from GPU and convert to list
                predictions = predictions.tolist()
                confidences = confidences.tolist()

        # Record the diagram scanning activity.
        decr = ('\n    > ' + self._step_descr.replace('\n', '\n    > ')) if len(self._step_descr) > 0 else ''
        step_description = self._step_name + decr
        for (x, y), pred, conf, (truth, truth_larger, truth_smaller) in \
                zip(self._batch_pending, predictions, confidences, ground_truths):
            self._scan_history.append(
                StepHistoryEntry((x, y), pred, conf, truth, truth_larger, truth_smaller, step_description))

        self._batch_pending.clear()  # Empty pending
        return list(zip(predictions, confidences))

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

    def is_max_up_or_left(self):
        """
        :return: True if the current coordinates have reach the top or the left of the diagram. False if not.
        """
        return self.is_max_up() or self.is_max_left()

    def is_max_up_right(self):
        """
        :return: True if the current coordinates have reach the top right corner of the diagram. False if not.
        """
        return self.is_max_up() and self.is_max_right()

    def is_max_up_or_right(self):
        """
        :return: True if the current coordinates have reach the top or the right of the diagram. False if not.
        """
        return self.is_max_up() or self.is_max_right()

    def is_max_down_left(self):
        """
        :return: True if the current coordinates have reach the bottom left corner of the diagram. False if not.
        """
        return self.is_max_down() and self.is_max_left()

    def is_max_down_or_left(self):
        """
        :return: True if the current coordinates have reach the bottom or the left the diagram. False if not.
        """
        return self.is_max_down() or self.is_max_left()

    def is_max_down_right(self):
        """
        :return: True if the current coordinates have reach the bottom right corner of the diagram. False if not.
        """
        return self.is_max_down() and self.is_max_right()

    def is_max_down_or_right(self):
        """
        :return: True if the current coordinates have reach the bottom or the right the diagram. False if not.
        """
        return self.is_max_down() or self.is_max_right()

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

    def get_ground_truths(self, x, y) -> (bool, bool, bool):
        """
        Get the ground truths of a specific position on the current diagram according to the labels.
        Come in 3 flavors: the real ground truth and 2 the "near" ground truths for detect soft errors.

        :param x: The x coordinate of the position to check.
        :param y: The y coordinate of the position to check.
        :return: The ground truth, the ground truth with a larger active area (smaller offset), the ground truth with a
         smaller active area (larger offset).
        """
        # Fetch ground truth from labels
        ground_truth = self.diagram.is_line_in_patch((x, y), self.patch_size, self.label_offsets)

        # Also fetch soft truth to detect error of type "almost good" when the line is near to the active area
        # +2 and -2 pixel to the offset, with limit to 0 and patch size - 2.
        soft_offsets_smaller = tuple(max(off - 2, 0) for off in self.label_offsets)
        soft_offsets_larger = tuple(min(off + 2, s - 2) for s, off in zip(self.patch_size, self.label_offsets))
        soft_truth_larger = self.diagram.is_line_in_patch((x, y), self.patch_size, soft_offsets_smaller)
        soft_truth_smaller = self.diagram.is_line_in_patch((x, y), self.patch_size, soft_offsets_larger)

        return ground_truth, soft_truth_larger, soft_truth_smaller

    def nb_pending(self) -> int:
        """ Return the number of patch inference pending. """
        return len(self._batch_pending)

    def plot_step_history(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        """
        Plot the diagram with the tuning steps of the current procedure.

        :param final_coord: The final coordinate of the tuning procedure
        :param success_tuning: Result of the tuning (True = Success)
        """
        d = self.diagram
        values = d.values.cpu()
        name = f'{self.diagram.file_basename} steps {"GOOD" if success_tuning else "FAIL"}\n{self}'

        # Parallel plotting for speed.
        with Pool(get_nb_loader_workers()) as pool:
            # diagram + label + step with classification color
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': values, 'image_name': name,
                                   'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                   'transition_lines': d.transition_lines, 'scan_history': self._scan_history,
                                   'final_coord': final_coord, 'show_offset': False, 'history_uncertainty': False})
            # label + step with classification color and uncertainty
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None,
                                   'image_name': name + ' uncertainty', 'interpolation_method': 'nearest',
                                   'pixel_size': d.x_axes[1] - d.x_axes[0], 'transition_lines': d.transition_lines,
                                   'scan_history': self._scan_history, 'final_coord': final_coord, 'show_offset': False,
                                   'history_uncertainty': True})
            # step with error and soft error color
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None, 'image_name': name + ' errors',
                                   'interpolation_method': 'nearest', 'pixel_size': d.x_axes[1] - d.x_axes[0],
                                   'transition_lines': d.transition_lines, 'scan_history': self._scan_history,
                                   'final_coord': final_coord, 'show_offset': False, 'scan_errors': True,
                                   'history_uncertainty': False})
            # step with error color and uncertainty
            pool.apply_async(plot_diagram,
                             kwds={'x_i': d.x_axes, 'y_i': d.y_axes, 'pixels': None,
                                   'image_name': name + ' errors uncertainty', 'interpolation_method': 'nearest',
                                   'pixel_size': d.x_axes[1] - d.x_axes[0], 'transition_lines': d.transition_lines,
                                   'scan_history': self._scan_history, 'final_coord': final_coord, 'show_offset': False,
                                   'scan_errors': True, 'history_uncertainty': True})

            # Wait for the processes to finish
            pool.close()
            pool.join()

    def plot_step_history_animation(self, final_coord: Tuple[int, int], success_tuning: bool) -> None:
        """
        Plot the animated diagram with the tuning steps of the current procedure.

        :param final_coord: The final coordinate of the tuning procedure
        :param success_tuning: Result of the tuning (True = Success)
        """

        name = f'{self.diagram.file_basename} steps {"GOOD" if success_tuning else "FAIL"}'
        # Generate a gif image
        plot_diagram_step_animation(self.diagram, name, self._scan_history, final_coord)

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

    def _tune(self) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError

    def run_tuning(self) -> AutotuningResult:
        """
        Run the tuning procedure and collect stats for results.

        :return: Tuning information and result.
        """

        tuned_x, tuned_y = self._tune()

        return AutotuningResult(diagram_name=self.diagram.file_basename,
                                procedure_name=str(self),
                                model_name=str(self.model),
                                nb_steps=self.get_nb_steps(),
                                nb_classification_success=self.get_nb_line_detection_success(),
                                charge_area=self.diagram.get_charge(tuned_x, tuned_y),
                                final_coord=(tuned_x, tuned_y))
