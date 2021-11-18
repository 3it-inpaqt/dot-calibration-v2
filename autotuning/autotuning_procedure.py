from typing import List, Optional, Tuple

from classes.classifier_nn import ClassifierNN
from classes.diagram import Diagram


class AutotuningProcedure:
    """ Procedure for autotuning tuning of static stability diagrams. """

    def __init__(self, model: Optional[ClassifierNN],
                 patch_size: Tuple[int, int],
                 label_offsets: Tuple[int, int] = (0, 0),
                 is_oracle_enable: bool = False,
                 default_step: Optional[Tuple[int, int]] = None):
        """
        Create a new procedure.

        :param model: The line detection model to use in this procedure.
        :param patch_size: The patch size to use in this procedure (should match with model expectation)
        :param is_oracle_enable: If true the line detection use directly on the labels instead of the model inference.
        :param default_step: The default move step. If None the (patch size - offset) is use.
        """
        if model is None and not is_oracle_enable:
            raise ValueError('If no model is provided, the oracle should be explicitly enable')
        if model is not None and is_oracle_enable:
            raise ValueError('If a model is provided, the oracle should not be enable')

        self.model: ClassifierNN = model
        self.patch_size = patch_size
        self.label_offsets = label_offsets
        self.is_oracle_enable = is_oracle_enable
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

        # Performance statistics
        # History: ((x, y), (line_detected, confidence))
        self._scan_history: List[Tuple[Tuple[int, int], Tuple[bool, float]]] = []
        # Number of measured pixels
        self._area_scanned = 0

    def __str__(self) -> str:
        return type(self).__name__

    def reset_procedure(self) -> None:
        """
        Reset procedure statistics. Make it ready to start a new one.
        """
        self.x = None
        self.y = None
        self._scan_history.clear()
        self._area_scanned = 0

    def is_transition_line(self, diagram: Diagram, coordinate: Optional[Tuple[int, int]] = None) -> (bool, float):
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :param diagram: The diagram to consider.
        :param coordinate: The coordinate at the top left of the sub-area. The size and the offset are fixed at the
         procedure creation. Use the current procedure x and y if None is provided.
        :return: The line classification (True = line detected) and
         the confidence score (0: low confidence to 1: very high confidence).
        """

        if coordinate is None:
            coordinate = self.x, self.y

        result: Tuple[bool, float]
        if self.is_oracle_enable:
            # Check the diagram label and return the classification with full confidence
            result = diagram.is_line_in_patch(coordinate, self.patch_size, self.label_offsets), 1
        else:
            # Cut the patch area and send it to the model for inference
            patch = diagram.get_patch(coordinate, self.patch_size)
            result = self.model.infer(patch)

        # Record the diagram scanning activity.
        self._scan_history.append((coordinate, result))
        self._area_scanned += self.patch_size[0] * self.patch_size[1]

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
        # (0, 0) is top left
        self.x -= step_size if step_size is not None else self._default_step_x

    def move_right(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the right.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        # (0, 0) is top left
        self.x += step_size if step_size is not None else self._default_step_x

    def move_up(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the top.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        # (0, 0) is top left
        self.y -= step_size if step_size is not None else self._default_step_y

    def move_down(self, step_size: Optional[int] = None) -> None:
        """
        Shift the current coordinates to the bottom.

        :param step_size: The step size for the shifting (number of pixels). If None the procedure default
         value is used, which is the (patch size - offset) if None is specified neither at the initialisation.
        """
        # (0, 0) is top left
        self.y += step_size if step_size is not None else self._default_step_y

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :param diagram: The diagram to use for the training
        :param start_coord: The starting coordinates (top right of the patch square)
        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError
