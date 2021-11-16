from typing import Optional, Tuple

from classes.classifier_nn import ClassifierNN
from classes.diagram import Diagram


class AutotuningProcedure:
    """ Procedure for autotuning tuning of static stability diagrams. """

    def __init__(self, model: Optional[ClassifierNN], patch_size: Tuple[int, int], is_oracle_enable: bool = False):
        """
        Create a new procedure.

        :param model: The line detection model to use in this procedure.
        :param patch_size: The patch size to use in this procedure (should match with model expectation)
        :param is_oracle_enable: If true the line detection use directly on the labels instead of the model inference.
        """
        self.model: ClassifierNN = model
        self.patch_size = patch_size
        self.is_oracle_enable = is_oracle_enable

        if model is None and not is_oracle_enable:
            raise ValueError('If no model is provided, the oracle should be explicitly enable')
        if model is not None and is_oracle_enable:
            raise ValueError('If a model is provided, the oracle should not be enable')

    def __str__(self) -> str:
        return type(self).__name__

    def is_transition_line(self, diagram: Diagram, coordinate: Tuple[int, int]) -> (bool, float):
        """
        Try to detect a line in a sub-area of the diagram using the current model or the oracle.

        :param diagram: The diagram to consider
        :param coordinate: The coordinate of the top left of the sub-area. The size is fixed at the procedure creation.
        :return: The line classification (True = line detected)
        and the confidence score (0: low confidence to 1: very high confidence)
        """
        if self.is_oracle_enable:
            # Check the diagram label and return the classification with full confidence
            return diagram.is_line_in_patch(coordinate, self.patch_size), 1
        else:
            # Cut the patch area and send it to the model for inference
            patch = diagram.get_patch(coordinate, self.patch_size)
            return self.model.infer(patch)

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :param diagram: The diagram to use for the training
        :param start_coord: The starting coordinates (top right of the patch square)
        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError
