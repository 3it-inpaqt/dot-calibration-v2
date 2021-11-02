from typing import Tuple

from classes.diagram import Diagram


class AutotuningProcedure:
    """ Procedure for autotuning tuning of static stability diagrams. """

    def __init__(self, model, patch_size: Tuple[int, int]):
        """
        Create a new procedure.

        :param model: The line detection model to use in this procedure.
        :param patch_size: The patch size to use in this procedure (should match with model expectation)
        """
        self.model = model
        self.patch_size_x, self.patch_size_y = patch_size

    def __str__(self) -> str:
        return type(self).__name__

    def tune(self, diagram: Diagram, start_coord: Tuple[int, int]) -> Tuple[int, int]:
        """
        Start the tuning procedure on a diagram.

        :param diagram: The diagram to use for the training
        :param start_coord: The starting coordinates (top right of the patch square)
        :return: The coordinates (not the gate voltage) in the diagram that is 1 electron regime,
         according to this tuning procedure.
        """
        raise NotImplementedError
