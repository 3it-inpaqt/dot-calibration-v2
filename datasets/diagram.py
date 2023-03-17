from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from classes.data_structures import ChargeRegime
from utils.misc import clip


class Diagram:
    """
    Abstract class that describe the interface of a diagram.
    """

    # The file name of this diagram (without file extension)
    file_basename: str

    def __init__(self, file_basename: str):
        self.file_basename = file_basename

    def get_random_starting_point(self) -> Tuple[int, int]:
        """
        Get a random starting point in the diagram.

        :return: The starting point coordinates as (x, y).
        """
        raise NotImplementedError

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixel)
        :return: The patch.
        """
        raise NotImplementedError

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        """
        Plot the diagram with matplotlib (save and/or show it depending on the settings).
        This method is a shortcut of plots.diagram.plot_diagram.

        :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
        :param label_extra: Optional extra information for the plot label.
        """
        raise NotImplementedError

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the dataset to a specific device (cpu or cuda) and/or a convert it to a different type.
        Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        raise NotImplementedError

    def get_charge(self, coord_x: int, coord_y: int) -> ChargeRegime:
        """
        Get the charge regime of a specific location in the diagram.

        :param coord_x: The x coordinate to check (not the voltage)
        :param coord_y: The y coordinate to check (not the voltage)
        :return: The charge regime
        """
        raise NotImplementedError

    def get_max_patch_coordinates(self) -> Tuple[int, int]:
        """
        Get the maximum coordinates of a patch in this diagram.

        :return: The maximum coordinates as (x, y)
        """
        raise NotImplementedError

    def get_values(self) -> Tuple[Optional[torch.Tensor], Sequence[float], Sequence[float]]:
        """
        Get the values of the diagram and the corresponding axis.

        :return: The values as a tensor, the list of x-axis values, the list of y-axis values
        """
        raise NotImplementedError

    def __str__(self):
        return self.file_basename

    @staticmethod
    def _coord_to_volt(coord: Iterable[float], min_v: float, max_v: float, value_step: float, snap: int = 0,
                       is_y: bool = False) -> List[float]:
        """
        Convert some coordinates to volt value for a specific stability diagram.

        :param coord: The list coordinates to convert
        :param min_v: The minimal valid value for the gate voltage in this diagram
        :param max_v: The maximal valid value for the gate voltage in this diagram
        :param value_step: The voltage difference between two coordinates (pixel size)
        :param snap: The snap margin, every points near to image border at this distance will be rounded to the image
         border (in number of pixels)
        :param is_y: If true this is the y-axis (to apply a flip)
        :return: The list of coordinates as gate voltage values
        """
        # Convert coordinates to actual voltage value
        coord = list(map(lambda t: t * value_step + min_v, coord))

        if is_y:
            # Flip Y axis (the label and the diagrams don't have the same y0 placement)
            coord = list(map(lambda t: max_v - t + min_v, coord))

        # Clip to border to avoid errors
        # TODO snap to borders
        coord = list(map(lambda t: clip(t, min_v, max_v), coord))

        return coord
