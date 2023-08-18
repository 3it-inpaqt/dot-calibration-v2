from abc import abstractmethod
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from classes.data_structures import ChargeRegime
from utils.misc import clip
from utils.settings import settings


class Diagram:
    """
    Abstract class that describe the interface of a diagram.
    """

    # The list of voltage for the first gate
    x_axes: Sequence[float]

    # The list of voltage for the second gate
    y_axes: Sequence[float]

    # The list of measured voltage according to the 2 gates
    values: torch.Tensor

    # The name of this diagram
    name: str

    def __init__(self, name: str):
        """
        Abstract class that describe the interface of a diagram.

        :param name: The usual name of the diagram.
        """
        self.name = name

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the data to a specific device (cpu or cuda) and/or a convert it to a different type. Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self.values = self.values.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    def get_max_patch_coordinates(self) -> Tuple[int, int]:
        """
        Get the maximum coordinates of a patch in this diagram.

        :return: The maximum coordinates as (x, y)
        """
        return len(self.x_axes) - settings.patch_size_x, len(self.y_axes) - settings.patch_size_y

    def voltage_to_coord(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert a voltage to a coordinate in the diagram relatively to the origin chosen.

        :param x: The voltage (x axes) to convert.
        :param y: The voltage (y axes) to convert.
        :return: The coordinate (x, y) in the diagram.
        """
        return round(((x - self.x_axes[0]) / settings.pixel_size)), round(((y - self.y_axes[0]) / settings.pixel_size))

    def coord_to_voltage(self, x: int, y: int, clip_in_diagram: bool = False) -> Tuple[float, float]:
        """
        Convert a coordinate in the diagram to a voltage.

        :param x: The coordinate (x axes) to convert.
        :param y: The coordinate (y axes) to convert.
        :param clip_in_diagram: If True and the coordinates are not in the diagram, the coordinate will be clipped to
            the closest value in the diagram before to convert them to volt. If True and the coordinates are not in
            the diagram, the pixel size will be used to convert them to volt.
        :return: The voltage (x, y) in this diagram.
        """
        if 0 <= x < len(self.x_axes):
            # If possible use the axes values
            x_v = self.x_axes[x]
        elif clip_in_diagram:
            # If outside the range, clip to the closest value
            x_v = self.x_axes[0] if x < 0 else self.x_axes[-1]
        else:
            # If no clipping, use the pixel size
            x_v = self.x_axes[0] + x * settings.pixel_size

        if 0 <= y < len(self.y_axes):
            # If possible use the axes values
            y_v = self.y_axes[y]
        elif clip_in_diagram:
            # If outside the range, clip to the closest value
            y_v = self.y_axes[0] if y < 0 else self.y_axes[-1]
        else:
            # If no clipping, use the pixel size
            y_v = self.y_axes[0] + y * settings.pixel_size

        return x_v, y_v

    def get_values(self) -> Tuple[Optional[torch.Tensor], Sequence[float], Sequence[float]]:
        """
        Get the values of the diagram and the corresponding axis.

        :return: The values as a tensor, the list of x-axis values, the list of y-axis values
        """
        return self.values.detach().cpu(), self.x_axes, self.y_axes

    @abstractmethod
    def get_random_starting_point(self) -> Tuple[int, int]:
        """
        Get a random starting point in the diagram.

        :return: The starting point coordinates as (x, y).
        """
        raise NotImplementedError

    @abstractmethod
    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int], normalized: bool = True) \
            -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixels)
        :param normalized: If True, the patch will be normalized between 0 and 1
        :return: The patch.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        """
        Plot the diagram with matplotlib (save and/or show it depending on the settings).
        This method is a shortcut of plots.diagram.plot_diagram.
        """
        raise NotImplementedError

    @abstractmethod
    def get_charge(self, coord_x: int, coord_y: int) -> ChargeRegime:
        """
        Get the charge regime of a specific location in the diagram.

        :param coord_x: The x coordinate to check (not the voltage)
        :param coord_y: The y coordinate to check (not the voltage)
        :return: The charge regime
        """
        raise NotImplementedError

    def __str__(self):
        return self.name

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
