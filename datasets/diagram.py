from typing import Iterable, Optional, Tuple

import torch

from utils.output import load_normalization
from utils.settings import settings


class Diagram:
    # The file name of this diagram (without file extension)
    file_basename: str

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        """
        Extract one patch in the diagram (data only, no label).

        :param coordinate: The coordinate in the diagram (not the voltage)
        :param patch_size: The size of the patch to extract (in number of pixel)
        :return: The patch
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

    def __str__(self):
        return self.file_basename

    @staticmethod
    def normalize(diagrams: Iterable["DiagramOffline"]) -> None:
        """
        Normalize the diagram with the same min/max value used during the training.
        The values are fetch via the normalization_values_path setting.
        :param diagrams: The diagrams to normalize.
        """
        if settings.autotuning_use_oracle:
            return  # No need to normalize if we use the oracle

        min_value, max_value = load_normalization()

        for diagram in diagrams:
            diagram.values -= min_value
            diagram.values /= max_value - min_value
