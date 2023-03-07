from typing import Optional, Tuple

import torch

from datasets.diagram import Diagram


class DiagramOnline(Diagram):

    def __init__(self, connector: "Connector"):
        self.connector = connector

    def get_patch(self, coordinate: Tuple[int, int], patch_size: Tuple[int, int]) -> torch.Tensor:
        pass

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        pass
