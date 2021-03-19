from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, Polygon

from plots.diagram import plot_diagram


@dataclass
class Diagram:
    """ Handle the diagram data and its annotations. """

    # The file name of this diagram (without file extension)
    file_basename: str

    # The list of voltage for the first gate
    x: Sequence[float]

    # The list of voltage for the second gate
    y: Sequence[float]

    # The list of measured voltage according to the 2 gates
    values: Sequence[float]

    # The transition lines annotations
    transition_lines: List[LineString]

    def get_patches(self, patch_size: Tuple[int, int] = (10, 10), overlap: Tuple[int, int] = (0, 0)) -> Generator:
        """
        Create patches from diagrams sub-area.

        :param patch_size: The size of the desired patches in number of pixels (x, y)
        :param overlap: The size of the patches overlapping in number of pixels (x, y)
        :return: A generator of patches.
        """
        patch_size_x, patch_size_y = patch_size
        overlap_size_x, overlap_size_y = overlap
        diagram_size_y, diagram_size_x = self.values.shape

        # Extract each patches
        i = 0
        for patch_y in range(0, diagram_size_y - patch_size_y, patch_size_y - overlap_size_y):
            # Patch coordinates (indexes)
            start_y = patch_y
            end_y = patch_y + patch_size_y
            # Patch coordinates (voltage)
            start_y_v = self.y[start_y]
            end_y_v = self.y[end_y]
            for patch_x in range(0, diagram_size_x - patch_size_x, patch_size_x - overlap_size_x):
                i += 1
                # Patch coordinates (indexes)
                start_x = patch_x
                end_x = patch_x + patch_size_x
                # Patch coordinates (voltage)
                start_x_v = self.x[start_x]
                end_x_v = self.x[end_x]

                # Create patch shape to find line intersection
                patch_shape = Polygon([(start_x_v, start_y_v),
                                       (end_x_v, start_y_v),
                                       (end_x_v, end_y_v),
                                       (start_x_v, end_y_v)])

                # Extract patch value
                patch = self.values[start_y:end_y, start_x:end_x]
                # Label is True if any line intersect the patch shape
                label = any([line.intersects(patch_shape) for line in self.transition_lines])

                # self.plot((start_x_v, end_x_v, start_y_v, end_y_v), f' - patch {i:n} - line {label}')
                yield patch, label

    def plot(self, focus_area: Optional[Tuple] = None, label_extra: Optional[str] = '') -> None:
        """
        Plot the diagram with matplotlib (save and/or show it depending on the settings).
        This method is a shortcut of plots.diagram.plot_diagram.

        :param focus_area: Optional coordinates to restrict the plotting area. A Tuple as (x_min, x_max, y_min, y_max).
        :param label_extra: Optional extra information for the plot label.
        """
        plot_diagram(self.x, self.y, self.values, self.file_basename + label_extra, 'nearest', self.x[1] - self.x[0],
                     transition_lines=self.transition_lines, focus_area=focus_area)
