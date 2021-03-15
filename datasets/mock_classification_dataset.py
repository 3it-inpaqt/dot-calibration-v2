import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset

from utils.output import save_plot


class MockClassificationDataset(Dataset):
    """
    Randomly generate a datasets of 2 dimensional points grouped in N classes.
    """

    def __init__(self, nb_classes: int = 2, data_point_per_class: int = 100):
        """
        Create a random dataset.

        :param nb_classes: Number of class
        :param data_point_per_class: Number of data point per class
        """
        self._nb_classes = nb_classes
        self.classes = ['class ' + str(i) for i in range(self._nb_classes)]
        self._data_point_per_class = data_point_per_class

        self._features = torch.zeros((nb_classes * data_point_per_class, 2), dtype=torch.float)
        self._labels = torch.zeros(nb_classes * data_point_per_class, dtype=torch.uint8)

        # Generate points for each classes with normal distribution.
        # Every class have a different center.
        for cl in range(nb_classes):
            index_start = cl * data_point_per_class
            index_end = cl * data_point_per_class + data_point_per_class

            self._features[index_start:index_end] = torch.normal(mean=cl * 2, std=1, size=(data_point_per_class, 2))
            self._labels[index_start:index_end] = cl

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        return self._features[index], self._labels[index]

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the dataset to a specific device (cpu or cuda) and/or a convert it to a different type.
        Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self._features = self._features.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        self._labels = self._labels.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    def show_plot(self) -> None:
        """
        Create a plot that represent the dataset and show it.
        """
        sns.scatterplot(x=self._features[:, 0],
                        y=self._features[:, 1],
                        hue=self._labels,
                        markers=True)
        plt.title(f'Data from {self._nb_classes} classes')
        save_plot('mock_dataset')
