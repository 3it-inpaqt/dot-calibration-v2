from pathlib import Path
from random import shuffle
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.diagram_offline import DiagramOffline
from utils.logger import logger
from utils.output import save_normalization, save_results
from utils.settings import settings

DATA_DIR = Path('data')


class QDSDLines(Dataset):
    """
    Quantum Dots Stability Diagrams (QDSD) dataset.
    Transition line classification task.
    """

    classes = [
        'no line',  # False (0)
        'line'  # True (1)
    ]

    def __init__(self, patches: List[Tuple], role: str, transform: Optional[List[Callable]] = None):
        """
        Create a dataset of transition lines patches.
        Should be build with QDSDLines.build_split_datasets.

        :param patches: The list of patches as (2D array of values, labels as boolean)
        :param transform: The list of transformation function to apply on the patches
        :param role: The role of this dataset ("train" or "test" or "validation")
        """

        if len(patches) == 0:
            raise RuntimeError(f'The list of {role} patches is empty. It could be because of a missing label, file or '
                               f'an error in a diagram name.')

        self.role = role

        # Get patches and their labels (using ninja unzip)
        self._patches, self._patches_labels = zip(*patches)

        # Convert to torch tensor
        self._patches = torch.stack(self._patches)
        self._patches_labels = torch.Tensor(self._patches_labels).bool()

        self.transform: List[Callable] = transform or []

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, index):
        patch = self._patches[index]
        # Apply transformation before to get it
        if len(self.transform) > 0:
            for transform_function in self.transform:
                patch = transform_function(patch)

        return patch, self._patches_labels[index]

    def get_data(self) -> torch.Tensor:
        """
        :return: The whole patches matrice of values (no transformation apply)
        """
        return self._patches

    def get_labels(self) -> torch.Tensor:
        """
        :return: The whole patches matrice of labels (no transformation apply)
        """
        return self._patches_labels

    def add_transform(self, transform_functions: List[Callable]) -> None:
        """
        Add a list of transformation function for data pre-processing.

        :param transform_functions: The list of transformation function.
        """
        self.transform.extend(transform_functions)

    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False,
           copy: bool = False):
        """
        Send the dataset to a specific device (cpu or cuda) and/or a convert it to a different type.
        Modification in place.
        The arguments correspond to the torch tensor "to" signature.
        See https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to.
        """
        self._patches = self._patches.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
        self._patches_labels = self._patches_labels.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)

    def get_stats(self) -> dict:
        """
        :return: Some statistics about this dataset
        """
        nb_patch = len(self)
        nb_line = torch.sum(self._patches_labels)
        return {
            f'{self.role}_dataset_size': nb_patch,
            f'{self.role}_dataset_nb_line': int(nb_line),
            f'{self.role}_dataset_nb_noline': int(nb_patch - nb_line),
        }

    def get_class_ratio(self) -> float:
        """
        :return: The class count ratio for: no_line / line
        """
        nb_patch = len(self)
        nb_line = torch.sum(self._patches_labels)

        return (nb_patch - nb_line) / nb_line

    def data_augmentation(self) -> None:
        """
        Apply data augmentation methods to the current dataset
        """

        # 180 degrees rotation to keep the orientation of the lines
        rotated_patches = self._patches.rot90(k=2, dims=(1, 2))

        # Extends dataset with rotation (label are unchanged)
        self._patches = torch.cat((self._patches, rotated_patches))
        self._patches_labels = torch.cat((self._patches_labels, self._patches_labels))

    @staticmethod
    def build_split_datasets(pixel_size,
                             research_group,
                             test_ratio_or_names: Union[float, List[str]],
                             single_dot: bool = True,
                             validation_ratio: float = 0,
                             patch_size: Tuple[int, int] = (10, 10),
                             overlap: Tuple[int, int] = (0, 0),
                             label_offset: Tuple[int, int] = (0, 0)) \
            -> Tuple["QDSDLines", "QDSDLines", Optional["QDSDLines"]]:
        """
        Initialize dataset of transition lines patches.
        The sizes of the test and validation dataset depend on the ratio. The train dataset get all remaining data.

        :param pixel_size: The dataset pixel size to load
        :param research_group: The dataset research group to load
        :param single_dot: The dataset number of quantum dot to load
        :param test_ratio_or_names: The ratio of patches reserved for test data or the list of diagram names to include
         in test data.
        :param validation_ratio: The ratio of patches reserved for validation data (ignored if 0)
        :param patch_size: The size in pixel (x, y) of the patch to process (sub-area of the stability diagram)
        :param overlap: The overlapping size, in pixel (x, y) of the patch
        :param label_offset: The width of the border to ignore during the patch labeling, in number of pixel (x, y)
        :return A tuple of datasets: (train, test, validation) or (train, test) if validation_ratio is 0
        """
        use_test_ratio = isinstance(test_ratio_or_names, float)
        test_patches = []
        patches = []
        if research_group == 'stefanie_czischek':
            patches = QDSDLines.load_stefanie_data()
        else:
            # Load fom files and labels (but lines only)
            diagrams = DiagramOffline.load_diagrams(pixel_size,
                                                    research_group,
                                                    Path(DATA_DIR, 'interpolated_csv.zip'),
                                                    Path(DATA_DIR, 'labels.json'),
                                                    single_dot,
                                                    True, False)

            for diagram in diagrams:
                diagram_patches = diagram.get_patches(patch_size, overlap, label_offset)
                if not use_test_ratio and diagram.name in test_ratio_or_names:
                    test_patches.extend(diagram_patches)
                else:
                    patches.extend(diagram_patches)

            logger.info(f'{len(patches) + len(test_patches)} items loaded from {len(diagrams)} diagrams')
            if not use_test_ratio:
                logger.info(f'{len(test_ratio_or_names)} diagrams used for test set ({len(test_patches)} items): '
                            f'{", ".join(test_ratio_or_names)}')

            if settings.use_ewma:
                patches = QDSDLines.use_ewma(patches)

        # In the case of test set defined by a diagram name, the valid ratio should be counted based on train size only
        nb_patches = len(patches) + len(test_patches) if use_test_ratio else len(patches)

        # Shuffle patches before to split them into different the datasets
        shuffle(patches)

        # Create data transform method if test noise is enabled
        if settings.test_noise > 0:
            test_transform = [AddGaussianNoise(std=settings.test_noise)]
        else:
            test_transform = None

        if use_test_ratio:
            test_index = round(nb_patches * test_ratio_or_names)
            test_set = QDSDLines(patches[:test_index], 'test', test_transform)
        else:
            test_index = 0
            test_set = QDSDLines(test_patches, 'test', test_transform)

        # With validation dataset
        if validation_ratio != 0:
            valid_index = test_index + round(nb_patches * validation_ratio)
            valid_set = QDSDLines(patches[test_index:valid_index], 'validation')
            train_set = QDSDLines(patches[valid_index:], 'train')

            datasets = (train_set, test_set, valid_set)

        # No validation dataset
        else:
            train_set = QDSDLines(patches[test_index:], 'train')

            datasets = (train_set, test_set, None)

        # Print and save stats about datasets
        stats = {}
        for dataset in (d for d in datasets if d):  # Skip None datasets
            stats.update(dataset.get_stats())

        logger.debug('Dataset:' + ''.join([f'\n\t{key}: {value}' for key, value in stats.items()]))
        save_results(**stats)

        if settings.normalization:
            # Normalize datasets in function of the settings
            QDSDLines.normalize(datasets, train_set)

        # Uncomment for data distribution plot (slow)
        # plot_data_space_distribution(datasets, 'Datasets pixel values distribution',
        #                              'normalized' if normalize else 'raw')

        return datasets

    @staticmethod
    def normalize(datasets: Iterable["QDSDLines"], ref_dataset: "QDSDLines") -> None:
        """
        Normalize some datasets according to a reference dataset (typically the train set).

        :param datasets: The datasets to normalize.
        :param ref_dataset: The reference dataset.
        """

        if settings.normalization == 'train-set':
            # Normalize every dataset in function of the min/max values from the training set
            ref_min = torch.min(ref_dataset._patches)
            ref_max = torch.max(ref_dataset._patches)

            # Save values to file, for consistant normalization later
            save_normalization(ref_min.item(), ref_max.item())

            for dataset in (d for d in datasets if d):  # Skip None datasets
                dataset._patches -= ref_min
                dataset._patches /= (ref_max - ref_min)

        if settings.normalization == 'patch':
            # Normalize the patchs in function of their own min/max values
            for dataset in (d for d in datasets if d):  # Skip None datasets
                min_per_patch = torch.amin(dataset._patches, dim=(1, 2), keepdim=True)
                max_per_patch = torch.amax(dataset._patches, dim=(1, 2), keepdim=True)
                dataset._patches = (dataset._patches - min_per_patch) / (max_per_patch - min_per_patch)

    @staticmethod
    def load_stefanie_data():
        # Stefanie's data is loaded in a 80 000 x L x L np array where L x L is the dimension of the patches in the
        # stability diagram. Patches contain 0s or 1s (the stability diagram was preprocessed).
        # The truth data is an array of lenght 80 000 that contains 0s and 1s.
        data_path = Path(DATA_DIR, 'stefanie_czischek', 'data.txt')
        labels_path = Path(DATA_DIR, 'stefanie_czischek', 'truth.txt')
        data = np.load(str(data_path))
        data = data.astype(np.float32)
        labels = np.load(str(labels_path))

        return list(zip(torch.tensor(data), torch.tensor(labels)))

    @staticmethod
    def use_ewma(patches):
        """
        We start by calculating the derivative of the pixels in a patch with respect to the voltage of one of the gates
        of the quantum dot. We then calculate the exponentially weighted moving average of the derivative and subtract
        it from the derivative. This leads to two approaches. In the first approach, we take the absolute value of this
        difference. In the second approach, we binarize this difference by assigning the value 1 to extreme values that
        are outside of k standard deviations from the mean of the difference, and 0 otherwise.
        See this paper: Moras, M. (2023). Outils d’identification du régime à un électron pour les boîtes quantiques
        semiconductrices. Master's thesis, Université de Sherbrooke.
        :param patches: stability diagrams patches and their labels.
        :return: labels and patches return after applying the preprocessing method.
        """
        patches, labels = zip(*patches)
        patches = torch.stack(patches)

        # Calculate the derivative
        delta_patches = patches[:, :, 1:] - patches[:, :, :-1]
        # Calculate the EWMA
        dimensions = delta_patches.size()
        ewma = torch.zeros(dimensions[0], dimensions[1], dimensions[2] - 2)
        ewma[:, :, 0] = delta_patches[:, :, 0:2].mean(dim=2) * (1 - settings.ewma_parameter) + \
                        delta_patches[:, :, 2] * settings.ewma_parameter
        for i in range(1, ewma.size()[2]):
            ewma[:, :, i] = ewma[:, :, i - 1] * (1 - settings.ewma_parameter) + \
                            delta_patches[:, :, i + 2] * settings.ewma_parameter
        # Calculate the difference between the derivative of the patches and the EWMA
        delta_patches_adj = delta_patches[:, :, 2:] - ewma

        # Take the abs value of delta_patches_adj or binarize delta_patches_adj based on its extreme values.
        if settings.is_ewma_with_abs:
            result_patches = delta_patches_adj
        else:
            # Calculate the pixels that are significantly different from the mean of their patch
            delta_patches_adj_mean = torch.mean(delta_patches_adj, dim=(1, 2))
            delta_patches_adj_std = torch.std(delta_patches_adj, dim=(1, 2))
            k = settings.ewma_threshold
            upper_bound = delta_patches_adj_mean + k * delta_patches_adj_std
            lower_bound = delta_patches_adj_mean - k * delta_patches_adj_std
            mask = (delta_patches_adj < lower_bound[:, None, None]) | \
                   (delta_patches_adj > upper_bound[:, None, None])
            # Convert the boolean mask to a tensor of 1.0 for True and 0.0 for False
            result_patches = mask.float()

        # Zip the patches with the labels
        result_patches = torch.split(result_patches, split_size_or_sections=1, dim=0)
        result_patches = [tensor.squeeze(dim=0) for tensor in result_patches]
        patches = list(zip(result_patches, labels))
        return patches


class AddGaussianNoise(object):
    """
    Add random gaussian noise to a tensor.
    """

    def __init__(self, std=1):
        self.std = std

    def __call__(self, tensor):
        # Add random noise to the tensor (the intensity is relative to the std and the tensor values range)
        value_range = torch.max(tensor) - torch.min(tensor)
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std * value_range

    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'
