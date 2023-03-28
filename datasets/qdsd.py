from pathlib import Path
from random import shuffle
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.diagram_offline import DiagramOffline
from utils.logger import logger
from utils.output import load_data_cache, save_data_cache, save_normalization, save_results
from utils.settings import settings

DATA_DIR = Path('data')


class QDSDLines(Dataset):
    """
    Quantum Dots Stability Diagrams (QDSD) dataset.
    Transition line classification task.
    """
    classes = []
    for nb in range(settings.dot_number + 2):
        if nb == 0:
            classes.append('no line')  # First class, all parameters in the label is False
        elif settings.dot_number == 1:
            classes.append('line 1')  # Last class, all parameters in the label is True
            break
        elif nb != 2 + settings.dot_number - 1:
            classes.append(f'line {nb}')  # Class of the line {nb}, column {nb} is True
        else:
            classes.append(f'crosspoint')  # Last class for the case of dot_number is superior at 1
            # All parameters in the label is True

    def __init__(self, patches: List[Tuple], role: str, transform: Optional[List[Callable]] = None):
        """
        Create a dataset of transition lines patches.
        Should be build with QDSDLines.build_split_datasets.

        :param patches: The list of patches as (2D array of values, labels as boolean)
        :param transform: The list of transformation function to apply on the patches
        :param role: The role of this dataset ("train" or "test" or "validation")
        """

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
                             validation_ratio: float = 0,
                             patch_size: Tuple[int, int] = (10, 10),
                             overlap: Tuple[int, int] = (0, 0),
                             label_offset: Tuple[int, int] = (0, 0),
                             normalize: bool = True) -> Tuple["QDSDLines", "QDSDLines", Optional["QDSDLines"]]:
        """
        Initialize dataset of transition lines patches.
        The sizes of the test and validation dataset depend on the ratio. The train dataset get all remaining data.

        :param pixel_size: The dataset pixel size to load
        :param research_group: The dataset research group to load
        :param test_ratio_or_names: The ratio of patches reserved for test data or the list of diagram names to include
         in test data.
        :param validation_ratio: The ratio of patches reserved for validation data (ignored if 0)
        :param patch_size: The size in pixel (x, y) of the patch to process (sub-area of the stability diagram)
        :param overlap: The overlapping size, in pixel (x, y) of the patch
        :param label_offset: The width of the border to ignore during the patch labeling, in number of pixel (x, y)
        :param normalize: If True the datasets values are normalized according to the train set values
        :return A tuple of datasets: (train, test, validation) or (train, test) if validation_ratio is 0
        """

        if settings.dot_number == 2:
            single_dot = False
        elif settings.dot_number == 1:
            single_dot = True
        else:
            single_dot = False
        use_test_ratio = isinstance(test_ratio_or_names, float)
        test_patches = []
        patches = []
        cache_path = Path(DATA_DIR, 'cache',
                          f'qdsd_lines_{research_group}_{pixel_size}V_'
                          f'{"single_" if single_dot else "double_"}'
                          f'{patch_size[0]}-{patch_size[1]}_{overlap[0]}-{overlap[1]}_'
                          f'{label_offset[0]}-{label_offset[1]}.p')

        if settings.use_data_cache and use_test_ratio and cache_path.is_file():
            # Fast load from cache
            patches = load_data_cache(cache_path)
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
                if not use_test_ratio and diagram.file_basename in test_ratio_or_names:
                    test_patches.extend(diagram_patches)
                else:
                    patches.extend(diagram_patches)

            logger.info(f'{len(patches) + len(test_patches)} items loaded from {len(diagrams)} diagrams')
            if not use_test_ratio:
                logger.info(f'{len(test_ratio_or_names)} diagrams used for test set ({len(test_patches)} items): '
                            f'{", ".join(test_ratio_or_names)}')

            if settings.use_data_cache and use_test_ratio:
                # Save in cache for later runs
                save_data_cache(cache_path, patches)

        # In case of test set defined by a diagram name the valid ratio should be counted based on train size only
        nb_patches = len(patches) + len(test_patches) if use_test_ratio else len(patches)

        # Shuffle patches before to split them into different the datasets
        shuffle(patches)

        # Create data transform method if test noise is enable
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

        if normalize:
            # Normalize datasets using train as a reference for min and max
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
        ref_min = torch.min(ref_dataset._patches)
        ref_max = torch.max(ref_dataset._patches)

        # Save values to file, for consistant normalization later
        save_normalization(ref_min.item(), ref_max.item())

        for dataset in (d for d in datasets if d):  # Skip None datasets
            dataset._patches -= ref_min
            dataset._patches /= (ref_max - ref_min)


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
