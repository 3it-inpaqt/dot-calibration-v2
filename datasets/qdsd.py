from pathlib import Path
from random import shuffle
from typing import Callable, List, Tuple

import torch
from torch.utils.data import Dataset

from classes.diagram import Diagram
from utils.logger import logger
from utils.output import load_data_cache, save_data_cache, save_results
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

    def __init__(self, patches: List[Tuple], role: str):
        """
        Create a dataset of transition lines patches.
        Should be build with QDSDLines.build_split_datasets.

        :param patches: The list of patches as (2D array of values, labels as boolean)
        :param role: The role of this dataset ("train" or "test" or "validation")
        """

        self.role = role

        # Get patches and their labels (using ninja unzip)
        self._patches, self._patches_labels = zip(*patches)

        # Convert to torch tensor
        self._patches = torch.Tensor(self._patches)
        self._patches_labels = torch.Tensor(self._patches_labels).bool()

        self.transform: List[Callable] = []

        # TODO the normalisation should be done for train and test at the same time
        # Normalise data voltage
        self._patches -= torch.min(self._patches)
        self._patches /= torch.max(self._patches)

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
        nb_patche = len(self)
        nb_line = torch.sum(self._patches_labels)
        return {
            f'{self.role}_dataset_size': nb_patche,
            f'{self.role}_dataset_nb_line': int(nb_line),
            f'{self.role}_dataset_nb_noline': int(nb_patche - nb_line),
        }

    @staticmethod
    def build_split_datasets(test_ratio: float, validation_ratio: float = 0, patch_size: Tuple[int, int] = (10, 10),
                             overlap: Tuple[int, int] = (0, 0)) -> Tuple["QDSDLines", ...]:
        """
        Initialize dataset of transition lines patches.
        The sizes of the test and validation dataset depend on the ratio. The train dataset get all remaining data.

        :param test_ratio: The ratio of patches reserved for test data
        :param validation_ratio: The ratio of patches reserved for validation data (ignored if 0)
        :param patch_size: The size in pixel (x, y) of the patch to process (sub-area of the stability diagram)
        :param overlap: The overlapping size in pixel (x, y) of the patch
        :return A tuple of datasets: (train, test, validation) or (train, test) if validation_ratio is 0
        """

        cache_path = Path(DATA_DIR, 'cache', f'qdsd_lines_{patch_size[0]}-{patch_size[1]}_{overlap[0]}-{overlap[1]}.p')
        if settings.use_data_cache and cache_path.is_file():
            # Fast load from cache
            patches = load_data_cache(cache_path)
        else:
            # Load fom files
            diagrams = Diagram.load_diagrams(Path(DATA_DIR, 'interpolated_csv.zip'),
                                             Path(DATA_DIR, 'transition_lines.csv'))

            patches = []
            for diagram in diagrams:
                patches.extend(diagram.get_patches(patch_size, overlap))

            logger.info(f'{len(patches)} items loaded from {len(diagrams)} diagrams')

            if settings.use_data_cache:
                # Save in cache for later runs
                save_data_cache(cache_path, patches)

        nb_patches = len(patches)

        # Shuffle patches before to split them into different the datasets
        shuffle(patches)

        test_index = round(nb_patches * test_ratio)
        test_set = QDSDLines(patches[:test_index], 'test')

        # With validation dataset
        if validation_ratio != 0:
            valid_index = test_index + round(nb_patches * validation_ratio)
            valid_set = QDSDLines(patches[test_index:valid_index], 'validation')
            train_set = QDSDLines(patches[valid_index:], 'train')

            datasets = (train_set, test_set, valid_set)

        # No validation dataset
        else:
            train_set = QDSDLines(patches[test_index:], 'train')

            datasets = (train_set, test_set)

        # Print and save stats about datasets
        stats = {}
        for dataset in datasets:
            stats.update(dataset.get_stats())

        logger.debug('Dataset:' + ''.join([f'\n\t{key}: {value}' for key, value in stats.items()]))
        save_results(**stats)

        return datasets
