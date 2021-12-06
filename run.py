import gc
import random

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import Dataset

from baselines.gap_baseline import GapBaseline
from baselines.std_baseline import StdBaseline
from classes.classifier_nn import ClassifierNN
from test import test
from train import train
from utils.logger import logger
from utils.metrics import network_metrics
from utils.output import init_out_directory, save_results, save_timers, set_plot_style
from utils.settings import settings
from utils.timer import SectionTimer


def preparation() -> None:
    """
    Prepare the environment before all operations.
    """

    # Settings are automatically loaded with the first import

    # Setup console logger but wait to create the directory before to setup the file output
    logger.set_console_level(settings.logger_console_level)

    # Create the output directory to save results and plots
    init_out_directory()

    if settings.run_name:
        logger.info(f'Run name: {settings.run_name}')

    # Set plot style
    set_plot_style()

    if settings.seed is not None:
        # Set random seeds for reproducibility
        random.seed(settings.seed)
        torch.manual_seed(settings.seed)
        np.random.seed(settings.seed)

    # Print settings
    logger.debug(settings)


def clean_up() -> None:
    """
    Clean up the environment after all operations. After that a new run can start again.
    """

    # Save recorded timers in a file
    save_timers()
    # Clear all timers
    Timer.timers.clear()

    # Disable the log file, so a new one can be set later
    if settings.run_name and settings.logger_file_enable:
        logger.disable_log_file()

    # Free CUDA memory
    gc.collect()
    torch.cuda.empty_cache()


@SectionTimer('baselines', 'debug')
def run_baselines(train_dataset: Dataset, test_dataset: Dataset, device: torch.device) -> None:
    """
    Run the baselines.

    :param train_dataset: The dataset to use for model training.
    :param test_dataset: The dataset to use for model testing.
    :param device: The processing device (cpu or cuda)
    """

    # Standard deviation baseline
    std = StdBaseline()
    std.train(train_dataset)
    std_accuracy = test(std, test_dataset, device)
    save_results(baseline_std_test_accuracy=std_accuracy)

    # Gap baseline (Max - Min)
    gap = GapBaseline()
    gap.train(train_dataset)
    gap_accuracy = test(gap, test_dataset, device)
    save_results(baseline_gap_test_accuracy=gap_accuracy)

    logger.info(f'Baselines accuracies:'
                f'\n\tstd: {std_accuracy:.2%}'
                f'\n\tgap: {gap_accuracy:.2%}')


def train_data_augmentation(train_dataset: Dataset, test_dataset: Dataset, validation_dataset: Dataset) -> None:
    """
    Run data augmentations methods on train if this setting is enable.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param validation_dataset: The validation dataset
    """
    if settings.train_data_augmentation:
        train_size = len(train_dataset)
        train_dataset.data_augmentation()
        train_size_aug = len(train_dataset)
        aug_rate = (train_size_aug / train_size) - 1
        logger.info(f'Datasets size:'
                    f'\n\ttrain {train_size:n} --> augmented to {train_size_aug:n} ({aug_rate:+.0%})'
                    f'\n\ttest {len(test_dataset):n}'
                    f'\n\tvalidation {len(validation_dataset):n}')
        save_results(train_dataset_augmentation=train_size_aug - train_size)
    else:
        logger.info(f'Datasets size:'
                    f'\n\ttrain {len(train_dataset):n}'
                    f'\n\ttest {len(test_dataset):n}'
                    f'\n\tvalidation {len(validation_dataset):n}')
        save_results(train_dataset_augmentation=0)


@SectionTimer('run')
def run(train_dataset: Dataset, test_dataset: Dataset, validation_dataset: Dataset, network: ClassifierNN) -> None:
    """
    Run the training and the testing of the network.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param validation_dataset: The validation dataset
    :param network: The neural network to train
    """

    # Run data augmentations methods on train if this setting is enable
    train_data_augmentation(train_dataset, test_dataset, validation_dataset)

    # Define transformation methods based on the network (for data pre-processing)
    train_dataset.add_transform(network.get_transforms())
    test_dataset.add_transform(network.get_transforms())
    validation_dataset.add_transform(network.get_transforms())

    # Automatically chooses between CPU and GPU if not specified
    if settings.device is None or settings.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(settings.device)

    logger.debug(f'pyTorch device selected: {device}')

    # Send the network and the datasets to the selected device (CPU or CUDA)
    # We assume the GPU have enough memory to store the whole network and datasets. If not it should be split.
    network.to(device)
    train_dataset.to(device)
    test_dataset.to(device)
    validation_dataset.to(device)

    # Save network stats and show if debug enable
    logger.info(f'Neural network type: {type(network).__name__}')
    network_metrics(network, test_dataset[0][0].shape, device)

    if settings.evaluate_baselines:
        # Run the baselines with the same data
        run_baselines(train_dataset, test_dataset, device)

    network.plot_parameters_sample('Sample of 9 weights from the last layer\nNot Trained network', 'pre_train')

    # Start the training
    train(network, train_dataset, validation_dataset, device)

    network.plot_parameters_sample('Sample of 9 weights from the last layer\nTrained network', 'post_train')

    # Start normal test
    test(network, test_dataset, device, final=True)

    # Arrived to the end successfully (no error)
    save_results(success_run=True)
