import gc
import os
import random
from typing import List, Optional

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import Dataset

from circuit_simulation.analog_tester import test_analog
from circuit_simulation.circuit_simulator import CircuitSimulator
from classes.classifier import Classifier
from classes.classifier_nn import ClassifierNN
from datasets.qdsd import QDSDLines
from models.bayeasian_cnn import BCNN
from models.bayeasian_ff import BFF
from models.cnn import CNN
from models.feed_forward import FeedForward
from models.gap_baseline import GapBaseline
from models.std_baseline import StdBaseline
from plots.train_results import plot_confidence, plot_reliability_diagram, plot_confidence_threshold_tuning
from runs.test import test
from runs.train import train
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

    # Setup console logger but wait to create the directory before to set up the file output
    logger.set_console_level(settings.logger_console_level)
    logger.set_formatter(settings.console_color)

    # Create the output directory to save results and plots
    init_out_directory()

    if settings.run_name:
        logger.info(f'Run name: {settings.run_name}')

    # Set plot style
    set_plot_style()

    if settings.seed is not None:
        fix_seed()

    # Print settings
    logger.debug(settings)


def fix_seed() -> None:
    """ Set random number generator seeds for reproducibility """
    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    # Enable debug to force reproducibility (less performant)
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


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
    std_metrics = test(std, test_dataset, device)
    save_results(baseline_std_test_metrics=std_metrics)

    # Gap baseline (Max - Min)
    gap = GapBaseline()
    gap.train(train_dataset)
    gap_metrics = test(gap, test_dataset, device)
    save_results(baseline_gap_test_metrics=gap_metrics)

    logger.info(f'Baselines {settings.main_metric}:'
                f'\n\tstd: {std_metrics.main:.2%}'
                f'\n\tgap: {gap_metrics.main:.2%}')


def train_data_augmentation(train_dataset: QDSDLines, test_dataset: QDSDLines,
                            validation_dataset: Optional[QDSDLines]) -> None:
    """
    Run data augmentations methods on train if this setting is enabled.

    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :param validation_dataset: The validation dataset
    """
    if settings.train_data_augmentation:
        train_size = len(train_dataset)
        train_dataset.data_augmentation()
        train_size_aug = len(train_dataset)
        aug_rate = (train_size_aug / train_size) - 1

        info = f'Datasets size:' \
               f'\n\ttrain {train_size:n} (ratio: {train_dataset.get_class_ratio():.2f}) ' \
               f'--> augmented to {train_size_aug:n} ({aug_rate:+.0%})' \
               f'\n\ttest {len(test_dataset):n} (ratio: {test_dataset.get_class_ratio():.2f})'
        if validation_dataset:
            info += f'\n\tvalidation {len(validation_dataset):n} (ratio: {validation_dataset.get_class_ratio():.2f})'

        save_results(train_dataset_augmentation=train_size_aug - train_size)
    else:
        info = f'Datasets size:\n\ttrain {len(train_dataset):n}\n\ttest {len(test_dataset):n}'
        if validation_dataset:
            info += f'\n\tvalidation {len(validation_dataset):n}'

        save_results(train_dataset_augmentation=0)

    logger.info(info)


def tune_confidence_thresholds(network, dataset, device) -> List[float]:
    """
    Search the best threshold for each class inferred by the network according to the confidence distribution in a
    dataset. The thresholds are selected by minimizing a score that represent a tradeoff between error and number of
    unknown classification.
    If the setting "confidence_threshold" is defined, return this value for every threshold.

    :param network: The neural network used for the classification and the confidence.
    :param dataset: The dataset used for confidence distribution and accuracy testing.
    :param device: The pytorch device.
    :return: A list of threshold (1 by class).
    """
    nb_classes = len(dataset.classes)

    # We simply use the confidence threshold if defined in settings
    if settings.confidence_threshold >= 0:
        return [settings.confidence_threshold] * nb_classes

    # Give a reference of the confidence list to fill it during the test
    confidence_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
    test(network, dataset, device, test_name='tune_thresholds', confidence_per_case=confidence_per_case)

    # Recreate the confusion matrix based on the confidence list length
    nb_per_case = np.array([[len(conf) for conf in pred] for pred in confidence_per_case])
    # Add NaN padding to confidence list to have a regular 3d array
    max_size = nb_per_case.max()
    confidence_padded = np.array([[c + [np.nan] * (max_size - len(c)) for c in pred] for pred in confidence_per_case])

    # 200 steps between confidence 0.5 and 1.0
    thresholds = [t / 400 for t in range(200, 400)]
    scores_history = []

    best_scores = [max_size] * nb_classes
    best_thresholds = [0.9] * nb_classes
    for threshold in thresholds:
        nb_unknown = (confidence_padded < threshold).sum(axis=2)
        nb_known = nb_per_case - nb_unknown
        nb_error = nb_known.sum(axis=0) - nb_known.diagonal()
        nb_unknown = nb_unknown.sum(axis=0)
        scores = nb_error + (nb_unknown * settings.auto_confidence_threshold_tau)
        scores_history.append(scores)

        # Update best threshold per predicted class
        for i in range(nb_classes):
            if scores[i] < best_scores[i]:
                best_thresholds[i] = threshold
                best_scores[i] = scores[i]

    logger.info(f'Best confidence thresholds on {dataset.role}: ' +
                ', '.join(f'{c}: {t:.1%}' for c, t in zip(dataset.classes, best_thresholds)))
    plot_confidence_threshold_tuning(thresholds, scores_history, len(dataset), dataset.role)
    plot_confidence(confidence_per_case, best_thresholds, dataset.role, per_classes=True)
    plot_reliability_diagram(confidence_per_case, dataset.role, 10)

    return best_thresholds


def init_model() -> ClassifierNN:
    """
    Initialise a model based on the current settings.

    :return: The model instance.
    """
    # Build the network
    nn_type = settings.model_type.upper()
    if nn_type == 'FF':
        return FeedForward(input_shape=(settings.patch_size_x, settings.patch_size_y))
    elif nn_type == 'BFF':
        return BFF(input_shape=(settings.patch_size_x, settings.patch_size_y))
    elif nn_type == 'CNN':
        return CNN(input_shape=(settings.patch_size_x, settings.patch_size_y))
    elif nn_type == 'BCNN':
        return BCNN(input_shape=(settings.patch_size_x, settings.patch_size_y))
    else:
        raise ValueError(f'Unknown model type "{settings.model_type}".')


def get_cuda_device() -> torch.device:
    """ Select the pytorch device according to the settings (cuda or cpu) """
    # Automatically chooses if auto
    if settings.device is None or settings.device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(settings.device)


@SectionTimer('line task')
def run_train_test(train_dataset: Dataset, test_dataset: Dataset, validation_dataset: Optional[Dataset],
                   network: ClassifierNN) -> Classifier:
    """
    Run the training and the testing of the network.

    :param train_dataset: The training dataset.
    :param test_dataset: The testing dataset.
    :param validation_dataset: The validation dataset. Don't use validation if None.
    :param network: The neural network to train.
    """

    # Run data augmentations methods on train if this setting is enabled
    train_data_augmentation(train_dataset, test_dataset, validation_dataset)

    # Automatically chooses the device according to the settings
    device = get_cuda_device()
    logger.debug(f'pyTorch device selected: {device}')

    # Send the network and the datasets to the selected device (CPU or CUDA)
    # We assume the GPU have enough memory to store the whole network and datasets. If not it should be split.
    network.to(device)
    train_dataset.to(device)
    test_dataset.to(device)
    if validation_dataset:
        validation_dataset.to(device)

    # Save network stats and show if debug enable
    logger.info(f'Neural network type: {type(network).__name__}')
    network_metrics(network, test_dataset[0][0].shape, device)

    if settings.evaluate_baselines:
        # Run the baselines with the same data
        run_baselines(train_dataset, test_dataset, device)

    # Start the training
    train(network, train_dataset, validation_dataset, test_dataset, device)

    # Tune confidence thresholds
    network.confidence_thresholds = tune_confidence_thresholds(
        network, validation_dataset if validation_dataset else train_dataset, device)
    save_results(confidence_thresholds=network.confidence_thresholds)

    # Start normal test
    test(network, test_dataset, device, final=True)

    # Arrived to the end successfully (no error)
    save_results(success_run=True)

    if settings.simulate_circuit:
        if settings.test_circuit:
            # Test the simulated circuit
            test_analog(network, test_dataset)
        else:
            return CircuitSimulator(network)

    return network