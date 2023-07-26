from typing import List, Tuple

import numpy as np

from plots.train_results import plot_confidence, plot_confidence_threshold_tuning
from runs.test import test
from utils.logger import logger
from utils.settings import settings


def tune_confidence_thresholds(network, dataset, device) -> List[float]:
    """
    Search the best threshold or each class inferred by the network according to the confidence distribution in a
    dataset. The thresholds are selected by minimizing a score that represents a tradeoff between error and number of
    unknown classifications.
    If the setting "confidence_threshold" is defined, return this value for every threshold.

    :param network: The neural network used for the classification and the confidence.
    :param dataset: The dataset to use for confidence distribution and accuracy testing.
    :param device: The pytorch device.
    :return: A list of threshold (1 by class).
    """
    nb_classes = len(dataset.classes)

    # Run a test with to estimate the confidences. Give a reference of the list to fill it during the test.
    confidence_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
    test(network, dataset, device, test_name='tune_thresholds', confidence_per_case=confidence_per_case)

    if settings.calibrate_confidence_method == 'fixed':
        # We simply use the confidence threshold if defined in settings for all classes
        thresholds = [settings.confidence_threshold] * nb_classes
    else:
        # Compute the best confidence thresholds for each class
        match settings.calibrate_confidence_method:
            case 'error_rate':
                thresholds, scores_history = error_rate_calibration(confidence_per_case)
            case 'dynamic':
                thresholds, scores_history = dynamic_calibration(confidence_per_case)
            case _:
                raise ValueError(f'Unknown confidence calibration method: {settings.calibrate_confidence_method}')

        logger.info(f'Best confidence thresholds on {dataset.role} with '
                    f'{settings.calibrate_confidence_method} calibration: ' +
                    ', '.join(f'{c}: {t:.1%}' for c, t in zip(dataset.classes, thresholds)))

        plot_confidence_threshold_tuning(thresholds, scores_history, len(dataset), dataset.role,
                                         settings.calibrate_confidence_method)

    plot_confidence(confidence_per_case, thresholds, dataset.role, per_classes=True)

    return thresholds


def error_rate_calibration(confidence_per_case) -> Tuple[List[float], List[List[float]]]:
    """
    Search the confidence thresholds that correspond that match the expected error of from confidence_threshold.
    Proceed independently for each class.

    :param confidence_per_case: A list of confidence distribution for each case.
    :return: A list of threshold (1 by class), and the history of error rates for each tested threshold.
    """
    nb_classes = len(confidence_per_case)

    # Recreate the confusion matrix based on the confidence list length
    nb_per_case = np.array([[len(conf) for conf in pred] for pred in confidence_per_case])
    # Add NaN padding to the confidence list to have a regular 3d array
    max_size = nb_per_case.max()
    confidence_padded = np.array([[c + [np.nan] * (max_size - len(c)) for c in pred] for pred in confidence_per_case])

    thresholds = [t / 400 for t in range(400)]
    error_rates_history = []

    best_thresholds = [None] * nb_classes  # Init with default values
    for threshold in thresholds:
        nb_threshold_up = (confidence_padded >= threshold).sum(axis=2)
        nb_per_class = nb_threshold_up.sum(axis=0)
        nb_error = nb_per_class - nb_threshold_up.diagonal()
        error_rates = nb_error / nb_per_class

        error_rates_history.append(error_rates.tolist())

        # Update the best threshold per predicted class
        for cls in range(nb_classes):
            if best_thresholds[cls] is None and error_rates[cls] <= 1 - settings.confidence_threshold:
                best_thresholds[cls] = threshold

    # In case of no threshold found, use the default value
    best_thresholds = [t if t is not None else settings.confidence_threshold for t in best_thresholds]

    return best_thresholds, error_rates_history


def dynamic_calibration(confidence_per_case):
    """
    Search the confidence thresholds that optimize the calibration score for each class.
    The score is a tradeoff between error and number of unknown classifications.
    This tradeoff is controlled by settings.dynamic_threshold_tau.

    :param confidence_per_case: A list of confidence distribution for each case.
    :return: A list of threshold (1 by class), and the history of scores for each tested threshold.
    """
    nb_classes = len(confidence_per_case)

    # Recreate the confusion matrix based on the confidence list length
    nb_per_case = np.array([[len(conf) for conf in pred] for pred in confidence_per_case])
    # Add NaN padding to the confidence list to have a regular 3d array
    max_size = nb_per_case.max()
    confidence_padded = np.array([[c + [np.nan] * (max_size - len(c)) for c in pred] for pred in confidence_per_case])

    thresholds = [t / 400 for t in range(400)]
    scores_history = []

    best_scores = [max_size] * nb_classes
    best_thresholds = [settings.confidence_threshold] * nb_classes  # Init with default values
    for threshold in thresholds:
        nb_unknown = (confidence_padded < threshold).sum(axis=2)
        nb_known = nb_per_case - nb_unknown
        nb_error = nb_known.sum(axis=0) - nb_known.diagonal()
        nb_unknown = nb_unknown.sum(axis=0)
        scores = nb_error + (nb_unknown * settings.dynamic_threshold_tau)
        scores_history.append(scores)

        # Update the best threshold per predicted class
        for i in range(nb_classes):
            if scores[i] < best_scores[i]:
                best_thresholds[i] = threshold
                best_scores[i] = scores[i]

    return best_thresholds, scores_history
