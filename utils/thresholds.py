from typing import List

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

    match settings.calibrate_confidence_method:
        case 'fixed':
            # We simply use the confidence threshold if defined in settings for all classes
            thresholds = [settings.confidence_threshold] * nb_classes
        case 'error_rate':
            thresholds = error_rate_calibration(dataset, device, confidence_per_case)
        case 'dynamic':
            thresholds = dynamic_calibration(dataset, device, confidence_per_case)
        case _:
            raise ValueError(f'Unknown confidence calibration method: {settings.calibrate_confidence_method}')

    plot_confidence(confidence_per_case, thresholds, dataset.role, per_classes=True)

    return thresholds


def error_rate_calibration(dataset, device, confidence_per_case):
    pass


def dynamic_calibration(dataset, device, confidence_per_case):
    nb_classes = len(dataset.classes)

    # Recreate the confusion matrix based on the confidence list length
    nb_per_case = np.array([[len(conf) for conf in pred] for pred in confidence_per_case])
    # Add NaN padding to confidence list to have a regular 3d array
    max_size = nb_per_case.max()
    confidence_padded = np.array([[c + [np.nan] * (max_size - len(c)) for c in pred] for pred in confidence_per_case])

    thresholds = [t / 400 for t in range(400)]
    scores_history = []

    best_scores = [max_size] * nb_classes
    best_thresholds = [0.9] * nb_classes
    for threshold in thresholds:
        nb_unknown = (confidence_padded < threshold).sum(axis=2)
        nb_known = nb_per_case - nb_unknown
        nb_error = nb_known.sum(axis=0) - nb_known.diagonal()
        nb_unknown = nb_unknown.sum(axis=0)
        scores = nb_error + (nb_unknown * settings.dynamic_threshold_tau)
        scores_history.append(scores)

        # Update best threshold per predicted class
        for i in range(nb_classes):
            if scores[i] < best_scores[i]:
                best_thresholds[i] = threshold
                best_scores[i] = scores[i]

    logger.info(f'Best confidence thresholds on {dataset.role}: ' +
                ', '.join(f'{c}: {t:.1%}' for c, t in zip(dataset.classes, best_thresholds)))
    plot_confidence_threshold_tuning(thresholds, scores_history, len(dataset), dataset.role)

    return best_thresholds
