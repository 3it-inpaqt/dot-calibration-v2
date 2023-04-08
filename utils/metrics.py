from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import Module
from torchinfo import summary

from classes.data_structures import CalibrationClassMetric, CalibrationMetrics, ClassMetrics, ClassificationMetrics
from utils.logger import logger
from utils.output import save_network_info
from utils.settings import settings


def confidence_bins(confidence_results: pd.DataFrame, nb_bins: int = 10, adaptative: bool = True) \
        -> (Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], int):
    """
    Split the confidence score in bins and compute the mean confidence and accuracy for each bin.

    :param confidence_results: The classification success and the confidence of each inference.
    :param nb_bins: The number of bins to consider.
    :param adaptative: If True, the bins will be adaptative (same number of elements in each bin) otherwise the bins
        will have a fixed confidence range (same confidence range in each bin).
    :return: The mean confidence per bin, the mean accuracy per bin, the bins boundaries, and the count per bin.
    """

    # Split in bins (fixed or adaptative)
    if adaptative:
        # Create N bins with the same number of elements (variable confidence range)
        bins = pd.qcut(confidence_results['confidence'], q=nb_bins, duplicates='drop')
        # It could be less than N bins if there is not enough diversity, then return None
        if len(bins) != nb_bins:
            return None, None, None, 0
    else:
        # Create N bins with fixed confidence range (variable number of elements)
        bins = pd.cut(confidence_results['confidence'], bins=nb_bins)

    grouped_by_bins = confidence_results.groupby(bins)

    # Compute the mean of the confidence in each bin
    mean_confidence_per_bin = grouped_by_bins['confidence'].mean()

    # Compute the accuracy in each bin
    mean_accuracy_per_bin = grouped_by_bins['correct'].mean()

    return mean_confidence_per_bin, mean_accuracy_per_bin, bins, confidence_results['confidence'].count()


def expected_calibration_error(confidence_results: pd.DataFrame, nb_bins: int = 10, adaptative: bool = True) -> float:
    """
    Expected Calibration Error (ECE) is a metric to quantify the calibration quality of a classifier.
    See https://doi.org/10.48550/arXiv.1902.06977 and https://doi.org/10.48550/arXiv.2107.03342 (5.B)

    :param confidence_results: The classification success and the confidence of each inference.
    :param nb_bins: The number of bins to consider.
    :param adaptative: If True, the bins will be adaptative (same number of elements in each bin) otherwise the bins
        will have a fixed confidence range (same confidence range in each bin).
    :return: The expected calibration error.
    """
    mean_confidence_per_bin, mean_accuracy_per_bin, _, count = confidence_bins(confidence_results, nb_bins, adaptative)

    # If count is 0 or the number of bins is too small, return NaN
    if count == 0:
        return float('nan')

    # Compute the calibration error as the sum of the absolute difference between the mean confidence and accuracy
    if adaptative:
        calibration_error = np.abs(mean_confidence_per_bin - mean_accuracy_per_bin).sum()
    else:
        # Weight by the number of elements in each bin if fixed size bins
        weights = count / count.sum()
        calibration_error = np.abs(weights * (mean_confidence_per_bin - mean_accuracy_per_bin)).sum()

    return calibration_error


def calibration_metrics(confidence_per_case: List[List[List[float]]]) -> CalibrationMetrics:
    """
    Compute different metrics to quantify the uncertainty calibration quality.

    :param confidence_per_case: The confidence of each prediction for each case (confusion matrix table).
    :return: The calibration metrics.
    """
    # Convert confidence array to dataframe
    flat_results = []
    for label in range(len(confidence_per_case)):
        for prediction in range(len(confidence_per_case[label])):
            for confidence in confidence_per_case[label][prediction]:
                flat_results.append((confidence, label == prediction, prediction))

    df = pd.DataFrame(flat_results, columns=['confidence', 'correct', 'prediction'])

    # Compute the metrics for each class
    metrics_per_class = []
    for i in range(len(confidence_per_case)):
        metrics_per_class.append(CalibrationClassMetric(
            ece=expected_calibration_error(df[df['prediction'] == i], settings.calibration_nb_bins, adaptative=False),
            aece=expected_calibration_error(df[df['prediction'] == i], settings.calibration_nb_bins, adaptative=True)
        ))

    return CalibrationMetrics(
        ece=expected_calibration_error(df, settings.calibration_nb_bins, adaptative=False),
        aece=expected_calibration_error(df, settings.calibration_nb_bins, adaptative=True),
        # Static Calibration Error as the average of the ECE of each predicted class
        sce=sum([m_cls.ece for m_cls in metrics_per_class]) / len(metrics_per_class),
        # Adaptative Static Calibration Error as the average of the aECE of each predicted class
        asce=sum([m_cls.aece for m_cls in metrics_per_class]) / len(metrics_per_class),
        classes=metrics_per_class
    )


def classification_metrics(confusion_matrix: np.ndarray) -> ClassificationMetrics:
    """
    Compute different metrics to quantify a classification quality.
    Everything is computer for each class and the overall score.
    Metrics: accuracy, precision, recall, f1.

    Example:
    - Input confusion matrix: [[9, 1], [4, 36]]

             |  9 |  1 |
      Labels -----------
             |  4 | 36 |
            Predictions

    - Output:
      'classes': 'f1': [0.7826086956521738, 0.935064935064935]
                 'nb': [10, 40]
                 'precisions': [0.6923076923076923, 0.972972972972973]
                 'recall': [0.9, 0.9]
      'overall': 'accuracy': 0.9
                 'f1': 0.8588368153585544
                 'nb': 50
                 'precisions': 0.8326403326403327
                 'recall': 0.9

    :param confusion_matrix: The confusion matrix that contains the number of couple (label, predictions).
    :return: The different classification result metrics (see ClassificationMetrics dataclass).
    """

    nb_labels = confusion_matrix.sum()
    nb_good_class = confusion_matrix.trace()

    classes_nb_labels = confusion_matrix.sum(1)
    classes_nb_predictions = confusion_matrix.sum(0)
    classes_nb_good_predictions = confusion_matrix.diagonal()

    # Division by 0 gives 0
    # Precision
    classes_precision = np.zeros(classes_nb_labels.shape, dtype=float)
    np.divide(classes_nb_good_predictions, classes_nb_predictions,
              out=classes_precision, where=classes_nb_predictions != 0)
    # Recall
    classes_recall = np.zeros(classes_nb_labels.shape, dtype=float)
    np.divide(classes_nb_good_predictions, classes_nb_labels,
              out=classes_recall, where=classes_nb_labels != 0)
    # F1
    classes_f1 = np.zeros(classes_nb_labels.shape, dtype=float)
    denominator = (classes_precision + classes_recall)
    np.divide((classes_precision * classes_recall), denominator, out=classes_f1, where=denominator != 0)
    classes_f1 *= 2

    return ClassificationMetrics(
        nb=int(nb_labels),
        accuracy=float(nb_good_class / nb_labels) if nb_labels > 0 else 0,
        precision=float(classes_precision.mean()),
        recall=float(classes_recall.mean()),
        f1=float(classes_f1.mean()),
        classes=[ClassMetrics(
            nb=int(classes_nb_labels[i]),
            precision=float(classes_precision[i]),
            recall=float(classes_recall[i]),
            f1=float(classes_f1[i])
        ) for i in range(len(confusion_matrix))]
    )


def network_metrics(network: Module, input_dim: List, device: Optional[torch.device],
                    save_output: bool = True) -> dict:
    """
    Extract useful information from the network.

    :param network: The network to analyse
    :param input_dim: The dimension of the input
    :param device: The device use (cpu or cuda)
    :param save_output: If true the metrics will be saved in a text file in the run directory
    :return: A dictionary of metrics with their values
    """
    input_dim = [settings.batch_size] + list(input_dim)
    network_info = summary(network, input_size=input_dim, device=device, verbose=0)

    logger.debug('Network info:\n' + str(network_info))

    metrics = {
        'name': type(network).__name__,
        'loss_function': network.get_loss_name(),
        'optimizer_function': network.get_optimizer_name(),
        'device': str(device),
        'total_params': network_info.total_params,
        'trainable_params': network_info.trainable_params,
        'non_trainable_params': network_info.total_params - network_info.trainable_params,
        'MAC_operations': network_info.total_mult_adds,
        'input_dimension': list(input_dim)
    }

    if save_output:
        save_network_info(metrics)

    return metrics
