from typing import List, Optional

import numpy as np
import torch
from torch.nn import Module
from torchinfo import summary

from classes.data_structures import ClassMetrics, ClassificationMetrics
from utils.logger import logger
from utils.output import save_network_info
from utils.settings import settings


def calibration_metrics(confidence_per_case: List[List[List[float]]], nb_bins: int = 10, adaptative: bool = True,
                        cls: Optional[int] = None):

    # Convert to matrix
    # Sort by confidence
    # Split in bins (fixed or adaptative)
    # Compute the mean of the confidence in each bin
    pass


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

    # Division by 0 give 0
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
