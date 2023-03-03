from math import ceil
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from classes.classifier_nn import ClassifierNN
from classes.data_structures import ClassificationMetrics
from plots.train_results import plot_classification_sample, plot_confidence, plot_reliability_diagram, \
    plot_confusion_matrix
from utils.logger import logger
from utils.metrics import classification_metrics
from utils.misc import get_nb_loader_workers
from utils.output import save_results
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer


def test(network: ClassifierNN, test_dataset: Dataset, device: torch.device, test_name: str = '', final: bool = False,
         limit: int = 0, confidence_per_case: List[List[List]] = None) \
        -> ClassificationMetrics:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the testing)
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
    :param confidence_per_case: If set the confidence results will be saved in this list.
    :return: The different classification result metrics.
    """

    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True,
                             num_workers=get_nb_loader_workers(device))
    nb_batch = min(len(test_loader), ceil(nb_test_items / settings.batch_size))
    nb_classes = len(test_dataset.classes)

    metrics: Optional[ClassificationMetrics] = None
    # All prediction count
    nb_labels_predictions = np.zeros((nb_classes, nb_classes), dtype=int)
    if network.confidence_thresholds:
        # Prediction under the confidence threshold count
        nb_labels_unknown_predictions = np.zeros((nb_classes, nb_classes), dtype=int)
    nb_samples_per_case = 16
    if final:
        samples_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
        if confidence_per_case is None:
            confidence_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
    else:
        samples_per_case = None

    # Disable gradient for performances
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'), \
            ProgressBarTesting(nb_batch, final) as progress:
        # Iterate batches
        for i, (inputs, labels) in enumerate(test_loader):
            progress.incr()
            # Stop testing after the limit
            if limit and i * settings.batch_size >= limit:
                break

            # Number of inference (will have no effect if the model is not Bayesian)
            if confidence_per_case is None:
                nb_sample = settings.bayesian_nb_sample_valid
            else:
                nb_sample = settings.bayesian_nb_sample_test
            # Forward
            predicted, confidences = network.infer(inputs, nb_sample)

            # Process each item of the batch to gather stats
            for patch, label, pred, conf in zip(inputs, labels, predicted, confidences):
                # Count the number of prediction for each label
                nb_labels_predictions[label][pred] += 1

                if network.confidence_thresholds and conf < network.confidence_thresholds[pred]:
                    # Also count predictions considered as unknown
                    nb_labels_unknown_predictions[label][pred] += 1

                if confidence_per_case is not None:
                    # Save confidence per class
                    confidence_per_case[label][pred].append(conf.item())
                if final:
                    # Save samples for later plots
                    if len(samples_per_case[label][pred]) < nb_samples_per_case:
                        samples_per_case[label][pred].append((patch.cpu(), conf))

            metrics = classification_metrics(nb_labels_predictions)
            progress.update(**{'acc': metrics.accuracy, settings.main_metric: metrics.main})

    # Give more information for the final test
    if final:
        logger.info(f'Test overall {metrics} (accuracy: {metrics.accuracy:.2%})')
        logger.info(f'Test {settings.main_metric} per classes:\n\t' +
                    "\n\t".join([f'{test_dataset.classes[i]}: {m.main:05.2%}' for i, m in enumerate(metrics)]))

        save_results(final_classification_results=metrics)
        plot_confusion_matrix(nb_labels_predictions, metrics, class_names=test_dataset.classes)
        plot_classification_sample(samples_per_case, test_dataset.classes, nb_labels_predictions)
        plot_confidence(confidence_per_case, network.confidence_thresholds, 'test', per_classes=True)
        plot_reliability_diagram(confidence_per_case, 'test', 10)

        # Final test results but with confidence threshold
        if network.confidence_thresholds:
            threshold_metrics = classification_metrics(nb_labels_predictions - nb_labels_unknown_predictions)
            unknown_rate = (nb_labels_unknown_predictions.sum() / nb_labels_predictions.sum()).item()
            save_results(threshold_classification_results=threshold_metrics)
            save_results(unknown_threshold_rate=unknown_rate)

            logger.info(f'Test overall with confidence threshold {threshold_metrics}'
                        f' (accuracy: {threshold_metrics.accuracy:.2%} - unknown: {unknown_rate:.2%})')
            logger.info(f'Test {settings.main_metric} per classes with confidence threshold:\n\t' +
                        "\n\t".join([
                            f'{test_dataset.classes[i]}: {m.main:05.2%} '
                            f'(CT: {network.confidence_thresholds[i]:.2%})'
                            for i, m in enumerate(threshold_metrics)
                        ]))
            plot_confusion_matrix(nb_labels_predictions, threshold_metrics, nb_labels_unknown_predictions,
                                  class_names=test_dataset.classes, plot_name='confusion_matrix_unknown')

    return metrics


class ProgressBarTesting(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to testing. """

    def __init__(self, nb_batch: int, auto_display: bool = True):
        super().__init__(nb_batch, 1, 'Testing ', auto_display=auto_display, enable_color=settings.console_color,
                         boring_mode=not settings.visual_progress_bar,
                         refresh_time=0.5 if settings.visual_progress_bar else 10,
                         metrics=(
                             ProgressBarMetrics('acc', print_value=lambda x: f'{x:<6.2%}', evolution_indicator=False),
                             ProgressBarMetrics(settings.main_metric, print_value=lambda x: f'{x:<6.2%}',
                                                evolution_indicator=False)
                         ))
