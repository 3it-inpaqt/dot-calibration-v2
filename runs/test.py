from math import ceil
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from classes.classifier_nn import ClassifierNN
from classes.data_structures import ClassificationMetrics
from plots.train_results import plot_classification_sample, plot_confidence, plot_confusion_matrix
from utils.logger import logger
from utils.metrics import classification_metrics
from utils.misc import get_nb_loader_workers
from utils.output import save_results
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer


def test(network: ClassifierNN, test_dataset: Dataset, device: torch.device, test_name: str = '', final: bool = False,
         limit: int = 0) -> ClassificationMetrics:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the testing)
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
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
    # Prediction under the confidence threshold count
    nb_labels_unknown_predictions = np.zeros((nb_classes, nb_classes), dtype=int)
    nb_samples_per_case = 16
    if final:
        samples_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
        confidence_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
    else:
        samples_per_case = confidence_per_case = None

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
            nb_sample = settings.bayesian_nb_sample_test if final else settings.bayesian_nb_sample_valid
            # Forward
            predicted, confidences = network.infer(inputs, nb_sample)

            # Process each item of the batch to gather stats
            for patch, label, pred, conf in zip(inputs, labels, predicted, confidences):
                # Count the number of prediction for each label
                nb_labels_predictions[label][pred] += 1
                if conf < settings.confidence_threshold:  # Also count predictions considered as unknown
                    nb_labels_unknown_predictions[label][pred] += 1

                if final:
                    # Save confidence per class
                    confidence_per_case[label][pred].append(conf.item())
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
        plot_confusion_matrix(nb_labels_predictions, metrics, nb_labels_unknown_predictions,
                              class_names=test_dataset.classes, plot_name='confusion_matrix_unknown')
        plot_classification_sample(samples_per_case, test_dataset.classes, nb_labels_predictions)
        plot_confidence(confidence_per_case)

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
