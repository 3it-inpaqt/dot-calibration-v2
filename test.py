import os

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.results import plot_classification_sample, plot_confusion_matrix
from utils.logger import logger
from utils.output import save_results
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer


def test(network: Module, test_dataset: Dataset, device: torch.device, test_name: str = '', final: bool = False,
         limit: int = 0) -> float:
    """
    Start testing the network on a dataset.

    :param network: The network to use.
    :param test_dataset: The testing dataset.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the testing)
    :param test_name: Name of this test for logging and timers.
    :param final: If true this is the final test, will show in log info and save results in file.
    :param limit: Limit of item from the dataset to evaluate during this testing (0 to run process the whole dataset).
    :return: The overall accuracy.
    """

    if test_name:
        test_name = ' ' + test_name

    nb_test_items = min(len(test_dataset), limit) if limit else len(test_dataset)
    logger.debug(f'Testing{test_name} on {nb_test_items:n} inputs')

    # Turn on the inference mode of the network
    network.eval()

    # Use the pyTorch data loader
    num_workers = 0 if device.type == 'cuda' else os.cpu_count()  # cuda doesn't support multithreading for data loading
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=num_workers)
    nb_batch = min(len(test_loader), nb_test_items // settings.batch_size)
    nb_classes = len(test_dataset.classes)

    nb_correct = 0
    nb_total = 0
    nb_labels_predictions = np.zeros((nb_classes, nb_classes), dtype=int)
    nb_samples_per_case = 16
    samples_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]

    # Disable gradient for performances
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'), \
            ProgressBarTesting(nb_batch, final and settings.visual_progress_bar) as progress:
        # Iterate batches
        for i, (inputs, labels) in enumerate(test_loader):
            progress.incr()
            # Stop testing after the limit
            if limit and i * settings.batch_size >= limit:
                break

            # Forward
            outputs = network(inputs)
            predicted = torch.round(outputs).bool()  # Round to 0 or 1

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()
            progress.update(accuracy=float(nb_correct / nb_total))
            for j, (label, pred) in enumerate(zip(labels, predicted)):
                # Count for accuracy per class
                nb_labels_predictions[label][pred] += 1
                # Save samples for later plots
                if final and len(samples_per_case[label][pred]) < nb_samples_per_case:
                    samples_per_case[label][pred].append(inputs[j].cpu())

    accuracy = float(nb_correct / nb_total)

    # Give more information for the final test
    if final:
        classes_accuracy = [float(l[i] / np.sum(l)) for i, l in enumerate(nb_labels_predictions)]
        logger.info(f'Test overall accuracy: {accuracy:05.2%}')
        logger.info(f'Test accuracy per classes:\n\t' +
                    "\n\t".join(
                        [f'{test_dataset.classes[i]}: {a:05.2%}' for i, a in enumerate(classes_accuracy)]))

        save_results(final_accuracy=accuracy, final_classes_accuracy=classes_accuracy)
        plot_confusion_matrix(nb_labels_predictions, class_names=test_dataset.classes)
        plot_classification_sample(samples_per_case, test_dataset.classes)

    return accuracy


class ProgressBarTesting(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to testing. """

    def __init__(self, nb_batch: int, auto_display: bool = True):
        super().__init__(nb_batch, 1, 'Testing ', auto_display=auto_display,
                         metrics=(
                             ProgressBarMetrics('accuracy', print_value=lambda x: f'{x:<6.2%}',
                                                evolution_indicator=False),
                         ))
