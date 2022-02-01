import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from classes.classifier_nn import ClassifierNN
from plots.train_results import plot_classification_sample, plot_confidence, plot_confusion_matrix
from utils.logger import logger
from utils.misc import get_nb_loader_workers
from utils.output import save_results
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer


def test(network: ClassifierNN, test_dataset: Dataset, device: torch.device, test_name: str = '', final: bool = False,
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
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=True,
                             num_workers=get_nb_loader_workers(device))
    nb_batch = min(len(test_loader), nb_test_items // settings.batch_size)
    nb_classes = len(test_dataset.classes)

    nb_correct = 0
    nb_total = 0
    nb_labels_predictions = np.zeros((nb_classes, nb_classes), dtype=int)
    nb_samples_per_case = 16
    if final:
        samples_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
        confidence_per_case = [[list() for _ in range(nb_classes)] for _ in range(nb_classes)]
    else:
        samples_per_case = confidence_per_case = None

    # Disable gradient for performances
    with torch.no_grad(), SectionTimer(f'network testing{test_name}', 'info' if final else 'debug'), \
            ProgressBarTesting(nb_batch, final and settings.visual_progress_bar) as progress:
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

            # Count the result
            nb_total += len(labels)
            nb_correct += torch.eq(predicted, labels).sum()
            progress.update(accuracy=float(nb_correct / nb_total))

            # Process each item of the batch to gather stats
            for j, (label, pred) in enumerate(zip(labels, predicted)):
                # Count for accuracy per class
                nb_labels_predictions[label][pred] += 1
                if final:
                    # Save confidence per class
                    confidence_per_case[label][pred].append(confidences[j].item())
                    # Save samples for later plots
                    if len(samples_per_case[label][pred]) < nb_samples_per_case:
                        samples_per_case[label][pred].append((inputs[j].cpu(), confidences[j]))

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
        plot_classification_sample(samples_per_case, test_dataset.classes, nb_labels_predictions)
        plot_confidence(confidence_per_case)

    return accuracy


class ProgressBarTesting(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to testing. """

    def __init__(self, nb_batch: int, auto_display: bool = True):
        super().__init__(nb_batch, 1, 'Testing ', auto_display=auto_display,
                         metrics=(
                             ProgressBarMetrics('accuracy', print_value=lambda x: f'{x:<6.2%}',
                                                evolution_indicator=False),
                         ))
