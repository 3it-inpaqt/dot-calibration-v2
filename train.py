import os
from typing import List

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

from plots.misc import plot_train_progress
from test import test
from utils.logger import logger
from utils.output import load_network, save_network, save_results
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer


def train(network: Module, train_dataset: Dataset, test_dataset: Dataset, device: torch.device) -> None:
    """
    Train the network using the dataset.

    :param network: The network to train in place.
    :param train_dataset: The dataset used to train the network.
    :param test_dataset: The dataset used to run intermediate test on the network during the training.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the training)
    """
    # If path set, try to load a pre trained network from cache
    if settings.trained_network_cache_path and load_network(network, settings.trained_network_cache_path):
        return  # Stop here if the network parameters are successfully loaded from cache file

    # Turn on the training mode of the network
    network.train()

    # Use the pyTorch data loader
    num_workers = 0 if device.type == 'cuda' else os.cpu_count()  # cuda doesn't support multithreading for data loading
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True, num_workers=num_workers)
    nb_batch = len(train_loader)

    # Define the indexes of checkpoints for each epoch
    # Eg.: with 'nb_batch' = 100 and 'checkpoints_per_epoch' = 3, then the indexes will be [0, 33, 66]
    checkpoints_i = [int(i / settings.checkpoints_per_epoch * nb_batch) for i in range(settings.checkpoints_per_epoch)]

    # Store the loss values for plot
    loss_evolution: List[float] = []
    accuracy_evolution: List[dict] = []
    epochs_stats: List[dict] = []

    # Use timer and progress bar
    with SectionTimer('network training') as timer, ProgressBarTraining(nb_batch) as progress:
        # Iterate epoch
        for epoch in range(settings.nb_epoch):
            progress.next_subtask()
            # Iterate batches
            for i, (inputs, labels) in enumerate(train_loader):
                progress.incr()

                # Checkpoint if enable for this batch
                if i in checkpoints_i:
                    timer.pause()
                    check_results = _checkpoint(network, epoch * nb_batch + i, train_dataset, test_dataset, device)
                    progress.update(accuracy=check_results['test_accuracy'])
                    accuracy_evolution.append(check_results)
                    timer.resume()

                # Run one training step for these data
                loss = network.training_step(inputs, labels)
                progress.update(loss=loss)
                loss_evolution.append(float(loss))
                # TODO Log progress and loss based on time interval in debug

            # Epoch statistics
            _record_epoch_stats(epochs_stats, loss_evolution[-len(train_loader):])

    # TODO clear memory after the run (stuck in GPU), for train and test

    if len(accuracy_evolution) == 0:
        save_results(epochs_stats=epochs_stats)
    else:
        # Do one last checkpoint to complet the plot
        accuracy_evolution.append(
            _checkpoint(network, settings.nb_epoch * (nb_batch + 1), train_dataset, test_dataset, device))
        save_results(epochs_stats=epochs_stats, accuracy_evolution=accuracy_evolution)

    if settings.save_network:
        save_network(network, 'trained_network')

    # Post train plots
    plot_train_progress(loss_evolution, accuracy_evolution, nb_batch)


def _checkpoint(network: Module, batch_num: int, train_dataset: Dataset, test_dataset: Dataset,
                device: torch.device) -> dict:
    """
    Pause the training to do some jobs, like intermediate testing and network backup.

    :param network: The current neural network
    :param batch_num: The batch number since the beginning of the training (used as checkpoint id)
    :param train_dataset: The training dataset
    :param test_dataset: The testing dataset
    :return: A dictionary with the tests results
    """

    # Save the current network
    if settings.checkpoint_save_network:
        save_network(network, f'{batch_num:n}_checkpoint_network')
        # TODO add a setting to save the network only if the accuracy is improved

    # Start tests
    if settings.checkpoint_test_size > 0:
        test_accuracy = test(network, test_dataset, device, test_name='checkpoint test',
                             limit=settings.checkpoint_test_size)
    if settings.checkpoint_train_size > 0:
        train_accuracy = test(network, train_dataset, device, test_name='checkpoint train',
                              limit=settings.checkpoint_train_size)

    # Set it back to train because it was switched during tests
    network.train()

    logger.debug(f'Checkpoint {batch_num:<6n} '
                 f'| test accuracy: {test_accuracy:5.2%} '
                 f'| train accuracy: {train_accuracy:5.2%}')

    return {
        'batch_num': batch_num,
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy
    }


def _record_epoch_stats(epochs_stats: List[dict], epoch_losses: List[float]) -> None:
    """
    Record the statics for one epoch.

    :param epochs_stats: The list where to store the stats, append in place.
    :param epoch_losses: The losses list of the current epoch.
    """
    stats = {
        'losses_mean': float(np.mean(epoch_losses)),
        'losses_std': float(np.std(epoch_losses))
    }

    # Compute the loss difference with the previous epoch
    stats['losses_mean_diff'] = 0 if len(epochs_stats) == 0 else stats['losses_mean'] - epochs_stats[-1]['losses_mean']

    # Add stat to the list
    epochs_stats.append(stats)

    # Log stats
    epoch_num = len(epochs_stats)
    logger.debug(f"Epoch {epoch_num:3}/{settings.nb_epoch} ({epoch_num / settings.nb_epoch:7.2%}) "
                 f"| loss: {stats['losses_mean']:.5f} "
                 f"| diff: {stats['losses_mean_diff']:+.5f} "
                 f"| std: {stats['losses_std']:.5f}")


class ProgressBarTraining(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to training. """
    def __init__(self, nb_batch: int):
        super().__init__(nb_batch, settings.nb_epoch, 'Training', 'ep.', auto_display=settings.visual_progress_bar,
                         metrics=(
                             ProgressBarMetrics('loss', more_is_good=False),
                             ProgressBarMetrics('accuracy', print_value=lambda x: f'{x:<6.2%}')
                         ))
