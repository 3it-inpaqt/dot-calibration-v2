from math import ceil
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler

from classes.classifier_nn import ClassifierNN
from plots.train_results import plot_train_progress
from runs.test import test
from utils.logger import logger
from utils.misc import get_nb_loader_workers
from utils.output import load_network_, load_previous_network_version_, save_network, save_results, push_notification
from utils.progress_bar import ProgressBar, ProgressBarMetrics
from utils.settings import settings
from utils.timer import SectionTimer, duration_to_str


def train(network: ClassifierNN, train_dataset: Dataset, validation_dataset: Optional[Dataset],
          test_dataset: Optional[Dataset], device: torch.device) \
        -> None:
    """
    Train the network using the dataset.

    :param network: The network to train in place.
    :param train_dataset: The dataset used to train the network.
    :param validation_dataset: The validation dataset used to run intermediate test on the network during the training.
    :param test_dataset: The test dataset used to run intermediate test on the network during the training.
    :param device: The device used to store the network and datasets (it can influence the behaviour of the training)
    """
    # If path set, try to load a pre-trained network from cache
    if settings.trained_network_cache_path and load_network_(network, settings.trained_network_cache_path, device):
        return  # Stop here if the network parameters are successfully loaded from cache file

    # Turn on the training mode of the network
    network.train()

    sampler = None
    if settings.balance_class_sampling:
        sampler = ImbalancedDatasetSampler(train_dataset,
                                           # Convert boolean to int
                                           callback_get_label=lambda dataset: list(map(int, dataset.get_labels())))

    # Use the pyTorch data loader
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size,
                              sampler=sampler,
                              shuffle=not settings.balance_class_sampling,
                              num_workers=get_nb_loader_workers(device))
    nb_batch = len(train_loader)
    nb_epoch = settings.nb_epoch if settings.nb_epoch > 0 else ceil(settings.nb_train_update / nb_batch)

    # Define the indexes of checkpoints
    checkpoints_epoch = set()
    if settings.checkpoints_per_epoch > 0:
        # Eg.: with 'nb_batch' = 100 and 'checkpoints_per_epoch' = 3, then the indexes will be [0, 33, 66]
        checkpoints_epoch = {int(i / settings.checkpoints_per_epoch * nb_batch)
                             for i in range(settings.checkpoints_per_epoch)}

    checkpoints_global = set()
    if settings.checkpoints_after_updates > 0:
        # Eg.: with 'nb_batch' = 100 and 'checkpoints_per_epoch' = 3, then the indexes will be [0, 33, 66]
        checkpoints_global = {i for i in range(0, nb_epoch * nb_batch, settings.checkpoints_after_updates)}

    logger.info(f'Train on {nb_batch:n} batches for {nb_epoch:n} epochs ({nb_epoch * nb_batch} parameters updates '
                f'and {len(checkpoints_global) + (len(checkpoints_epoch) * nb_epoch)} checkpoints)')

    # Metrics to monitoring the training and draw plots
    loss_evolution: List[float] = []
    metrics_evolution: List[dict] = []
    epochs_stats: List[dict] = []
    best_checkpoint: dict = {'score': 0, 'batch_num': None}

    # Use timer and progress bar
    with SectionTimer('network training') as timer, ProgressBarTraining(nb_batch, nb_epoch) as progress:
        # Iterate epoch
        for epoch in range(nb_epoch):
            progress.next_subtask()
            # Iterate batches
            for i, (inputs, labels) in enumerate(train_loader):
                progress.incr()

                # Checkpoint if enable for this batch
                if i in checkpoints_epoch or (epoch * nb_batch + i) in checkpoints_global:
                    timer.pause()
                    check_metrics = _checkpoint(network, epoch * nb_batch + i, train_dataset, validation_dataset,
                                                test_dataset, best_checkpoint, device)
                    progress.update(
                        **{'acc': check_metrics['validation' if validation_dataset else 'train'].accuracy,
                           settings.main_metric: check_metrics['validation' if validation_dataset else 'train'].main})
                    metrics_evolution.append(check_metrics)
                    timer.resume()

                # Run one training step for these data
                loss = network.training_step(inputs, labels)
                progress.update(loss=loss)
                loss_evolution.append(float(loss))

            # Epoch statistics
            _record_epoch_stats(epochs_stats, loss_evolution[-len(train_loader):], nb_epoch)

    save_results(epochs_stats=epochs_stats)
    if len(metrics_evolution) > 0:
        # Do one last checkpoint to complet the plot
        metrics_evolution.append(
            _checkpoint(network, nb_epoch * nb_batch, train_dataset, validation_dataset, test_dataset, best_checkpoint,
                        device))

    if settings.save_network:
        save_network(network, 'final_network')

    if best_checkpoint['batch_num'] is not None:
        save_results(best_score=best_checkpoint['score'],
                     best_validation_batch=best_checkpoint['batch_num'])

        # Apply early stopping by loading best version
        if settings.early_stopping:
            _apply_early_stopping(network, best_checkpoint, nb_batch, nb_epoch, device)

    # Post train plots
    plot_train_progress(loss_evolution, metrics_evolution, nb_batch, best_checkpoint)
    push_notification('Training finished',
                      f'Training of "{settings.run_name}" finished in {duration_to_str(timer.last)}')


def _checkpoint(network: ClassifierNN, batch_num: int, train_dataset: Dataset, validation_dataset: Optional[Dataset],
                test_dataset: Optional[Dataset], best_checkpoint: dict, device: torch.device) -> dict:
    """
    Pause the training to do some jobs, like intermediate testing and network backup.

    :param network: The current neural network
    :param batch_num: The batch number since the beginning of the training (used as checkpoint id)
    :param train_dataset: The training dataset
    :param validation_dataset: The validation dataset
    :param validation_dataset: The testing dataset
    :return: A dictionary with the tests results
    """

    # Save the current network
    if settings.checkpoint_save_network:
        save_network(network, f'{batch_num:n}_checkpoint_network')

    validation_metrics = train_metrics = test_metrics = None
    # Test on the validation dataset
    if validation_dataset and settings.checkpoint_validation:
        validation_metrics = test(network, validation_dataset, device, test_name='checkpoint validation')
        # Check if this is the new best score
        if validation_metrics.main > best_checkpoint['score']:
            logger.debug(f'New best validation {validation_metrics}')
            best_checkpoint['score'] = validation_metrics.main
            best_checkpoint['batch_num'] = batch_num
            # Save new best parameters
            if settings.early_stopping:
                # TODO find a workaround to save it if this is an unnamed run.
                save_network(network, 'best_network')

    # Test on testing dataset
    if test_dataset and settings.checkpoint_test:
        test_metrics = test(network, test_dataset, device, test_name='checkpoint testing')

    # Test on a subset of train dataset
    if settings.checkpoint_train_size > 0:
        train_metrics = test(network, train_dataset, device, test_name='checkpoint train',
                             limit=settings.checkpoint_train_size)

    # Set it back to train because it was switched during tests
    network.train()

    logger.debug(f'Checkpoint {batch_num:<6n} | validation {validation_metrics} | train {train_metrics}')

    return {'batch_num': batch_num, 'validation': validation_metrics, 'train': train_metrics, 'test': test_metrics}


def _record_epoch_stats(epochs_stats: List[dict], epoch_losses: List[float], nb_epoch: int) -> None:
    """
    Record the statics for one epoch.

    :param epochs_stats: The list where to store the stats, append in place.
    :param epoch_losses: The losses list of the current epoch.
    :param nb_epoch: The number of epoch.
    """
    stats = {
        'losses_mean': float(np.mean(epoch_losses)),
        'losses_std': float(np.std(epoch_losses))
    }

    # Compute the loss difference with the previous epoch
    diff = 0 if len(epochs_stats) == 0 else stats['losses_mean'] - epochs_stats[-1]['losses_mean']

    # Add stat to the list
    epochs_stats.append(stats)

    # Log stats
    epoch_num = len(epochs_stats)
    logger.debug(f"Epoch {epoch_num:3}/{nb_epoch} ({epoch_num / nb_epoch:7.2%}) "
                 f"| loss: {stats['losses_mean']:.5f} | diff: {diff:+.5f} | std: {stats['losses_std']:.5f}")


def _apply_early_stopping(network: ClassifierNN, best_checkpoint: dict, nb_batch: int, nb_epoch: int,
                          device: torch.device, ) -> None:
    """
    Apply early stopping by loading the best network according to the validation classification main metric ran during
    checkpoints.

    :param network: The network to load (in place).
    :param best_checkpoint: A dictionary containing at least 'batch_num' of the best one.
    :param nb_batch: The number of batch per epoch during this training.
    :param nb_epoch: The number of epoch.
    :param device: The device used to store the network and datasets.
    """
    best_batch_num = best_checkpoint['batch_num']

    # Skip if the last version of the network is already the best.
    if best_batch_num == nb_epoch * nb_batch:
        logger.info('Early stopping not necessary here because the last epoch is th best. '
                    f'The number of epoch ({nb_epoch}) could be increased.')
        return

    best_epoch_num = best_batch_num // nb_batch + 1
    last_batch_num = nb_epoch * nb_batch
    logger.info(f'Applying early stopping by loading the best version of the network: '
                f'batch {best_batch_num:n}/{last_batch_num:n} '
                f'({best_batch_num / last_batch_num:.0%}) - '
                f'epoch {best_epoch_num}/{nb_epoch}')

    # Load network in place
    if not load_previous_network_version_(network, 'best_network', device):
        logger.error('Impossible to load previous version of the network to apply early stopping.')


class ProgressBarTraining(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to training. """

    def __init__(self, nb_batch: int, nb_epoch: int):
        super().__init__(nb_batch, nb_epoch, 'Training', 'ep.',
                         enable_color=settings.console_color,
                         boring_mode=not settings.visual_progress_bar,
                         refresh_time=0.5 if settings.visual_progress_bar else 10,
                         metrics=(
                             ProgressBarMetrics('loss', more_is_good=False),
                             ProgressBarMetrics('acc', print_value=lambda x: f'{x:<6.2%}'),
                             ProgressBarMetrics(settings.main_metric, print_value=lambda x: f'{x:<6.2%}')
                         ))
