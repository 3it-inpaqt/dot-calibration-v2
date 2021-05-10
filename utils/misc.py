import os
from typing import Iterable, Tuple

import torch
from torch import nn

from utils.logger import logger
from utils.settings import settings


def get_nb_loader_workers(device: torch.device) -> int:
    """
    Estimate the number based on: the device > the user settings > hardware setup

    :param device: The torch device.
    :return: The number of data loader workers.
    """

    # Use the pyTorch data loader
    if device.type == 'cuda':
        # CUDA doesn't support multithreading for data loading
        nb_workers = 0
    elif settings.nb_loader_workers:
        # Use user setting if set (0 mean auto)
        nb_workers = settings.nb_loader_workers
    else:
        # Try to detect the number of available CPU
        # noinspection PyBroadException
        try:
            nb_workers = len(os.sched_getaffinity(0))
        except Exception:
            nb_workers = os.cpu_count()

    logger.debug(f'Data loader using {nb_workers} workers')

    return nb_workers


def clip(n, smallest, largest):
    """ Shortcut to clip a value between 2 others """
    return max(smallest, min(n, largest))


def calc_out_conv_layers(input_size: Tuple[int, int], layers: Iterable[nn.Conv2d]) -> Tuple[int, ...]:
    """
    Compute the size of output dimension of a list of convolutional layers, according to the initial input size.
    Doesn't take the batch size into account.

    :param input_size: The initial input size.
    :param layers: The list of convolutional layers.
    :return: The final output dimension.
    """
    out_h, out_w = input_size
    for layer in layers:
        out_h = (out_h + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) \
                / layer.stride[0] + 1

        out_w = (out_w + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) \
                / layer.stride[1] + 1

    return layers[-1].out_channels, int(out_h), int(out_w)
