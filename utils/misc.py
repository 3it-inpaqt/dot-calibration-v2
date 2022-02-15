import os
from copy import copy
from dataclasses import is_dataclass
from typing import Any, Dict, Iterable, List, Tuple, Union

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
        # Compatibility with BayesianConv2d where padding, dilation and stride are integer instead of tuple
        padding = (layer.padding, layer.padding) if isinstance(layer.padding, int) else layer.padding
        dilation = (layer.dilation, layer.dilation) if isinstance(layer.dilation, int) else layer.dilation
        stride = (layer.stride, layer.stride) if isinstance(layer.stride, int) else layer.stride

        out_h = (out_h + 2 * padding[0] - dilation[0] * (layer.kernel_size[0] - 1) - 1) / stride[0] + 1

        out_w = (out_w + 2 * padding[1] - dilation[1] * (layer.kernel_size[1] - 1) - 1) / stride[1] + 1

    return layers[-1].out_channels, int(out_h), int(out_w)


def yaml_preprocess(item: Any) -> Union[str, int, float, List, Dict]:
    """
    Convert complex object to datatype accepted by yaml format.

    :param item: The item to process.
    :return: The converted item.
    """
    # FIXME: detect recursive structures

    # If a primitive type know by yalm, then everything is good,
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float) or isinstance(item, bool):
        return item

    # If dataclass use dictionary conversion
    if is_dataclass(item):
        item = item.__dict__

    # If it's a dictionary, process the values
    if isinstance(item, dict):
        item = copy(item)
        for name, value in item.items():
            item[name] = yaml_preprocess(value)

        return item

    try:
        # Try to convert to a list, if not possible throws error and convert it to string
        item = list(item)
        item = copy(item)

        for i in range(len(item)):
            item[i] = yaml_preprocess(item[i])

        return item

    except TypeError:
        # Not iterable, then convert to string
        return str(item)
