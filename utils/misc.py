import os
from copy import copy
from dataclasses import asdict, is_dataclass
from math import ceil
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from blitz.modules import BayesianConv2d
from torch import nn

from utils.settings import settings


def get_nb_loader_workers(device: torch.device = None) -> int:
    """
    Estimate the number based on: the device > the user settings > hardware setup

    :param device: The torch device.
    :return: The number of data loader workers.
    """

    # Use the pyTorch data loader
    if device and device.type == 'cuda':
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

        nb_workers = ceil(nb_workers / 2)  # The optimal number seems to be half of the cores

    return nb_workers


def clip(n, smallest, largest):
    """ Shortcut to clip a value between 2 others """
    return max(smallest, min(n, largest))


def calc_out_conv_layers(input_size: Tuple[int, int],
                         layers: Iterable[Union[nn.Conv2d, nn.MaxPool2d, nn.Sequential, BayesianConv2d]]) \
        -> Tuple[int, ...]:
    """
    Compute the size of output dimension of a list of convolutional and max pooling layers, according to the initial
    input size.
    Doesn't take the batch size into account.

    :param input_size: The initial input size.
    :param layers: The list of convolutional layers.
    :return: The final output dimension.
    """
    # Keep only the layer that affect the size of the data
    resize_layers = []
    for la in layers:
        if type(la) in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, BayesianConv2d]:
            resize_layers.append(la)
        elif type(la) is nn.Sequential:
            for sub_la in la:
                # TODO make it recursive
                if type(sub_la) in [nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, BayesianConv2d]:
                    resize_layers.append(sub_la)

    out_h, out_w = input_size
    out_channels = 1
    for la in resize_layers:
        # Compatibility with Max Pooling and BayesianConv2d where properties are integer instead of tuple
        padding = (la.padding, la.padding) if isinstance(la.padding, int) else la.padding
        if not isinstance(la, nn.AvgPool2d):
            dilation = (la.dilation, la.dilation) if isinstance(la.dilation, int) else la.dilation
        stride = (la.stride, la.stride) if isinstance(la.stride, int) else la.stride
        kernel_size = (la.kernel_size, la.kernel_size) if isinstance(la.kernel_size, int) else la.kernel_size

        if isinstance(la, nn.AvgPool2d):
            out_h = (out_h + 2 * padding[0] - kernel_size[0]) / stride[0] + 1
            out_w = (out_w + 2 * padding[1] - kernel_size[1]) / stride[1] + 1
        else:
            out_h = (out_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
            out_w = (out_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

        if type(la) in [nn.Conv2d, BayesianConv2d]:
            # Get the channel count from the last conv layer
            out_channels = la.out_channels

    return out_channels, int(out_h), int(out_w)


def yaml_preprocess(item: Any) -> Union[str, int, float, List, Dict]:
    """
    Convert complex object to datatype accepted by yaml format.

    :param item: The item to process.
    :return: The converted item.
    """
    # FIXME: detect recursive structures

    # Convert Numpy accurate float representation to standard python float
    if isinstance(item, np.float_):
        return float(item)

    # If a primitive type know by yalm, then everything is good,
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float) or isinstance(item, bool):
        return item

    # If dataclass use dictionary conversion
    if is_dataclass(item) and not isinstance(item, type):
        item = asdict(item)

    # If it's a dictionary, process the values
    if isinstance(item, dict):
        item = copy(item)
        for name, value in item.items():
            item[name] = yaml_preprocess(value)  # TODO Process name too?

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


def short_number(n: Union[int, float]) -> str:
    """
    Convert an integer into a short string notation using 'k' for 1 000 and 'M' for 1 000 000.

    Args:
        n: The integer to format.

    Returns:
        The formatted string.
    """
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f'{n / 1_000:.1f}'.rstrip('0').rstrip('.') + 'k'  # Remove unnecessary decimal 0
    # >= 1_000_000
    return f'{n / 1_000_000:.1f}'.rstrip('0').rstrip('.') + 'M'  # Remove unnecessary decimal 0