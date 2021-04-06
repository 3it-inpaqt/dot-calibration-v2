from typing import Iterable, Tuple

from torch import nn


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
