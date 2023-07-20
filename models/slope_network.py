import math
from pathlib import Path
from typing import Union, Tuple, Sequence, Optional

import torch
from torch.nn import Module, Sequential, ReLU, Linear, LeakyReLU

from classes.classifier_nn import ClassifierNN
from utils.logger import logger
from utils.settings import settings


def init_slope_model() -> ClassifierNN:
    """
    Initialise a model based on the current settings.

    :return: The model instance.
    """
    # Build the network
    nn_type = settings.slope_model_type.upper()
    network_option = [
        settings.slope_hidden_layers_size,
        settings.slope_conv_layers_channel,
        settings.slope_conv_layers_kernel,
        settings.slope_max_pooling_layers,
        settings.slope_batch_norm_layers,
        settings.slope_dropout
    ]
    if nn_type == 'WILLIAM':
        return Slope(input_shape=(settings.patch_size_x, settings.patch_size_y), network_option=network_option)
    else:
        raise ValueError(f'Unknown model type "{settings.slope_model_type}".')


def load_slope_network(network: Module, file_path: Union[str, Path], device: torch.device) -> bool:
    """
    Load a full description of the network parameters and states from a previous save file.

    :param network: The network to load into (in place)
    :param file_path: The path to the file to load
    :param device: The pytorch device where to load the network
    :return: True if the file exist and is loaded, False if the file is not found.
    """

    cache_path = Path(file_path) if isinstance(file_path, str) else file_path
    if cache_path.is_file():
        network.load_state_dict(torch.load(cache_path, map_location=device))
        logger.info(f'Slope network parameters loaded from file ({cache_path})')
        return True
    return False


def Slope(input_shape: Tuple[int, int], network_option: Optional[Sequence] = ()):
    """
    Create a new network with fully connected hidden layers.
    The number hidden layers is based on the settings.
    :param input_shape: The dimension of one item of the dataset used for the training
    :param network_option: Optional, for slope estimation network, load parameter for this network
    """
    layer_sizes = [math.prod(input_shape)]
    layer_sizes.extend(network_option[0])
    layer_sizes.append(1)
    model = Sequential()
    for i in range(len(layer_sizes) - 1):
        model.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 3:
            model.append(LeakyReLU())
        elif i == len(layer_sizes) - 3:
            model.append(ReLU())
    return model


def norm_value(data: torch.Tensor, min: float, max: float) -> torch.Tensor:
    data_norm = data.clone()
    data_norm -= min
    data_norm /= max
    return data_norm
