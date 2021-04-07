import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import optim

from utils.misc import calc_out_conv_layers
from utils.settings import settings


class CNN(nn.Module):
    """
    Simple fully connected feed forward classifier neural network.
    """

    def __init__(self, input_shape: Tuple[int, int]):
        """
        Create a new network with convolutional layers, followed by fully connected hidden layers.
        The number hidden layers is based on the settings.

        :param input_shape: The dimension of one item of the dataset used for the training
        """
        super().__init__()

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=12, kernel_size=4))
        self.conv_layers.append(nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4))

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        fc_layer_sizes = [math.prod(calc_out_conv_layers(input_shape, self.conv_layers))]
        fc_layer_sizes.extend(settings.hidden_layers_size)
        fc_layer_sizes.append(1)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layer_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i + 1]))

        self._criterion = nn.BCELoss()  # Binary Cross Entropy
        # self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """

        # Run convolution layers
        for conv in self.conv_layers:
            x = torch.relu(conv(x))

        # Flatten the data (but not the batch)
        x = torch.flatten(x, 1)

        # Run fully connected layers
        for fc in self.fc_layers:
            x = torch.sigmoid(fc(x))  # Last activation function should output in [0;1] for BCELoss

        # Flatten [batch_size, 1] to [batch_size]
        return torch.flatten(x)

    def training_step(self, inputs: Any, labels: Any):
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        # Zero the parameter gradients
        self._optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels.float())
        loss.backward()
        self._optimizer.step()

        return loss

    def get_loss_name(self) -> str:
        """
        :return: The name of the loss function (criterion).
        """
        return type(self._criterion).__name__

    def get_optimizer_name(self) -> str:
        """
        :return: The name of the optimiser function.
        """
        return type(self._optimizer).__name__

    @staticmethod
    def get_transforms():
        """
        Define the data pre-processing to apply on the datasets before to use this neural network.
        """
        return [lambda x: x.view(1, x.shape[0], -1)]  # Add the channel dimension