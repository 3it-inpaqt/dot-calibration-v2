import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import optim

from utils.settings import settings


class FeedForward(nn.Module):
    """
    Simple fully connected feed forward classifier neural network.
    """

    def __init__(self, input_shape: Tuple[int, int]):
        """
        Create a new network with fully connected hidden layers.
        The number hidden layers is based on the settings.

        :param input_shape: The dimension of one item of the dataset used for the training
        """
        super().__init__()

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        layers_size = [math.prod(input_shape)]
        layers_size.extend(settings.hidden_layers_size)
        layers_size.append(1)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layers_size) - 1):
            self.fc_layers.append(nn.Linear(layers_size[i], layers_size[i + 1]))

        self._criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy including sigmoid layer
        # self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """

        # Run fully connected layers
        for fc in self.fc_layers[:-1]:
            x = torch.relu(fc(x))

        # Last layer doesn't use sigmoid because it's include in the loss function
        x = self.fc_layers[-1](x)

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

        return loss.item()

    def infer(self, inputs) -> bool:
        """
        Use network inference for classification a set of input.

        :param inputs: The inputs to classify.
        :return: The class inferred by the network.
        """
        outputs = self(inputs)
        # Use sigmoid to convert the output into probability (during the training it's done inside BCEWithLogitsLoss)
        return torch.round(torch.sigmoid(outputs)).bool()  # Round to 0 or 1

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
        return [lambda x: torch.flatten(x)]  # Flatten the image
