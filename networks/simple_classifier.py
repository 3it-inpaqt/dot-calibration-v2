from typing import Any

import torch
import torch.nn as nn
from torch import optim

from utils.settings import settings


class SimpleClassifier(nn.Module):
    """
    Simple classifier neural network.
    Should be use as an example.
    """

    def __init__(self, input_size: int):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 200)  # Input -> Hidden 1
        self.fc2 = nn.Linear(200, 100)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(100, 1)  # Hidden 2 -> Output

        self._criterion = nn.BCELoss()  # Binary Cross Entropy
        # self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Last activation function should output in [0;1] for BCELoss
        return x.flatten()

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
        return [lambda x: torch.flatten(x)]  # Flatten the image
