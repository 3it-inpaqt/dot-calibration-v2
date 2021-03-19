from typing import Any

import torch.nn as nn
import torch.nn.functional as f
from torch import optim

from utils.settings import settings


class SimpleClassifier(nn.Module):
    """
    Simple classifier neural network.
    Should be use as an example.
    """

    def __init__(self, input_size: int, nb_classes: int):
        """
        Create a new network with 2 hidden layers fully connected.

        :param input_size: The size of one item of the dataset used for the training
        :param nb_classes: Number of class to classify
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, 400)  # Input -> Hidden 1
        self.fc2 = nn.Linear(400, 200)  # Hidden 1 -> Hidden 2
        self.fc3 = nn.Linear(200, nb_classes)  # Hidden 2 -> Output

        # Convert the tensor to long before to call the CrossEntropy to match with the expected data type.
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.SGD(self.parameters(), lr=settings.learning_rate, momentum=settings.momentum)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        # Flatten the image
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
        loss = self._criterion(outputs, labels.long())
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
