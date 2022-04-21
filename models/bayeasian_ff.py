import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch import optim

from classes.classifier_bayes_nn import ClassifierBayesNN
from utils.settings import settings


@variational_estimator
class BFF(ClassifierBayesNN):
    """
    Bayesian fully connected feed forward classifier neural network.
    """

    def __init__(self, input_shape: Tuple[int, int], class_ratio: float = None):
        """
        Create a bayesian new network with fully connected hidden layers.
        The number hidden layers is based on the settings.

        :param input_shape: The dimension of one item of the dataset used for the training
        :param class_ratio: The class ratio for: no_line / line
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
            self.fc_layers.append(BayesianLinear(layers_size[i], layers_size[i + 1]))

        # Binary Cross Entropy including sigmoid layer
        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self.parameters(), lr=settings.learning_rate)

    def forward(self, x: Any) -> Any:
        """
        Define the forward logic.

        :param x: One input of the dataset
        :return: The output of the network
        """
        # Flatten input but not the batch
        x = x.flatten(start_dim=1)

        # Run fully connected layers
        for fc in self.fc_layers[:-1]:
            x = torch.relu(fc(x))

        # Last layer doesn't use sigmoid because it's include in the loss function
        # FIXME random error here
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
        loss = self.sample_elbo(inputs=inputs,
                                labels=labels.float(),
                                criterion=self._criterion,
                                sample_nbr=settings.bayesian_nb_sample_train,
                                complexity_cost_weight=settings.bayesian_complexity_cost_weight)
        loss.backward()
        self._optimizer.step()

        return loss.item()
