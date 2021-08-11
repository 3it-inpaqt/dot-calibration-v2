import math
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
from torch import optim

from classes.classifier_nn import ClassifierNN
from utils.settings import settings


@variational_estimator
class BFF(ClassifierNN):
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

        # Balance or not the loss function
        if settings.balance_with_weights:
            pos_weight = torch.Tensor([class_ratio])
        else:
            pos_weight = None

        # Binary Cross Entropy including sigmoid layer
        self._criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
                                sample_nbr=settings.bayesian_nb_sample,
                                complexity_cost_weight=settings.bayesian_complexity_cost_weight)
        loss.backward()
        self._optimizer.step()

        return loss.item()

    def infer(self, inputs, nb_samples: int = 100) -> (List[bool], List[float]):
        """
        Use network inference for classification a set of input.

        :param inputs: The inputs to classify.
        :param nb_samples: The number of inference iteration to run on for each input. The inference will be done
         on the mean value.
        :return: The class inferred by the network and the confidences information.
        """
        # Prediction samples
        # Use sigmoid to convert the output into probability (during the training it's done inside BCEWithLogitsLoss)
        outputs = [torch.sigmoid(self(inputs)) for _ in range(nb_samples)]
        outputs = torch.stack(outputs)

        # Compute the mean, std and entropy
        means = outputs.mean(axis=0)
        stds = outputs.std(axis=0)
        pi = torch.Tensor([math.pi])
        entropies = torch.log(2 * pi * torch.pow(stds.cpu(), 2)) / 2 + 1 / 2
        confidences = list(zip(*[means.cpu(), stds.cpu(), entropies.cpu()]))

        # Round the samples mean value to 0 or 1
        predictions = torch.round(means).bool()
        return predictions, confidences

    @staticmethod
    def get_transforms():
        """
        Define the data pre-processing to apply on the datasets before to use this neural network.
        """
        return [lambda x: torch.flatten(x)]  # Flatten the image
