import math
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch import optim

from classes.classifier_nn import ClassifierNN
from utils.settings import settings


class FeedForward(ClassifierNN):
    """
    Simple fully connected feed forward classifier neural network.
    """

    def __init__(self, input_shape: Tuple[int, int], class_ratio: float = None):
        """
        Create a new network with fully connected hidden layers.
        The number hidden layers is based on the settings.

        :param input_shape: The dimension of one item of the dataset used for the training
        :param class_ratio: The class ratio for: no_line / line
        """
        super().__init__()

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        layer_sizes = [math.prod(input_shape)]
        layer_sizes.extend(settings.hidden_layers_size)
        layer_sizes.append(1)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = nn.Sequential()
            # Fully connected
            layer.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            # If this is not the output layer
            if i != len(layer_sizes) - 2:
                # Batch normalisation
                if settings.batch_norm_layers[i]:
                    layer.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                # Activation function
                if settings.simulate_circuit:
                    layer.append(nn.Hardtanh(min_val=0, max_val=1))
                    # layer.append(nn.ReLU())
                else:
                    # layer.append(nn.ReLU())
                    layer.append(nn.Sigmoid())
                # Dropout
                if settings.dropout > 0:
                    layer.append(nn.Dropout(settings.dropout))

            self.fc_layers.append(layer)

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
            x = fc(x)

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
        # If the parameters_clipping is set, clip every parameter of the model
        if settings.parameters_clipping is not None and settings.parameters_clipping > 0:
            with torch.no_grad():
                for param in self.parameters():
                    param.clamp_(-settings.parameters_clipping, settings.parameters_clipping)

        return loss.item()

    def infer(self, inputs, nb_sample=0) -> (List[bool], List[float]):
        """
        Use network inference for classification a set of input.

        :param inputs: The inputs to classify.
        :param nb_sample: Not used here, just added for simple compatibility with Bayesian models.
        :return: The class inferred by this method and the confidence it this result (between 0 and 1).
        """
        # Use sigmoid to convert the output into probability (during the training it's done inside BCEWithLogitsLoss)
        outputs = torch.sigmoid(self(inputs))

        # We assume that a value far from 0 or 1 mean low confidence (e.g. output:0.25 => class 0 with 50% confidence)
        confidences = torch.abs(0.5 - outputs) * 2
        predictions = torch.round(outputs).bool()  # Round to 0 or 1
        return predictions, confidences.cpu()
