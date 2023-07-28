import math
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch import optim

from classes.classifier_nn import ClassifierNN
from utils.misc import calc_out_conv_layers
from utils.settings import settings


class CNN(ClassifierNN):
    """
    Convolutional classifier neural network.
    """

    def __init__(self, input_shape: Tuple[int, int], class_ratio: float = None):
        """
        Create a new network with convolutional layers, followed by fully connected hidden layers.
        The number hidden layers is based on the settings.

        :param input_shape: The dimension of one item of the dataset used for the training
        :param class_ratio: The class ratio for: no_line / line
        """
        super().__init__()

        self.conv_layers = nn.ModuleList()
        last_nb_channel = 1

        # Create convolution layers
        for channel, kernel, max_pool, batch_norm in zip(settings.conv_layers_channel,
                                                         settings.conv_layers_kernel,
                                                         settings.max_pooling_layers,
                                                         settings.batch_norm_layers[:len(settings.conv_layers_kernel)]):
            layer = nn.Sequential()
            # Convolution
            layer.append(nn.Conv2d(in_channels=last_nb_channel, out_channels=channel, kernel_size=(kernel, kernel)))
            # Batch normalisation
            if batch_norm:
                layer.append(nn.BatchNorm2d(num_features=channel))
            # Activation function
            layer.append(nn.ReLU())
            # Max pooling
            if max_pool:
                layer.append(nn.MaxPool2d(kernel_size=(2, 2)))
            # Dropout
            if settings.dropout > 0:
                layer.append(nn.Dropout(settings.dropout))

            self.conv_layers.append(layer)
            last_nb_channel = channel

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        fc_layer_sizes = [math.prod(calc_out_conv_layers(input_shape, self.conv_layers))]
        fc_layer_sizes.extend(settings.hidden_layers_size)
        fc_layer_sizes.append(1)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_layer_sizes) - 1):
            layer = nn.Sequential()
            # If this is not the output layer
            if i != len(fc_layer_sizes) - 2:
                # Fully connected
                layer.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i + 1], settings.bias_in_hidden_layer))
                # Batch normalisation
                if settings.batch_norm_layers[len(settings.conv_layers_channel) + i]:
                    layer.append(nn.BatchNorm1d(fc_layer_sizes[i + 1]))
                # Activation function
                layer.append(nn.ReLU())
                # Dropout
                if settings.dropout > 0:
                    layer.append(nn.Dropout(settings.dropout))
            else:
                layer.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i + 1], True))

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
        # Add channel dimension
        x = x.unsqueeze(dim=1)

        # Run convolution layers
        for conv in self.conv_layers:
            x = conv(x)

        # Flatten the data (but not the batch)
        x = torch.flatten(x, 1)

        # Run fully connected layers
        for fc in self.fc_layers[:-1]:
            x = fc(x)

        # Last layer doesn't use sigmoid because it's include in the loss function
        x = self.fc_layers[-1](x)

        # Flatten [batch_size, 1] to [batch_size]
        return torch.squeeze(x)

    def training_step(self, inputs: Any, labels: Any) -> float:
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
