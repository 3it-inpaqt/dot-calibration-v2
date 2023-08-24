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

        # If hardware_aware_training is True, param_temp_cache is used to store replaced params temporarily
        self.param_temp_cache = []
        self.mask_temp_cache = []

        # Number of neurons per layer
        # eg: input_size, hidden size 1, hidden size 2, ..., nb_classes
        layer_sizes = [math.prod(input_shape)]
        layer_sizes.extend(settings.hidden_layers_size)
        layer_sizes.append(1)

        # Create fully connected linear layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = nn.Sequential()
            # If this is not the output layer
            if i != len(layer_sizes) - 2:
                # Fully connected
                layer.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], settings.bias_in_hidden_layer))
                # Batch normalisation
                if settings.batch_norm_layers[i]:
                    layer.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                # Activation function
                layer.append(nn.ReLU())
                # Dropout
                if settings.dropout > 0:
                    layer.append(nn.Dropout(settings.dropout))
            else:
                # Fully connected
                layer.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], True))

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

        # force weights to 0, parameters_clipping or - parameters_clipping to simulate memristors that are blocked
        if settings.hardware_aware_training:
            self.force_weights()

        # Select and set a proportion of weights to 0 if dropconnect should be used.
        if settings.use_dropconnect:
            self.dropconnect()

        # Forward + Backward + Optimize
        outputs = self(inputs)
        loss = self._criterion(outputs, labels.float())
        loss.backward()
        self._optimizer.step()

        # If weights were forced to some values, reset the old values before the next training step
        if settings.hardware_aware_training or settings.use_dropconnect:
            with torch.no_grad():
                for i, param in enumerate(self.parameters()):
                    param.data = param.data * ~self.mask_temp_cache[i] + self.param_temp_cache[i]

            self.param_temp_cache = []
            self.mask_temp_cache = []

        # If the parameters_clipping is set, clip every parameter of the model (used when simulating circuits)
        if settings.parameters_clipping is not None and settings.simulate_circuit:
            with torch.no_grad():
                for param in self.parameters():
                    param.clamp_(-settings.parameters_clipping, settings.parameters_clipping)

        return loss.item()

    def force_weights(self):
        """
        Forces the weights to some values to simulate memristors that are blocked
        :return: None
        """
        with torch.no_grad():
            if settings.parameters_clipping is not None:
                parameters_abs_max = settings.parameters_clipping
            else:
                parameters_abs_max = max(layer.data.abs().max() for layer in self.parameters()).item()
            for param in self.parameters():
                # if a memristor has probability p of being blocked and the value of a weight is encoded in two
                # memristors, then the probability that the value of a weight doesn't get modified by a blocked
                # memristor is (1-p)(1-p).
                memristor_blocked_prob = settings.ratio_failure_LRS + settings.ratio_failure_HRS
                weight_affected_prob = 1 - (1 - memristor_blocked_prob) ** 2
                weight_affected_mask = torch.rand_like(param) < weight_affected_prob

                # Generate masks for weights to be modified by two stucked memristors (both_memristors_affected_mask)
                # or a single stuck memristor encoding the positive or negative values (pos_memristors_affected_mask,
                # neg_memristors_affected_mask)
                mask_generator = torch.rand_like(param)
                both_memristors_affected_mask = mask_generator < (memristor_blocked_prob ** 2) / weight_affected_prob
                pos_memristors_affected_mask = mask_generator > 1 - memristor_blocked_prob * \
                                               (1 - memristor_blocked_prob) / weight_affected_prob
                neg_memristors_affected_mask = torch.bitwise_not(torch.bitwise_or(both_memristors_affected_mask,
                                                                                  pos_memristors_affected_mask))

                both_memristors_affected_mask = torch.bitwise_and(both_memristors_affected_mask, weight_affected_mask)
                pos_memristors_affected_mask = torch.bitwise_and(pos_memristors_affected_mask, weight_affected_mask)
                neg_memristors_affected_mask = torch.bitwise_and(neg_memristors_affected_mask, weight_affected_mask)

                # When both memristors encoding a weight are affected, generate the masks to select the weights where
                # the memristors are LRS-LRS, LRS-HRS, HRS-LRS, HRS-HRS
                lrs_prob = settings.ratio_failure_LRS / memristor_blocked_prob
                hrs_prob = 1 - lrs_prob
                lrs_lrs_prob = lrs_prob ** 2
                lrs_hrs_prob = lrs_prob * hrs_prob
                hrs_lrs_prob = hrs_prob * lrs_prob
                hrs_hrs_prob = hrs_prob ** 2
                mask_generator = torch.rand_like(param)
                lrs_lrs_mask = mask_generator < lrs_lrs_prob
                lrs_hrs_mask = torch.bitwise_and(mask_generator > lrs_lrs_prob,
                                                 mask_generator < lrs_lrs_prob + lrs_hrs_prob)
                hrs_lrs_mask = torch.bitwise_and(mask_generator > lrs_lrs_prob + lrs_hrs_prob,
                                                 mask_generator < lrs_lrs_prob + lrs_hrs_prob + hrs_lrs_prob)
                hrs_hrs_mask = mask_generator > 1 - hrs_hrs_prob
                both_memristors_lrs_lrs_mask = torch.bitwise_and(lrs_lrs_mask, both_memristors_affected_mask)
                both_memristors_lrs_hrs_mask = torch.bitwise_and(lrs_hrs_mask, both_memristors_affected_mask)
                both_memristors_hrs_lrs_mask = torch.bitwise_and(hrs_lrs_mask, both_memristors_affected_mask)
                both_memristors_hrs_hrs_mask = torch.bitwise_and(hrs_hrs_mask, both_memristors_affected_mask)

                # When only one memristor encoding a weight is affected, generate the masks to select the weights where
                # the memristor encodes positive values and is LRS, same for HRS, the memristor encodes negative
                # values and is LRS, same for HRS.
                mask_generator = torch.rand_like(param)
                lrs_mask = mask_generator < lrs_prob
                hrs_mask = torch.bitwise_not(lrs_mask)
                pos_memristors_lrs_mask = torch.bitwise_and(pos_memristors_affected_mask, lrs_mask)
                pos_memristors_hrs_mask = torch.bitwise_and(pos_memristors_affected_mask, hrs_mask)
                neg_memristors_lrs_mask = torch.bitwise_and(neg_memristors_affected_mask, lrs_mask)
                neg_memristors_hrs_mask = torch.bitwise_and(neg_memristors_affected_mask, hrs_mask)

                # Keep in memory the mask used to select the weights to be modified and the values of these weights
                # before they are modified
                self.mask_temp_cache.append(weight_affected_mask)
                self.param_temp_cache.append(param.data * weight_affected_mask)

                # Modify the weights
                param.data[both_memristors_lrs_lrs_mask] = 0.0
                param.data[both_memristors_lrs_hrs_mask] = parameters_abs_max
                param.data[both_memristors_hrs_lrs_mask] = -parameters_abs_max
                param.data[both_memristors_hrs_hrs_mask] = 0.0
                param.data[pos_memristors_lrs_mask] = param.data[pos_memristors_lrs_mask] + parameters_abs_max
                param.data[torch.bitwise_and(pos_memristors_hrs_mask, param > 0)] = 0.0
                param.data[neg_memristors_lrs_mask] = param.data[neg_memristors_lrs_mask] - parameters_abs_max
                param.data[torch.bitwise_and(neg_memristors_hrs_mask, param < 0)] = 0.0
                param.clamp_(-parameters_abs_max, parameters_abs_max)


    def dropconnect(self):
        """
        Sets a proportion of weights to 0
        :return: None
        """
        with torch.no_grad():
            for param in self.parameters():
                # Select the weights to drop
                weights_to_drop_mask = torch.rand_like(param) < settings.dropconnect_prob
                # Save the weights and masks used
                self.mask_temp_cache.append(weights_to_drop_mask)
                self.param_temp_cache.append(param.data * weights_to_drop_mask)
                # Set the weights to 0
                param.data[weights_to_drop_mask] = 0.0

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
