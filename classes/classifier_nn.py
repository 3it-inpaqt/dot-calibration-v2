from typing import Any

from torch import nn

from classes.classifier import Classifier


class ClassifierNN(nn.Module, Classifier):

    def training_step(self, inputs: Any, labels: Any) -> float:
        """
        Define the logic for one training step.

        :param inputs: The input from the training dataset, could be a batch or an item
        :param labels: The label of the item or the batch
        :return: The loss value
        """
        raise NotImplemented

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

    def plot_parameters_sample(self, title: str, file_name: str, n: int = 9) -> None:
        """
        Plot some parameters of the network.

        :param title: The overall title of the plot .
        :param file_name: The name of the file where the plot is saved.
        :param n: The number of sample to include in the plot.
        """
        pass
