from typing import List

import torch

from datasets.qdsd import QDSDLines
from plots.baselines import plot_data_feature_distribution
from utils.logger import logger
from utils.timer import SectionTimer


class GapBaseline:
    """
    Baseline base on the values gap (Max - Min) of patches values.
    The classification is made according to an optimal threshold value.
    """
    threshold: float = None
    class_below: bool = True
    confidence_thresholds: List[float] = None

    @SectionTimer('gap baseline training', 'debug')
    def train(self, train_dataset: QDSDLines) -> None:
        """
        Train the model to find the optimal threshold value splitting the classes with the best accuracy.

        :param train_dataset: The training dataset.
        """

        labels = train_dataset.get_labels()
        patches_gap = self.transform_data(train_dataset.get_data())

        nb_values = len(patches_gap)
        best_threshold = None
        best_class_below = None
        best_good_classified = 0

        # TODO could probably be optimised with dichotomic search
        for current_gap_threshold in patches_gap:

            # Count the number of correct classification base on this threshold
            nb_good_classified = int(torch.sum((patches_gap < current_gap_threshold) == labels))

            # Check if this is the best threshold so far
            if nb_good_classified > best_good_classified:
                best_good_classified = nb_good_classified
                best_threshold = current_gap_threshold
                best_class_below = True
            # Check if this is the opposite of the best threshold so far
            elif (nb_values - nb_good_classified) > best_good_classified:
                best_good_classified = nb_values - nb_good_classified
                best_threshold = current_gap_threshold
                best_class_below = False

        self.threshold = float(best_threshold)
        self.class_below = best_class_below

        train_accuracy = best_good_classified / nb_values
        logger.debug(f'Gap baseline best threshold: {self.threshold:.4f} ({train_accuracy:.2%} accuracy on train)')

        plot_data_feature_distribution(patches_gap, labels, 'gap value',
                                       'Max - Min distribution in patches values\nby class',
                                       self.threshold,
                                       train_dataset.classes)

    def eval(self):
        """ Method created to fake a torch.Module behaviour """
        pass

    def infer(self, inputs, _) -> (List[bool], List[float]):
        """
        Simulate network inference for classification a set of input.

        :param inputs: The inputs to classify.
        :param _: Compatibility with bayesian infer.
        :return: The classes inferred by this method and the confidences these this results (between 0 and 1).
        """
        scores = self(inputs)
        predictions = torch.round(scores).bool()  # Round to 0 or 1
        confidences = torch.ones_like(predictions)  # Hardcode confidence at 100% for the baseline
        return predictions, confidences

    def __call__(self, patch_batch: torch.Tensor) -> torch.Tensor:
        """
        Classify a batch of patches, based on the threshold value.

        :param patch_batch: The list of patch.
        :return: The model prediction (list of class).
        """

        # Invert the classification base on the class under the threshold.
        if self.class_below:
            return (self.transform_data(patch_batch) < self.threshold).float()

        return (self.transform_data(patch_batch) >= self.threshold).float()

    @staticmethod
    def transform_data(patches: torch.Tensor) -> torch.Tensor:
        """
        Convert a list of patch into a list a standard deviation values.

        :param patches: The patches to convert.
        """
        # Flatten patches
        patches = patches.view((patches.shape[0], -1))
        # Compute max - min of each patches
        return torch.max(patches, dim=1)[0] - torch.min(patches, dim=1)[0]
