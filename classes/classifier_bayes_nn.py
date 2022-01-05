import math
from typing import List

import torch

from classes.classifier_nn import ClassifierNN
from utils.settings import settings


class ClassifierBayesNN(ClassifierNN):

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
        # Round the samples mean value to 0 or 1
        predictions = torch.round(means).bool()

        # Compute the confidence metric according to the settings
        confidences = None
        if settings.bayesian_confidence_metric == 'std':
            confidences = ClassifierBayesNN._std_confidence(outputs)
        elif settings.bayesian_confidence_metric == 'norm_std':
            confidences = ClassifierBayesNN._norm_std_confidence(outputs)
        elif settings.bayesian_confidence_metric == 'entropy':
            confidences = ClassifierBayesNN._entropy_confidence(outputs)
        elif settings.bayesian_confidence_metric == 'norm_entropy':
            confidences = ClassifierBayesNN._norm_entropy_confidence(outputs)

        return predictions, confidences.cpu()

    @staticmethod
    def _std_confidence(model_outputs):
        return model_outputs.std(axis=0)

    @staticmethod
    def _norm_std_confidence(model_outputs):
        stds = model_outputs.std(axis=0)
        # The model output is between 0 and 1, so the mean value is 0.5
        return 1 - (stds / 0.5)

    @staticmethod
    def _entropy_confidence(model_outputs):
        stds = model_outputs.std(axis=0)
        pi = torch.Tensor([math.pi])
        # FIXME the formula doesn't seem good
        entropies = torch.log(2 * pi * torch.pow(stds.cpu(), 2)) / 2 + 1 / 2
        return entropies

    @staticmethod
    def _norm_entropy_confidence(model_outputs):
        raise NotImplemented
