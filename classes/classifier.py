from typing import List, Optional


class Classifier:
    confidence_thresholds: List[float] | float = None

    def infer(self, inputs, nb_samples: Optional[int] = 100) -> (List[bool], List[float]):
        """
        Use a model inference for classification a set of input.

        :param inputs: The inputs to classify.
        :param nb_samples: The number of inference iteration to run on for each input. The inference will be done
         on the mean value. Used in bayesian models.
        :return: The class inferred by the model and the confidence score.
        """

        raise NotImplemented

    def is_above_confident_threshold(self, class_infer: int | bool, confidence: float):
        """
        Check if an inference should be considered as valid.

        :param class_infer: The class inferred
        :param confidence: The model confidence for this classification
        :return: True if the confidence is above or equal to the thresholds. False If not. Always return true if the
         confidence thresholds are not set.
        """

        if self.confidence_thresholds:
            if isinstance(self.confidence_thresholds, float):
                # If the confidence_thresholds is a scalar, every class have the same threshold
                class_thr = self.confidence_thresholds
            else:
                class_thr = self.confidence_thresholds[class_infer]
            return confidence >= class_thr

        return True  # No confidence thresholds defined

    def __str__(self) -> str:
        return type(self).__name__
