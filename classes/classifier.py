from typing import List, Optional


class Classifier:

    def infer(self, inputs, nb_samples: Optional[int] = 100) -> (List[bool], List[float]):
        """
        Use a model inference for classification a set of input.

        :param inputs: The inputs to classify.
        :param nb_samples: The number of inference iteration to run on for each input. The inference will be done
         on the mean value. Used in bayesian models.
        :return: The class inferred by the model and the confidence score.
        """

        raise NotImplemented

    def __str__(self) -> str:
        return type(self).__name__
