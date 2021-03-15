from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from utils.output import save_plot
from utils.settings import settings


def plot_train_progress(loss_evolution: List[float], accuracy_evolution: List[dict] = None,
                        batch_per_epoch: int = 0) -> None:
    """
    Plot the evolution of the loss and the accuracy during the training.

    :param loss_evolution: A list of loss for each batch.
    :param accuracy_evolution: A list of dictionaries as {batch_num, test_accuracy, train_accuracy}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    """
    with sns.axes_style("ticks"):
        fig, ax1 = plt.subplots()

        # Vertical lines for each batch
        if batch_per_epoch:
            for epoch in range(0, len(loss_evolution) + 1, batch_per_epoch):
                ax1.axvline(x=epoch, color='black', linestyle=':', alpha=0.2,
                            label='epoch' if epoch == 0 else '')  # only one label for the legend

        # Plot loss
        ax1.plot(loss_evolution, label='loss', color='tab:gray')
        ax1.set_ylabel('Loss')
        ax1.set_ylim(bottom=0)

        if accuracy_evolution:
            # Plot the accuracy evolution if available
            ax2 = plt.twinx()
            checkpoint_batches = [a['batch_num'] for a in accuracy_evolution]
            ax2.plot(checkpoint_batches, [a['test_accuracy'] for a in accuracy_evolution],
                     label='test accuracy',
                     color='tab:green')
            ax2.plot(checkpoint_batches, [a['train_accuracy'] for a in accuracy_evolution],
                     label='train accuracy',
                     color='tab:orange')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim(bottom=0, top=1)

            # Place legends at the bottom
            ax1.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.25))
            ax2.legend(loc="lower right", bbox_to_anchor=(1.2, -0.25))
        else:
            # Default legend position if there is only loss
            ax1.legend()

        plt.title('Training evolution')
        ax1.set_xlabel(f'Batch number (size: {settings.batch_size:n})')
        save_plot('train_progress')


def plot_confusion_matrix(nb_labels_predictions: np.ndarray, class_names: List[str] = None,
                          annotations: bool = True) -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param class_names: The list of readable classes names
    :param annotations: If true the accuracy will be written in every cell
    """

    # TODO add info columns as Supplementary Table 1 of
    #  https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-018-04484-2/MediaObjects/41467_2018_4484_MOESM1_ESM.pdf

    overall_accuracy = nb_labels_predictions.trace() / nb_labels_predictions.sum()
    rate_labels_predictions = nb_labels_predictions / nb_labels_predictions.sum(axis=1).reshape((-1, 1))

    # TODO dynamic font size
    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto',
                annot=annotations,
                cbar=(not annotations))
    plt.title(f'Confusion matrix of {len(nb_labels_predictions)} classes '
              f'with {overall_accuracy * 100:.2f}% overall accuracy')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    save_plot('confusion_matrix')
