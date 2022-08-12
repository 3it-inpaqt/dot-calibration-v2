from copy import copy
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from classes.data_structures import ClassificationMetrics
from datasets.qdsd import QDSDLines
from plots.data import plot_samples
from utils.output import save_plot
from utils.settings import settings


def plot_train_progress(loss_evolution: List[float],
                        metrics_evolution: List[dict] = None,
                        batch_per_epoch: int = 0,
                        best_checkpoint: dict = None) -> None:
    """
    Plot the evolution of the loss and the main metric during the training.

    :param loss_evolution: A list of loss for each batch.
    :param metrics_evolution: A list of dictionaries as {batch_num, validation, train}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    :param best_checkpoint: A dictionary containing information about the best version of the network according to
        validation score processed during checkpoints.
    """
    with sns.axes_style("ticks"):
        fig, ax1 = plt.subplots()

        # Vertical lines for each batch
        if batch_per_epoch:
            if len(loss_evolution) / batch_per_epoch > 400:
                batch_per_epoch *= 100
                label = '100 epochs'
            elif len(loss_evolution) / batch_per_epoch > 40:
                batch_per_epoch *= 10
                label = '10 epochs'
            else:
                label = 'epoch'

            for epoch in range(0, len(loss_evolution) + 1, batch_per_epoch):
                # Only one with label for clean legend
                ax1.axvline(x=epoch, color='black', linestyle=':', alpha=0.2, label=label if epoch == 0 else '')

        # Plot loss
        ax1.plot(loss_evolution, label='loss', color='dodgerblue', alpha=0.3)
        ax1.set_ylim(bottom=0)

        if metrics_evolution:
            legend_y_anchor = -0.25

            # Plot the main metric evolution if available
            ax2 = plt.twinx()
            checkpoint_batches = [a['batch_num'] for a in metrics_evolution]
            # Train evolution
            train_main_metric = [a['train'].main for a in metrics_evolution]
            ax2.plot(checkpoint_batches, train_main_metric, label=f'train {settings.main_metric}',
                     color='limegreen', linestyle=(0, (2, 1)), alpha=0.5)
            # Testing evolution
            if 'test' in metrics_evolution[0] and metrics_evolution[0]['test'] is not None:
                test_main_metric = [a['test'].main for a in metrics_evolution]
                ax2.plot(checkpoint_batches, test_main_metric, label=f'test {settings.main_metric}',
                         color='darkolivegreen')
            # Validation evolution
            if 'validation' in metrics_evolution[0] and metrics_evolution[0]['validation'] is not None:
                valid_main_metric = [a['validation'].main for a in metrics_evolution]
                ax2.plot(checkpoint_batches, valid_main_metric, label=f'validation {settings.main_metric}',
                         color='green')

            # Star marker for best validation metric
            if best_checkpoint and best_checkpoint['batch_num'] is not None:
                ax2.plot(best_checkpoint['batch_num'], best_checkpoint['score'], color='green',
                         marker='*', markeredgecolor='k', markersize=10, label=f'best valid. {settings.main_metric}')
                legend_y_anchor -= 0.1

            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax2.set_ylim(bottom=0, top=1)

            # Color for both axes
            ax2.set_ylabel(settings.main_metric.capitalize(), color='darkgreen', fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='darkgreen')
            ax1.set_ylabel('Loss', color='dodgerblue', fontweight='bold')
            ax1.tick_params(axis='y', labelcolor='dodgerblue')

            # Place legends at the bottom
            ax1.legend(loc="lower left", bbox_to_anchor=(-0.1, legend_y_anchor))
            ax2.legend(loc="lower right", bbox_to_anchor=(1.2, legend_y_anchor))
        else:
            # Default legend position if there is only loss
            ax1.legend()
            # And no label color
            ax1.set_ylabel('Loss')

        plt.title('Training evolution')
        ax1.set_xlabel(f'Batch number\n(batch size: {settings.batch_size:n})')
        save_plot('train_progress')


def plot_confusion_matrix(nb_labels_predictions: np.ndarray,
                          metrics: ClassificationMetrics,
                          nb_labels_unknown_predictions: np.ndarray = None,
                          class_names: List[str] = None,
                          annotations: bool = True,
                          plot_name: str = 'confusion_matrix') -> None:
    """
    Plot the confusion matrix for a set a predictions.

    :param nb_labels_predictions: The count of prediction for each label.
    :param nb_labels_unknown_predictions: The count of prediction for each label where the confidence is under the
        threshold.
    :param metrics: The classification results metrics.
    :param class_names: The list of readable classes names.
    :param annotations: If true the recall will be written in every cell.
    :param plot_name: The name of the plot file name.
    """

    labels_classes = copy(class_names) if class_names else list(range(len(nb_labels_predictions)))
    pred_classes = copy(class_names) if class_names else list(range(len(nb_labels_predictions)))

    if nb_labels_unknown_predictions is not None:
        unknown_rate_str = f'{nb_labels_unknown_predictions.sum() / nb_labels_predictions.sum():.2%}'
        # Remove the unknown prediction from to matrix count
        nb_labels_predictions = nb_labels_predictions - nb_labels_unknown_predictions
        # Stack the sum of unknown prediction at the right of the matrix
        nb_labels_predictions = np.c_[nb_labels_predictions, nb_labels_unknown_predictions.sum(axis=1)]
        pred_classes.append('unknown')

    rate_labels_predictions = nb_labels_predictions / nb_labels_predictions.sum(axis=1).reshape((-1, 1))

    sns.heatmap(rate_labels_predictions,
                vmin=0,
                vmax=1,
                square=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=pred_classes,
                yticklabels=labels_classes,
                annot=annotations,
                cbar=(not annotations))
    classes_str = f'{len(nb_labels_predictions)} classes{"" if nb_labels_unknown_predictions is None else " + unknown"}'
    plt.title(f'Confusion matrix of {classes_str}\n'
              f'accuracy {metrics.accuracy:.2%} | precision {metrics.precision:.2%}\n'
              f'recall {metrics.recall:.2%} | F1 {metrics.f1:.2%}'
              f'{"" if nb_labels_unknown_predictions is None else " | unknown " + unknown_rate_str}')
    plt.xlabel('Predictions')
    plt.ylabel('Labels')
    save_plot(plot_name)


def plot_classification_sample(samples_per_case: List[List[List[Tuple[List, float]]]], class_names: List[str],
                               nb_labels_predictions: np.ndarray) -> None:
    """
    Plot samples of every classification cases.

    :param samples_per_case: The list of cases as:
        [label class index][prediction class index][(patch value, confidence)]
    :param class_names: The class name to convert the indexes
    :param nb_labels_predictions: The count of each classification category [label class index][prediction class index]
    """
    for label, predictions in enumerate(samples_per_case):
        for prediction, patchs in enumerate(predictions):
            if len(patchs) > 0:
                # TODO this zip / unzip chain is messy
                patchs_values, confidences = zip(*patchs)
                # Get the total number of item for the current label-prediction couple
                category_size = nb_labels_predictions[label][prediction]

                if label == prediction:
                    title = f'Good classification of "{class_names[label]}"' \
                            f'\nSample {len(patchs_values):n}/{category_size:n}'
                else:
                    title = f'Bad classification of "{class_names[label]}" (detected as {class_names[prediction]})' \
                            f'\nSample {len(patchs_values):n}/{category_size:n}'

                plot_samples(patchs_values, title, f'classification_{class_names[label]}-{class_names[prediction]}',
                             confidences=confidences)


def plot_confidence(confidence_per_case: List[List[List[float]]], unknown_thresholds: List[float]) -> None:
    """
    Plot the confidence density based on validity of the classification.

    :param confidence_per_case: The list of confidence score per classification case
      as [label class index][prediction class index]
    """
    good_pred_confidence = list()
    bad_pred_confidence = list()

    # Group confidence by prediction success
    for label in range(len(confidence_per_case)):
        for prediction in range(len(confidence_per_case[label])):
            if label == prediction:
                good_pred_confidence.extend(confidence_per_case[label][prediction])
            else:
                bad_pred_confidence.extend(confidence_per_case[label][prediction])

    # Convert to dataframe to please seaborn
    df = pd.DataFrame({'confidence': good_pred_confidence + bad_pred_confidence,
                       'is_correct': [True] * len(good_pred_confidence) + [False] * len(bad_pred_confidence)})

    palette = {True: "tab:green", False: "tab:red"}
    sns.displot(df, x='confidence', hue='is_correct', kind='hist', palette=palette, legend=False, multiple="layer",
                element="step", bins=100)

    colors = ['tab:purple', 'tab:olive']
    for cls, threshold, color in zip(QDSDLines.classes, unknown_thresholds, colors):
        plt.axvline(x=threshold, color=color, linestyle=':')

    plt.ylabel('Count')
    plt.xlabel('Classification confidence')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.yscale('log')
    plt.legend(labels=['Good classification', 'Bad classification'] +
                      [f'Confidence threshold {cls}' for cls in QDSDLines.classes], loc='upper left')
    plt.title(f'Classification confidence\nfor {len(df):n} test patches')

    save_plot('confidence_distribution')


def plot_confidence_threshold_tuning(thresholds: List, scores_history: List, sample_size: int) -> None:
    """
    Plot the evolution of performance score depending on the confidence threshold.

    :param thresholds: The thresholds tested
    :param scores_history: The score for each threshold and each classes
    :param sample_size: The size of the dataset used to compute the scores
    """
    scores_history = list(zip(*scores_history))  # Group by class

    for i in range(len(scores_history)):
        plt.plot(thresholds, scores_history[i], label=QDSDLines.classes[i])

    plt.ylabel('Score (lower is better)')
    plt.xlabel('Confidence threshold')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.title(f'Evolution of performance score\ndepending on the confidence threshold ({sample_size} sample)')

    save_plot('threshold_tuning')
