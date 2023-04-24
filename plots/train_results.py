from copy import copy
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from classes.data_structures import CalibrationMetrics, ClassificationMetrics, TestMetrics
from datasets.qdsd import QDSDLines
from plots.data import plot_samples
from utils.output import save_plot
from utils.settings import settings
from utils.str_formatting import short_number


def plot_train_progress(loss_evolution: List[float],
                        metrics_evolution: List[Dict[str, Union[TestMetrics, None, int]]] = None,
                        batch_per_epoch: int = 0,
                        best_checkpoint: dict = None,
                        calibration_progress: bool = False) -> None:
    """
    Plot the evolution of the loss and the main metric during the training.

    :param loss_evolution: A list of loss for each batch.
    :param metrics_evolution: A list of dictionaries as {batch_num, validation, train}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    :param best_checkpoint: A dictionary containing information about the best version of the network according to
        validation score processed during checkpoints.
    :param calibration_progress: If True and the metric evolution is provided, the calibration progress will be plotted.
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

        if metrics_evolution and len(metrics_evolution) > 0:
            legend_y_anchor = -0.25

            # Plot the main metric evolution if available
            ax2 = plt.twinx()
            checkpoint_batches = [a['batch_num'] for a in metrics_evolution]
            # Train evolution
            train_main_metric = [a['train'].classification.main for a in metrics_evolution]
            ax2.plot(checkpoint_batches, train_main_metric, label=f'train {settings.main_metric}',
                     color='limegreen', linestyle=(0, (2, 1)), alpha=0.5)

            # Testing evolution
            if 'test' in metrics_evolution[0] and metrics_evolution[0]['test'] is not None:
                test_main_metric = [a['test'].classification.main for a in metrics_evolution]
                ax2.plot(checkpoint_batches, test_main_metric, label=f'test {settings.main_metric}',
                         color='darkolivegreen')
            # Validation evolution
            if 'validation' in metrics_evolution[0] and metrics_evolution[0]['validation'] is not None:
                valid_main_metric = [a['validation'].classification.main for a in metrics_evolution]
                ax2.plot(checkpoint_batches, valid_main_metric, label=f'validation {settings.main_metric}',
                         color='green')

                # Star marker for best validation metric
                if best_checkpoint and best_checkpoint['batch_num'] is not None:
                    ax2.plot(best_checkpoint['batch_num'], best_checkpoint['score'], color='green',
                             marker='*', markeredgecolor='k', markersize=10,
                             label=f'best valid. {settings.main_metric}')
                    legend_y_anchor -= 0.1

                if calibration_progress:
                    calibration_main_metric = [a['validation'].calibration.main for a in metrics_evolution if
                                               a and a != float('nan')]
                    if len(calibration_main_metric) > 0:
                        ax3 = plt.twinx()
                        ax3.plot(checkpoint_batches, calibration_main_metric,
                                 label=settings.main_calibration_metric, color='blueviolet')
                        ax3.set_ylim(bottom=0)
                        ax3.set_ylabel(settings.main_calibration_metric, color='blueviolet', fontweight='bold')
                        ax3.tick_params(axis='y', labelcolor='blueviolet')
                        # Position the calibration axis on the right
                        ax3.spines['right'].set_position(('outward', 70))
                        ax2.yaxis.tick_right()

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


def plot_confidence(confidence_per_case: List[List[List[float]]], unknown_thresholds: List[float],
                    dataset_role: str, combined: bool = True, per_classes: bool = False) -> None:
    """
    Plot the confidence density based on validity of the classification.

    :param confidence_per_case: The list of confidence score per classification case
      as [label class index][prediction class index]
    :param unknown_thresholds: The list of thresholds for each class
    :param dataset_role: The role of the dataset (train, validation, test)
    :param combined: If true the confidence will be combined for all classes
    :param per_classes: If true the confidence will be plotted per class
    """

    assert combined or per_classes, 'At least one of the two plot type must be selected'

    # Create subplots axes
    nb_subplots = 1 + (len(QDSDLines.classes) if per_classes else 0)
    fig, axes = plt.subplots(nb_subplots, 1, figsize=(10, 5 * nb_subplots))

    # Legend color
    palette = {True: "tab:green", False: "tab:red"}
    threshold_colors = ['tab:purple', 'tab:olive']

    # Plot the combined confidence distribution (all classes)
    if combined:
        good_pred_confidence = list()
        bad_pred_confidence = list()

        # Group confidence by prediction success
        for label in range(len(confidence_per_case)):
            for prediction in range(len(confidence_per_case[label])):
                if label == prediction:
                    good_pred_confidence.extend(confidence_per_case[label][prediction])
                else:
                    bad_pred_confidence.extend(confidence_per_case[label][prediction])

        ax = axes[0] if len(axes) > 1 else axes
        # Convert to dataframe to please seaborn
        df = pd.DataFrame({'confidence': good_pred_confidence + bad_pred_confidence,
                           'is_correct': [True] * len(good_pred_confidence) + [False] * len(bad_pred_confidence)})
        # Plot the global confidence distribution
        sns.histplot(df, x='confidence', hue='is_correct', palette=palette, legend=False, multiple="layer",
                     element="step", bins=100, ax=ax)
        ax.set_title(f'Global ({len(df):,d} samples)')
        ax.set_yscale('log')
        ax.set_xlabel('Classification confidence')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        # Plot the thresholds as vertical lines
        for cls, threshold, color in zip(QDSDLines.classes, unknown_thresholds, threshold_colors):
            ax.axvline(x=threshold, color=color, linestyle=':')

    # Plot the confidence distribution for each class
    if per_classes:
        iterator = enumerate(zip(unknown_thresholds, threshold_colors, axes[1 if combined else 0:]))
        for cls_i, (threshold, threshold_color, ax) in iterator:
            good_pred_confidence = list()
            bad_pred_confidence = list()

            # Group confidence by prediction success for the current class
            for label in range(len(confidence_per_case)):
                for prediction in range(len(confidence_per_case[label])):
                    if prediction == cls_i:
                        if label == prediction:
                            good_pred_confidence.extend(confidence_per_case[label][prediction])
                        else:
                            bad_pred_confidence.extend(confidence_per_case[label][prediction])

            # Convert to dataframe to please seaborn
            df = pd.DataFrame({'confidence': good_pred_confidence + bad_pred_confidence,
                               'is_correct': [True] * len(good_pred_confidence) + [False] * len(bad_pred_confidence)})
            # Plot the global confidence distribution
            sns.histplot(df, x='confidence', hue='is_correct', palette=palette, legend=False,
                         multiple="layer", element="step", bins=100, ax=ax)
            ax.set_title(QDSDLines.classes[cls_i].capitalize() + f' ({len(df):,d} samples)')
            ax.set_yscale('log')
            ax.set_xlabel('Classification confidence')
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

            # Plot the threshold as vertical line
            ax.axvline(x=threshold, color=threshold_color, linestyle=':')

    # Add the legend in the first subplot
    first_ax = axes[0] if len(axes) > 1 else axes
    first_ax.legend(labels=['Good classification', 'Bad classification'] +
                           [f'Confidence threshold {cls}' for cls in QDSDLines.classes], loc='upper left')

    # Set global title
    fig.suptitle(f'Confidence distribution from {dataset_role} dataset')

    save_plot(f'confidence_distribution_{dataset_role}')


def plot_confidence_threshold_tuning(thresholds: List, scores_history: List, sample_size: int,
                                     dataset_role: str) -> None:
    """
    Plot the evolution of performance score depending on the confidence threshold.

    :param thresholds: The thresholds tested
    :param scores_history: The score for each threshold and each classes
    :param sample_size: The size of the dataset used to compute the scores
    :param dataset_role: The role (train, valid or test) of the dataset used to compute the scores
    """
    scores_history = list(zip(*scores_history))  # Group by class

    for i in range(len(scores_history)):
        plt.plot(thresholds, scores_history[i], label=QDSDLines.classes[i])

    plt.ylabel('Score (lower is better)')
    plt.xlabel('Confidence threshold')
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.legend()
    plt.title(f'Evolution of performance score\ndepending on the confidence threshold\n'
              f'{sample_size} samples from {dataset_role} dataset')

    save_plot(f'threshold_tuning_{dataset_role}')


def plot_reliability_diagrams(cal_metrics: CalibrationMetrics, dataset_role: str, nb_sample: int, classes: List[str]) \
        -> None:
    """
    Represent the confidence calibration with several bar plots.

    :param cal_metrics: The calibration metrics (include the error and the bins)
    :param dataset_role: The role of the dataset (train, validation, test)
    :param nb_sample: The number of samples used to compute the calibration metrics
    """

    # Plot the main calibration metric
    if cal_metrics.main_bins is not None:
        fig, ax = plt.subplots()
        plot_reliability_diagram(ax, cal_metrics.main, cal_metrics.main_bins, settings.main_calibration_metric,
                                 dataset_role, nb_sample, is_subplot=False)
        save_plot(f'reliability_diagram_{dataset_role.replace(" ", "_")}_{settings.main_calibration_metric}')

    # Plot every non-adaptative reliability diagrams plots
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 1 + len(classes))
    axes = gs.subplots(sharex=False, sharey=True)
    plot_reliability_diagram(axes[0], cal_metrics.ece, cal_metrics.ece_bins, 'ECE', dataset_role, nb_sample)
    for i, cls_str in enumerate(classes):
        if cal_metrics[i] is not None:
            plot_reliability_diagram(axes[1 + i], cal_metrics[i].ece, cal_metrics[i].ece_bins,
                                     f'ECE {cls_str}', dataset_role, nb_sample)
    axes[0].set_ylabel('Accuracy')
    fig.suptitle(f'Reliability diagram from {short_number(nb_sample)} {dataset_role} samples')
    save_plot(f'reliability_diagram_{dataset_role.replace(" ", "_")}_non_adaptative')

    # Plot every adaptative reliability diagrams plots
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(1, 1 + len(classes))
    axes = gs.subplots(sharex=False, sharey=True)
    plot_reliability_diagram(axes[0], cal_metrics.aece, cal_metrics.aece_bins, 'aECE', dataset_role, nb_sample)
    for i, cls_str in enumerate(classes):
        if cal_metrics[i] is not None:
            plot_reliability_diagram(axes[1 + i], cal_metrics[i].aece, cal_metrics[i].aece_bins,
                                     f'aECE {cls_str}', dataset_role, nb_sample)
    axes[0].set_ylabel('Accuracy')
    fig.suptitle(f'Reliability diagram from {short_number(nb_sample)} {dataset_role} samples')
    save_plot(f'reliability_diagram_{dataset_role.replace(" ", "_")}_adaptative')


def plot_reliability_diagram(ax, error_score: float, mean_accuracy_per_bin: pd.Series, error_name: str,
                             dataset_role: str, nb_sample: int, is_subplot: bool = True) -> None:
    """
    Plot one reliability diagram.

    :param ax: The plot axis to plot on.
    :param error_score: The value of the error score.
    :param mean_accuracy_per_bin: The mean accuracy per bin.
    :param error_name: The name of the error score.
    :param dataset_role: The role of the dataset use to compute these bins and error (train, validation, test).
    :param nb_sample: The number of samples used to compute these bins and error.
    :param is_subplot: If this plot is a subplot or not.
    """
    # It is possible that there is no bin (if there is enough diversity)
    if mean_accuracy_per_bin is None:
        return

    random_guess = 0.5  # 1 / nb classes
    with sns.axes_style("ticks"):
        # Bar plot
        ax.bar(x=[b.mid for b in mean_accuracy_per_bin.index], height=mean_accuracy_per_bin.values,
               width=[b.length for b in mean_accuracy_per_bin.index])
        # Reference x=y line
        ax.axline((0, random_guess), slope=1 - random_guess, color='black', linestyle='--', label='Perfect calibration')

        # Limit x-axis from 0% to 100%
        ax.set_xlim(0, 1)
        # Limit y-axis from random guess to 100%
        ax.set_ylim(max(0.0, random_guess * 0.95), 1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

        ax.set_xlabel('Confidence')
        if not is_subplot:
            ax.set_ylabel('Accuracy')
            ax.legend()

        if is_subplot:
            ax.set_title(f'{error_name}: {error_score:.2f}')
        else:
            ax.set_title(f'Reliability diagram ({error_name}: {error_score:.2f})\n'
                         f'from {short_number(nb_sample)} {dataset_role} samples')
