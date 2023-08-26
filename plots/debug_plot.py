from typing import List

import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from datasets.qdsd import QDSDLines
from utils.logger import logger
from utils.output import save_plot
from utils.settings import settings


def plot_repartition(data, file_name):
    """
    Plot repartition of the dataset
    :param data: data
    :param file_name: file name
    :return:
    """
    # Skip saving if the name of the run is not set
    if settings.is_unnamed_run():
        return

    Data = []
    for inputs, labels in data:
        if len(labels) == 2:
            Data.append(QDSDLines.class_mapping(labels))
        else:
            for label in labels:
                Data.append(QDSDLines.class_mapping(label))

    # Generate labels
    labels = {i: f'Line {i}' for i in range(1, settings.dot_number + 1)}
    labels[0] = 'No line'
    labels[settings.dot_number + 1] = 'Crosspoint'

    # Convert numbers to labels
    data_labels = [labels[i] for i in Data]

    # Generate colors
    if settings.dot_number != 2:
        colors = cm.rainbow(np.linspace(0, 1, settings.dot_number + 2))
    else:
        colors = ['deepskyblue', 'yellowgreen', 'salmon', 'yellow']

    # Create a dictionary to map labels to colors
    label_color_dict = {label: color for label, color in zip(labels.values(), colors)}

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    # Unique labels and their counts
    unique_labels, counts = np.unique(data_labels, return_counts=True)

    # Create an array for the positions of the bars on the x-axis
    r = np.arange(len(unique_labels))

    # For each unique label
    for i, label in enumerate(unique_labels):
        # Plot a bar with the color associated with the label
        ax.bar(r[i], counts[i], color=label_color_dict[label], edgecolor='black', width=0.6)
        # Add the number of occurrences above the bar
        ax.text(r[i], counts[i] + 0.5, str(counts[i]), ha='center', va='bottom')

    # Center labels on each bar
    ax.set_xticks(r)
    ax.set_xticklabels(unique_labels)

    # Set title and labels
    ax.set_title(f'{file_name}: {len(Data)}')
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')

    # Show plot
    save_plot(file_name=file_name, figure=fig)
    logger.debug(f'Repartition save as {file_name}.png')
    if not settings.show_images:
        plt.close(fig)


def classification(data, labels, target):
    logger.debug(f'Classification {QDSDLines.classes[target]}')
    if len(labels) == 2:
        if QDSDLines.class_mapping(labels) == target:
            return data, labels
        else:
            [], []
    dat = []
    label = []
    for nb in range(len(labels)):
        lab = QDSDLines.class_mapping(labels[nb])
        if lab == target:
            dat.append(data[nb])
            label.append(labels[nb])

    if not dat:
        return [], []

    d = torch.stack(dat)
    l = torch.stack(label)
    # logger.debug(f'After class: nb = {len(d)}, Inputs: {np.shape(d)}, labels: {np.shape(l)}, target: {target}')
    return d, l


def plot_train_progress_class(loss: List[float], metrics_evolution: List[dict], nb: int, color: List[float],
                              batch_per_epoch: int = 0, best_checkpoint: dict = None) -> None:
    """
    Plot the evolution of the loss and the main metric during the training.

    :param ax_loss: Matplotlib axes for loss plot.
    :param ax_metric: Matplotlib axes for metric plot.
    :param loss: A list of loss for each batch.
    :param nb: corresponding class
    :param metrics_evolution: A list of dictionaries as {batch_num, validation, train}.
    :param batch_per_epoch: The number of batch per epoch to plot x ticks.
    :param best_checkpoint: A dictionary containing information about the best version of the network according to
        validation score processed during checkpoints.
    """

    if batch_per_epoch:
        if len(loss) / batch_per_epoch > 400:
            batch_per_epoch *= 100
        elif len(loss) / batch_per_epoch > 40:
            batch_per_epoch *= 10

    # == Loss representation == #

    if len(metrics_evolution) > 0:
        plt.subplot(settings.dot_number + 3, 2, 1 + nb * 2)
    else:
        plt.subplot(settings.dot_number + 3, 1, 1 + nb)

    # plt.gca().set_facecolor('white')
    # plt.gca().spines['top'].set_visible(True)
    # plt.gca().spines['right'].set_visible(True)

    # Vertical lines for each batch
    if batch_per_epoch:
        for epoch in range(0, len(loss) + 1, batch_per_epoch):
            plt.axvline(x=epoch, color='black', linestyle=':', alpha=0.2)
    plt.plot(loss, color=color)
    # loss display
    # plt.ylim(bottom=0, top=max(loss)*1.2)
    if nb == settings.dot_number + 2:
        plt.grid(False, axis='x')
        plt.xlabel('Step')
        plt.ylabel(f'All class')
    else:
        plt.ylabel(f'{QDSDLines.classes[nb]}')
        plt.gca().get_xaxis().set_visible(False)
    if nb == 0:
        plt.title('Training: loss')

    if not len(metrics_evolution) > 0:
        return

    # == Train == #

    plt.subplot(1, 2, 2)
    plt.title('Training: train')
    # plt.gca().set_facecolor('white')
    # plt.gca().spines['top'].set_visible(True)
    # plt.gca().spines['right'].set_visible(True)
    # Vertical lines for each batch
    if nb == 0 and batch_per_epoch:
        for epoch in range(0, len(loss) + 1, batch_per_epoch):
            plt.axvline(x=epoch, color='black', linestyle=':', alpha=0.2)
    legend_y_anchor = -0.25

    # Plot the main metric evolution if available
    checkpoint_batches = [a['batch_num'] for a in metrics_evolution]
    # Train evolution
    train_main_metric = [a['train'].main for a in metrics_evolution]
    plt.plot(checkpoint_batches, train_main_metric, color=lighten_color(color, 2), linestyle=(0, (2, 1)), alpha=0.3)

    # Validation evolution
    if 'validation' in metrics_evolution[0] and metrics_evolution[0]['validation'] is not None:
        valid_main_metric = [a['validation'].main for a in metrics_evolution]
        plt.plot(checkpoint_batches, valid_main_metric, color=color, alpha=0.5)

        # Star marker for best validation metric
        if nb == (settings.dot_number + 2) and best_checkpoint and best_checkpoint['batch_num'] is not None:
            plt.plot(best_checkpoint['batch_num'], best_checkpoint['score'], color='tab:gray',
                     marker='*', markeredgecolor='k', markersize=10)
            plt.axvline(best_checkpoint['batch_num'], best_checkpoint['score'], color='tab:gray')
            legend_y_anchor -= 0.1

    # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    # plt.ylim(bottom=0)

    # Color for both axes
    plt.ylabel(f'Metric: {settings.main_metric.capitalize()}')
    plt.xlabel('Step')
    valid = mlines.Line2D([], [], color='black',
                          label=f'Validation dataset')
    train = mlines.Line2D([], [], linestyle=(0, (2, 1)), color='lightgray',
                          label=f'Train dataset')
    best = mlines.Line2D([], [], color='white', marker='*', markeredgecolor='k', markersize=10,
                         label=f'Best network {settings.main_metric}')

    plt.legend(handles=[valid, train, best], loc='upper left')


def lighten_color(input_color, amount_to_lighten):
    """
    Returns a lighter color.
    """
    cmap = LinearSegmentedColormap.from_list("temp_cmap", [input_color, (1, 1, 1)])
    return cmap(amount_to_lighten)


def generate_dataset(dataset, dataset_name):
    data_list = []
    label_list = []
    for u in range(len(dataset)):
        data_list.append(dataset[u][0])
        label_list.append(dataset[u][1])
    return [QDSDLines(classification(data_list, label_list, j), f'check_class_{dataset_name}_{j}')
            for j in range(settings.dot_number + 2)]
