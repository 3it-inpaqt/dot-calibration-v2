import math
from typing import Literal

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torchmetrics.classification import BinaryConfusionMatrix

from utils.misc import short_number
from utils.output import save_plot


def plot_confusion_matrix(cm: BinaryConfusionMatrix, title: str, file_name: str,
                          annotation_rate: bool = True, annotation_count: bool = True,
                          norm_direction: Literal["labels", "predictions", "all"] = 'labels',
                          color_map: str = 'Blues',
                          y_label: str = 'Labels', x_label: str = 'Predictions') -> None:
    """
    Plot a confusion matrix as heatmap.

    Args:
        cm: The confusion matrix metric instance.
        title: The figure title.
        file_name: The name of the plot file.
        annotation_rate: If True, show the normalized rate for each cell of the matrix as a percentage.
        annotation_count: If True, show the item count for each cell of the matrix.
        norm_direction: Define the normalisation direction.
            * "labels": normalisation over the labels (rows)
            * "predictions": normalisation over the predictions (columns)
            * "all": normalisation over the whole matrix (sum)
        color_map: The color map to use for the heatmap.
        y_label: The label of y-axis.
        x_label: The label of x-axis.
    """

    data = cm.confmat.data.to('cpu')
    total = torch.sum(data).item()

    if norm_direction == 'labels':
        normalized_data = data / data.sum(axis=1).reshape((-1, 1))  # Calculate rate per label
    elif norm_direction == 'predictions':
        normalized_data = data / data.sum(axis=0)  # Calculate rate per predictions
    elif norm_direction == 'all':
        normalized_data = data / data.sum()  # Calculate rate over the whole matrix
    else:
        raise ValueError(f'Invalid normalized_data value: "{norm_direction}"')

    annot = False
    ftm = ''
    # Create annotation string if annotation_count or annotation_rate is True
    if annotation_count:
        annot = []
        for i in range(len(data)):
            row_annot = []
            annot.append(row_annot)
            for j in range(len(data[i])):
                annot_str = (f'{normalized_data[i][j]:.2%}\n\n' if annotation_rate else '')
                annot_str += short_number(data[i][j].item())
                row_annot.append(annot_str)
    elif annotation_rate:
        annot = True  # Enable annotation with default string formatter
        ftm = '.2%'

    sns.heatmap(normalized_data,
                vmin=0,
                vmax=1,
                square=True,
                fmt=ftm,
                cmap=color_map,
                annot=annot,
                cbar=not annot)

    plt.title(title + f' ({short_number(total)} items)')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    save_plot(file_name)


def plot_analog_vs_digital_before_threshold(inferences: DataFrame) -> None:
    """
    Plot a scatter figure that represent the difference between of output between the digital and the analog model
    before the threshold.

    Args:
        inferences: A dataframe that contains the inference results.
    """
    # Draw perfect matching reference
    plt.axline((0, 0), slope=1, color='tab:gray', alpha=0.5, label='Perfect matching')

    # Draw thresholds
    plt.axhline(y=0, color='tab:red', linestyle='--', alpha=0.5, label='Thresholds')
    plt.axvline(x=0, color='tab:red', linestyle='--', alpha=0.5)

    # Draw comparison plots
    inferences['Fidelity'] = inferences['analog_output'] == inferences['digital_output']
    ax = sns.scatterplot(inferences, x='digital_before_th', y='analog_logic_before_th', hue='Fidelity',
                         palette={True: 'tab:blue', False: 'tab:orange'}, legend='brief')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))  # Legend outside the box

    plt.ylabel('Analog logic output')
    plt.xlabel('Digital output')
    plt.title('Model output before threshold comparison\nbetween the digital and the analog simulation')
    save_plot('outputs_before_threshold_comparison')


def plot_analog_before_threshold_hist(analog_before_threshold):
    """
    Plot a histogram of the voltage at the output of the circuit
    :param analog_before_threshold: voltage at the output of the circuit
    """
    threshold_precision = 0.01
    step = threshold_precision / 2

    v_max = max(analog_before_threshold)
    v_max_sign = math.copysign(1, v_max)
    v_max = math.ceil(abs(v_max)/(threshold_precision/2)) * step * v_max_sign

    v_min = min(analog_before_threshold)
    v_min_sign = math.copysign(1, v_min)
    v_min = math.ceil(abs(v_min)/(threshold_precision/2)) * step * v_min_sign

    bins = np.arange(v_min, v_max + step, step)
    plt.hist(analog_before_threshold, bins=bins)
    plt.xlabel("(V)")
    plt.ylabel("Frequency")
    plt.title("Voltage at the output of the circuit before the threshold")
    save_plot("analog_before_threshold_hist")