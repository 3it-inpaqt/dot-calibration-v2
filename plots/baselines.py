from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from utils.output import save_plot


def plot_data_feature_distribution(feature_values: torch.Tensor, labels: torch.Tensor, feature_name: str, title: str,
                                   threshold: float = None, label_names: List[str] = None) -> None:
    """
    Plot the distribution of a specific feature of a dataset, grouped by class.

    :param feature_values: The feature value (extracted from the dataset)
    :param labels: The labels (class) of each value
    :param feature_name: The name of the feature (for plot information)
    :param title: The title of the plot
    :param threshold: An optional threshold value to add in the plot
    :param label_names: The labels printable name for the legend
    """
    # We have to deal with legend labels manually
    legend_labels = label_names.copy()
    legend_labels.reverse()

    # Convert to dataframe to please seaborn
    df = pd.DataFrame({feature_name: feature_values.cpu(), 'labels': labels.cpu()})

    sns.displot(df, x=feature_name, hue='labels', kind='kde', fill=True, legend=False)

    if threshold:
        plt.axvline(x=threshold, color='r')
        legend_labels.insert(0, 'best threshold')

    plt.ylabel('density')
    plt.title(title)
    plt.legend(labels=legend_labels)

    save_plot(f'{feature_name}_distribution')
