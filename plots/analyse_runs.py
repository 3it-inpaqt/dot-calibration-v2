from collections import Counter
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
from tabulate import tabulate

from datasets.diagram import ChargeRegime
from utils.output import load_runs, set_plot_style


def load_runs_clean(patterns: Union[str, List[str]]) -> pd.DataFrame:
    """
    Load all information form files in the out directory matching with patterns.
    Then process some field for easier use.

    :param patterns: The pattern, or a list of pattern, to filter runs.
    :return: A dataframe containing all information, with the columns as "file.key".
    """
    data = load_runs(patterns)

    # Move metrics results to root level
    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        data[metric.capitalize()] = data['results.final_classification_results'].map(lambda a: a[metric])

    return data


def compare_autotuning():
    data = load_runs_clean(['tuning-michel*'])

    # Compute accuracy and number of steps for each autotuning run
    target = str(ChargeRegime.ELECTRON_1)
    good_charge = Counter()
    total = Counter()
    nb_steps = Counter()
    for _, row in data.iterrows():
        for tuning_result in row['results.tuning_results']:
            # procedure_name contrains the model type too
            result_id = (tuning_result['procedure_name'], tuning_result['diagram_name'])
            total[result_id] += 1
            if tuning_result['charge_area'] == target:
                good_charge[result_id] += 1

            nb_steps[result_id] += tuning_result['nb_steps']

    grouped_results = []  # Group by (diagram - procedure - model)
    for procedure_name, diagram_name in total.keys():
        success_rate = good_charge[(procedure_name, diagram_name)] / total[(procedure_name, diagram_name)]
        mean_steps = nb_steps[(procedure_name, diagram_name)] / total[(procedure_name, diagram_name)]
        procedure_name = procedure_name.replace(' (', '\n').replace(')', '').replace('Uncertainty', ' Unc.')
        grouped_results.append([diagram_name, procedure_name, success_rate, mean_steps])

    # Convert to dataframe
    grouped_results = DataFrame(grouped_results, columns=['diagram', 'procedure_model', 'success', 'mean_steps'])

    # Order by success rate
    order = grouped_results.groupby(['procedure_model'])['success'].mean().reset_index().sort_values('success')

    # Change color base on the model used
    color_map = {'random': 'lightgrey', '\nCNN': 'tab:blue', '\nBCNN': 'tab:purple', '\nOracle': 'grey'}
    colors = []
    for procedure_name in order['procedure_model']:
        for c_filter, color in color_map.items():
            if c_filter in procedure_name:
                colors.append(color)
                break

    # Plot accuracy
    ax = sns.barplot(x="procedure_model", y="success", data=grouped_results, order=order['procedure_model'],
                     palette=colors, ci=None)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # Limit min / max value +/- 10%
    ax.set(ylim=(max(order['success'].min() - 0.1, 0), min(order['success'].max() + 0.1, 1)))
    plt.ylabel('Success rate')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures success to reach {target} electron regime')
    plt.tight_layout()
    plt.show(block=False)

    # Plot steps
    sns.barplot(x="procedure_model", y="mean_steps", data=grouped_results, order=order['procedure_model'],
                palette=colors, ci=None)
    plt.ylabel('Average steps')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures number of steps')
    plt.tight_layout()
    plt.show(block=False)


def compare_autotuning_stats():
    data = load_runs_clean([
        'ref-louis-ff',
        'ref-louis-cnn',
        'ref-louis-bcnn',
    ])
    stats_cols_names = {'timers.network_training': 'Training (s)',
                        'network_info.total_params': 'Nb parameters',
                        'Accuracy': 'Accuracy',
                        'F1': 'F1', }

    hidden_cols = ('settings.plot_diagrams', 'settings.save_gif', 'settings.save_images', 'settings.show_images',
                   'settings.visual_progress_bar', 'settings.logger_console_level', 'settings.save_video',
                   'settings.console_color', 'settings.seed', 'settings.label_offset_x', 'settings.label_offset_y',
                   'settings.pixel_size')

    # Remove hidden columns
    for col in hidden_cols:
        del data[col]

    # Select stats columns
    stats_cols = data[stats_cols_names.keys()]
    stats_cols = stats_cols.rename(columns=stats_cols_names)

    # Select settings columns
    settings_cols = data.filter(regex=r'^settings\..*')
    # Convert list to tuple for hashable uniques values
    settings_cols = settings_cols.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
    # Remove columns with same values
    nb_unique = settings_cols.nunique()
    cols_to_drop = nb_unique[nb_unique == 1].index
    settings_cols.drop(cols_to_drop, axis=1, inplace=True)
    # Rename and order
    settings_cols.rename(columns={n: n[9:] for n in settings_cols.columns}, inplace=True)
    settings_cols.insert(0, 'run_name', settings_cols.pop('run_name'))

    # Print all
    all_cols = pd.concat([settings_cols, stats_cols], axis=1)
    all_cols.rename(columns={n: n.capitalize().replace('_', ' ') for n in all_cols.columns}, inplace=True)
    print(tabulate(all_cols, showindex=False, headers="keys", tablefmt='fancy_grid'))


def compare_models():
    data = load_runs('ref-01*')

    # Rename col for auto plot labels
    data.rename(columns={'settings.research_group': 'Datasets',
                         'settings.model_type': 'Model'}, inplace=True)

    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        metric_col = metric.capitalize()
        data[metric_col] = data['results.final_classification_results'].map(lambda a: a[metric])

        order = ['FF', 'CNN', 'BCNN']
        ax = sns.barplot(x="Model", y=metric_col, data=data, hue='Datasets', ci='sd', order=order)

        # Limit min / max value +/- 10%
        ax.set(ylim=(max(data[metric_col].min() - 0.1, 0), min(data[metric_col].max() + 0.1, 1)))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        # Write values in bars
        for ctr in ax.containers:
            ax.bar_label(ctr, padding=-25, labels=[f'{x:.1%}' for x in ctr.datavalues], fontsize=14, color='w')

        plt.title(f'Trained models {metric_col} comparison')
        # Put the legend out of the figure (middle bottom)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='center', ncol=2)
        plt.tight_layout()
        plt.show(block=False)


def repeat_analyse():
    # Print average result of a repeated run with different seed
    data = load_runs('tmp-*')

    mean_acc = data["results.final_accuracy"].mean()
    std_acc = data["results.final_accuracy"].std()
    std_baseline = data['results.baseline_std_test_accuracy'][0]
    print(f'{len(data):n} runs - avg accuracy: {mean_acc:.2%} (std:{std_acc:.2%}) - baseline: {std_baseline:.2%}')


def layers_size_analyse():
    data = load_runs('layers_size-*')

    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        data[metric.capitalize()] = data['results.final_classification_results'].map(lambda a: a[metric])
        data['Nb Layer'] = data['settings.hidden_layers_size'].map(lambda a: len(a))

        # Rename col for auto plot labels
        data.rename(columns={'network_info.total_params': 'Nb Parameter',
                             'settings.research_group': 'Datasets',
                             'settings.model_type': 'Model'}, inplace=True)

        # Hidden size -> metric
        grid = sns.relplot(data=data, kind='line', x='Nb Parameter', y=metric.capitalize(),
                           hue='Datasets', col='Model', row='Nb Layer',
                           facet_kws={'sharex': 'col', 'margin_titles': True})

        grid.set_axis_labels(x_var='Total number of parameters')
        grid.set(xscale='log')
        # grid.axes.yaxis.set_major_formatter(PercentFormatter(1))
        # grid.fig.suptitle(f'Evolution of the {metric} score in function of number of parameters')
        # TODO main title, y-axis as % and global x and y axes

    plt.tight_layout()
    plt.show(block=False)


def patch_size_analyse():
    # Load selected runs' files
    data = load_runs('patch_size_cnn*')

    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        data[metric.capitalize()] = data['results.final_classification_results'].map(lambda a: a[metric])

        # TODO add independent lines for both classes
        # if metric != 'accuracy':
        #     data[f'No Line {metric.capitalize()}'] = data['results.final_classification_results']\
        #         .map(lambda a: a['classes'][0][metric])
        #     data[f'Line {metric.capitalize()}'] = data['results.final_classification_results']\
        #         .map(lambda a: a['classes'][1][metric])

        # Rename col for auto plot labels
        data.rename(columns={'network_info.total_params': 'Nb Parameter',
                             'settings.research_group': 'Datasets',
                             'settings.model_type': 'Model'}, inplace=True)

        # Patch size -> metric
        sns.relplot(data=data, kind='line', x='settings.patch_size_x', y=metric.capitalize(), hue='Datasets')
        # if metric != 'accuracy':
        #     sns.relplot(data=data, kind='line', x='settings.patch_size_x', y=f'No Line {metric.capitalize()}',
        #                 hue='Datasets', linestyle=':')
        #     sns.relplot(data=data, kind='line', x='settings.patch_size_x', y=f'Line {metric.capitalize()}',
        #                 hue='Datasets', linestyle='--')
        # sns.lineplot(data=data, x='settings.patch_size_x', y='results.baseline_std_test_accuracy',
        #              label='STD baseline')

        plt.title(f'Evolution of the {metric} in function of patch size')
        plt.xlabel('Patch size in pixel')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        plt.tight_layout()
        plt.show(block=False)

        # Patch size -> Train size
        # uniq_seed = data.drop_duplicates(subset='settings.patch_size_x').copy()
        # uniq_seed['total_train'] = uniq_seed['results.train_dataset_size'] + uniq_seed['results.train_dataset_augmentation']
        # sns.lineplot(data=uniq_seed, x='settings.patch_size_x', y='total_train')
        # plt.title('Size of training dataset in function of patch size')
        # plt.xlabel('Patch size in pixel')
        # plt.ylabel('Number of training patch')
        # plt.show(block=False)


def batch_size_analyse():
    # Load selected runs' files
    data = load_runs('train_batch_size-*')
    # Rename col for auto plot labels
    data.rename(columns={'timers.network_training': 'Training Time (s)',
                         'settings.batch_size': 'Batch Size',
                         'settings.research_group': 'Datasets',
                         'settings.model_type': 'Model'}, inplace=True)

    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        data[metric.capitalize()] = data['results.final_classification_results'].map(lambda a: a[metric])

        # Patch size -> metric
        grid = sns.relplot(data=data, kind='line', x='Batch Size', y=metric.capitalize(), hue='Datasets', col='Model')

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        grid.fig.suptitle(f'Evolution of the {metric.capitalize()} in function of batch size')

        plt.tight_layout()
        plt.show(block=False)

    # Patch size -> training time
    sns.relplot(data=data, kind='line', x='Batch Size', y='Training Time (s)', hue='Datasets', col='Model')

    grid.fig.suptitle('Evolution of the training time in function of batch size')

    plt.tight_layout()
    plt.show(block=False)


if __name__ == '__main__':
    # Set plot style
    set_plot_style()

    compare_autotuning()
