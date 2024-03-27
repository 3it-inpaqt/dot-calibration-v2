import math
from collections import Counter
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
from tabulate import tabulate

from datasets.diagram_offline import ChargeRegime, DiagramOffline
from utils.output import load_runs, set_plot_style
from utils.settings import settings

LATEX_FORMAT = False

if LATEX_FORMAT:
    DATASET_NAMES = {'michel_pioro_ladriere': 'Si-QD', 'louis_gaudreau': 'GaAs-QD', 'eva_dupont_ferrier': 'IMEC'}
else:
    DATASET_NAMES = {'michel_pioro_ladriere': 'Michel', 'louis_gaudreau': 'Louis', 'eva_dupont_ferrier': 'Eva'}

PALETTE_MODELS = {'random': 'lightgrey',
                  'FF': 'tab:green',
                  'CNN': 'tab:orange',
                  'BCNN': 'tab:blue',
                  'Oracle': 'grey'}


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
        data[metric.capitalize()] = (data['results.final_classification_results']
                                     .map(lambda a: None if isinstance(a, float) and math.isnan(a) else a[metric]))
        data['ct-' + metric.capitalize()] = (data['results.threshold_classification_results']
        .map(
            lambda a: None if isinstance(a, float) and math.isnan(a) else a[metric])
        )

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
            # Exclude full scan
            if tuning_result['procedure_name'].startswith('FullScan'):
                continue
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

    # Plot success rate
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


def compare_autotuning_alt():
    data = load_runs_clean(['tuning-*'])

    # Compute accuracy and number of steps for each autotuning run
    target = str(ChargeRegime.ELECTRON_1)
    good_charge = Counter()
    total = Counter()
    nb_steps = Counter()
    for _, row in data.iterrows():
        for tuning_result in row['results.tuning_results']:
            # Only plot Jump procedure
            if not tuning_result['procedure_name'].startswith('Jump'):
                continue

            # name: Model + "Uncertainty"
            procedure_name = tuning_result['model_name'].replace('FeedForward', 'FF')
            procedure_name += '\nUncertainty' if tuning_result['procedure_name'].startswith('JumpUncertainty') else ''
            result_id = (procedure_name, tuning_result['diagram_name'], row['settings.research_group'])

            total[result_id] += 1
            if tuning_result['charge_area'] == target:
                good_charge[result_id] += 1

            nb_steps[result_id] += tuning_result['nb_steps']

    grouped_results = []  # Group by (diagram - procedure - model)
    for result_id in total.keys():
        procedure_name, diagram_name, research_group = result_id
        success_rate = good_charge[result_id] / total[result_id]
        mean_steps = nb_steps[result_id] / total[result_id]
        grouped_results.append([diagram_name, procedure_name, research_group, success_rate, mean_steps])

    # Convert to dataframe
    grouped_results = DataFrame(grouped_results, columns=['diagram', 'procedure_model', 'Datasets', 'success',
                                                          'mean_steps'])

    order = ['FF', 'FF\nUncertainty', 'CNN', 'CNN\nUncertainty', 'BCNN', 'BCNN\nUncertainty']
    # Plot success rate
    sns.barplot(x="procedure_model", y="success", data=grouped_results, ci=None, hue='Datasets', order=order)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.ylabel('Success rate')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures success to reach {target} electron regime')
    # Put the legend out of the figure (middle bottom)
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc='center', ncol=2)
    plt.tight_layout()
    plt.show(block=False)

    # Plot steps
    sns.barplot(x="procedure_model", y="mean_steps", data=grouped_results, ci=None, hue='Datasets', order=order)
    plt.ylabel('Average steps')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures number of steps')
    # Put the legend out of the figure (middle bottom)
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc='center', ncol=2)
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
    data = load_runs_clean(['tuning-louis*', 'tuning-michel*'])

    data.rename(columns={'settings.research_group': 'Datasets',
                         'settings.model_type': 'Model'}, inplace=True)

    for metric in ['Recall', 'Precision', 'Accuracy', 'F1']:
        order = ['FF', 'CNN', 'BCNN']
        metric_mean = data.groupby(['Datasets', 'Model'])[metric].mean()
        ax = sns.barplot(x="Model", y=metric, data=data, hue='Datasets', ci='sd', order=order)

        # Limit min / max value +/- 10%
        ax.set(ylim=(max(metric_mean.min() - 0.1, 0), min(metric_mean.max() + 0.1, 1)))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        # Write values in bars
        for ctr in ax.containers:
            ax.bar_label(ctr, padding=-25, labels=[f'{x:.1%}' for x in ctr.datavalues], fontsize=14, color='w')

        plt.title(f'Trained models {metric} comparison')
        # Put the legend out of the figure (middle bottom)
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc='center', ncol=2)
        plt.tight_layout()
        plt.show(block=False)


def repeat_analyse():
    # Print average result of a repeated run with different seed
    data = load_runs_clean('repeat-batch1024*')

    # Accuracy
    mean_acc = data['Accuracy'].mean()
    std_acc = data['Accuracy'].std()
    print(f'{len(data):n} runs - avg Accuracy: {mean_acc:.2%} (std:{std_acc:.2%})')

    # Main metric
    mean_acc = data[settings.main_metric.capitalize()].mean()
    std_acc = data[settings.main_metric.capitalize()].std()
    print(f'{len(data):n} runs - avg {settings.main_metric.capitalize()}: {mean_acc:.2%} (std:{std_acc:.2%})')

    ct1 = data['results.confidence_thresholds'].map(lambda a: a[0])
    ct2 = data['results.confidence_thresholds'].map(lambda a: a[1])
    print(f'\nWith confidence thresholds: {ct1.mean():.1%} (std: {ct1.std():.1%}) - '
          f'{ct2.mean():.1%} (std: {ct2.std():.1%}) - {data["results.unknown_threshold_rate"].mean():.2%} under CT')

    # Accuracy
    mean_acc = data['ct-Accuracy'].mean()
    std_acc = data['ct-Accuracy'].std()
    print(f'{len(data):n} runs - avg CT Accuracy: {mean_acc:.2%} (std:{std_acc:.2%})')

    # Main metric
    mean_acc = data['ct-' + settings.main_metric.capitalize()].mean()
    std_acc = data['ct-' + settings.main_metric.capitalize()].std()
    print(f'{len(data):n} runs - avg CT {settings.main_metric.capitalize()}: {mean_acc:.2%} (std:{std_acc:.2%})')


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


def uncertainty_analysis():
    data = load_runs_clean(['uncertainty-*-BCNN'])

    # Rename col for auto plot labels
    data.rename(columns={'results.final_tuning_result': 'Tuning success',
                         'settings.confidence_threshold': 'Confidence Threshold'},
                inplace=True)

    def avg_steps(results):
        nb_steps = []
        for tuning_result in results:
            nb_steps.append(tuning_result['nb_steps'])
        return sum(nb_steps) / len(nb_steps)

    data['Average steps'] = data['results.tuning_results'].apply(avg_steps)

    # Tuning success (left axes)
    ax1 = sns.lineplot(data=data, x='Confidence Threshold', y='Tuning success', color='tab:blue')
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('Tuning success', color='tab:blue', fontweight='bold')

    # Average steps (right axes)
    ax2 = plt.twinx()
    ax2.grid(False)
    sns.lineplot(data=data, x='Confidence Threshold', y='Average steps', ax=ax2, color='tab:orange')
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Average steps', color='tab:orange', fontweight='bold')

    plt.title(f'Evolution of the tuning success in function of\n classification confidence threshold')
    plt.tight_layout()
    plt.show(block=False)


def uncertainty_test_noise():
    """
    Plot the evolution of the tuning success in function of the noise level.
    """
    data = load_runs_clean(['uncertainty_test_noise-*'])

    # Rename col for auto plot labels
    data.rename(columns={'settings.test_noise': 'Gaussian noise',
                         'settings.research_group': 'Dataset',
                         'results.unknown_threshold_rate': 'Unknown rate',
                         'settings.model_type': 'Model'}, inplace=True)

    datasets = data['Dataset'].unique()
    datasets.sort()
    nb_datasets = len(datasets)
    plot, axes = plt.subplots(2, nb_datasets, figsize=(5 + nb_datasets * 6, 10), sharex='col', sharey='row')

    # Make a column for each dataset
    for i, dataset in enumerate(datasets):
        score, uncertainty = (axes[0][i], axes[1][i]) if nb_datasets > 1 else (axes[0], axes[1])
        d = data[data['Dataset'] == dataset]
        sns.lineplot(data=d, x='Gaussian noise', y='F1 Uncertainty', hue='Model', ax=score, palette=PALETTE_MODELS,
                     legend=(i == 0))
        sns.lineplot(data=d, x='Gaussian noise', y='F1', hue='Model', linestyle='--', ax=score, palette=PALETTE_MODELS,
                     legend=False)
        sns.lineplot(data=d, x='Gaussian noise', y='Unknown rate', hue='Model', ax=uncertainty, palette=PALETTE_MODELS,
                     legend=False)
        score.set_title(DATASET_NAMES[dataset])

        score.set_ylabel('F1-scores')
        score.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        score.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        uncertainty.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        uncertainty.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

        # Add legend entry for dashed line (no uncertainty score)
        if i == 0:
            handles, _ = score.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color='black', linestyle='--', label='No uncertainty'))
            score.legend(handles=handles)

    plt.suptitle(f'Classification performance in function of test noise')
    plt.tight_layout()
    plt.show(block=False)


def models_result_table():
    data = load_runs_clean(['tuning_cross_valid-42*'])

    # Filter columns
    data = data[['settings.research_group', 'settings.model_type', 'Accuracy', 'ct-Accuracy', 'F1', 'ct-F1',
                 'results.unknown_threshold_rate', 'settings.seed']]

    # Rename columns
    data.rename(columns={'settings.research_group': 'Dataset',
                         'settings.model_type': 'Model',
                         'Accuracy': 'Accuracy',
                         'ct-Accuracy': 'Th. Accuracy',
                         'F1': 'F1',
                         'ct-F1': 'Th. F1',
                         'results.unknown_threshold_rate': 'Unknown rate',
                         'settings.seed': 'Seed'}, inplace=True)

    # Compute classification error difference
    data['acc_err_diff'] = ((1 - data['Accuracy']) - (1 - data['Th. Accuracy'])) / (1 - data['Accuracy'])
    data['f1_err_diff'] = ((1 - data['F1']) - (1 - data['Th. F1'])) / (1 - data['F1'])

    # ================================ Result for each seed (average on diagrams)
    result_by_seed = data.groupby(['Dataset', 'Model', 'Seed']).agg(
        acc=('Accuracy', 'mean'),
        acc_std=('Accuracy', 'std'),
        th_acc=('Th. Accuracy', 'mean'),
        th_acc_std=('Th. Accuracy', 'std'),
        acc_err_diff=('acc_err_diff', 'mean'),
        acc_err_diff_std=('acc_err_diff', 'std'),
        f1=('F1', 'mean'),
        f1_std=('F1', 'std'),
        th_f1=('Th. F1', 'mean'),
        th_f1_std=('Th. F1', 'std'),
        unknown_rate=('Unknown rate', 'mean'),
        unknown_rate_std=('Unknown rate', 'std'),
    )

    # ================================ Result grouped by method (variability by seed)
    results = result_by_seed.groupby(['Dataset', 'Model']).agg(
        acc=('acc', 'mean'),
        acc_std=('acc', 'std'),
        th_acc=('th_acc', 'mean'),
        th_acc_std=('th_acc', 'std'),
        acc_err_diff=('acc_err_diff', 'mean'),
        acc_err_diff_std=('acc_err_diff', 'std'),
        f1=('f1', 'mean'),
        f1_std=('f1', 'std'),
        th_f1=('th_f1', 'mean'),
        th_f1_std=('th_f1', 'std'),
        unknown_rate=('unknown_rate', 'mean'),
        unknown_rate_std=('unknown_rate', 'std'),
    )

    # Sorting rows
    results.sort_values(by=['Dataset', 'Model'], ascending=[False, True], inplace=True)

    print('\n\n----------------------------------------------------\n')
    print('Model results\n')
    print(results.to_string())

    print('\n\n----------------------------------------------------\n')
    print('Average benefit\n')

    # Compute the average accuracy difference with and without the threshold
    acc_diff = (results['th_acc'] - results['acc']).mean()
    acc_diff_std = (results['th_acc'] - results['acc']).std()

    # results['error_diff_all'] = ((1 - results['acc']) - (1 - results['th_acc'])) / (1 - results['acc'])
    error_diff = results['acc_err_diff'].mean()
    error_diff_std = results['acc_err_diff'].std()

    unknown_rate_diff = results['unknown_rate'].mean()
    unknown_rate_diff_std = results['unknown_rate'].std()

    print(f'Average accuracy difference: {acc_diff:.2%} (std: {acc_diff_std:.2%})')
    print(f'Which is {error_diff:.2%} error reduction (std: {error_diff_std:.2%})')
    print(f'Average unknown rate: {unknown_rate_diff:.2%} (std: {unknown_rate_diff_std:.2%})')

    # Remove the 'group by' index and compact rename columns
    results.reset_index(inplace=True)

    # Merge the mean and std in the same column as string
    for metric_name in ['acc', 'th_acc', 'acc_err_diff', 'f1', 'th_f1', 'unknown_rate']:
        results[metric_name] = (results[metric_name].map('{:.1%}'.format) +
                                results[metric_name + '_std'].map(lambda s: ' ±{:.1f}'.format(s * 100)))
        del results[metric_name + '_std']

    results.columns = ['Dataset', 'Model', 'Accuracy', r'Accuracy above\\threshold',
                       r'Error reduction\\using threshold', 'F1', r'F1 above\\threshold', r'Rate below\\threshold']

    # Show the latex version for paper
    print('\n\n----------------------------------------------------\n\n')
    print(tabulate(results, headers='keys', tablefmt='latex', showindex=False))
    # Show the same in pretty table
    print('\n\n----------------------------------------------------\n\n')
    results.columns = [col.replace(r'\\', '\n') for col in results.columns]
    print(tabulate(results, headers='keys', tablefmt='fancy_grid', showindex=False))


def results_table():
    data = load_runs_clean(['tuning_cross_valid-42*'])
    # oracle_baseline = load_runs(['tuning-42*-oracle-*'])
    # data = pd.concat([data, oracle_baseline], ignore_index=True)

    # Make a dataframe with a line per tuning result
    tuning_table = []
    target = str(ChargeRegime.ELECTRON_1)
    for _, row in data.iterrows():
        for tuning_result in row['results.tuning_results']:
            # Only consider Jump procedure
            if not (tuning_result['procedure_name'].startswith('Jump') or
                    tuning_result['procedure_name'] == 'Random Baseline'):
                continue

            # Is a baseline if the model is the oracle or the procedure is random
            if bool(row['settings.autotuning_use_oracle']):
                if tuning_result['procedure_name'] == 'Random Baseline':
                    model_type = 'random'
                    accuracy = f1 = 0
                else:
                    model_type = 'oracle'
                    accuracy = f1 = 1
            # Not a baseline, get the normal values
            else:
                model_type = row['settings.model_type']
                accuracy = row['Accuracy']
                f1 = row['F1']

            tuning_table.append([
                row['settings.research_group'],  # Dataset
                model_type,  # Model
                accuracy,  # Model accuracy
                f1,  # Model F1 score
                'Uncertainty' in tuning_result['procedure_name'],  # Use uncertainty
                tuning_result['diagram_name'],  # Diagram
                tuning_result['nb_steps'],  # Nb scan
                tuning_result['nb_classification_success'],  # Nb good inference
                tuning_result['charge_area'] == target,  # Autotuning success
                row['settings.seed']  # Seed
            ])

    # Convert to dataframe for convenance
    tuning_table = pd.DataFrame(tuning_table,
                                columns=['dataset', 'model', 'model_test_acc', 'model_test_f1', 'use_uncertainty',
                                         'diagram', 'nb_scan', 'nb_good_inference', 'tuning_success', 'seed'])

    # Print the number of runs
    print(f'Number of runs: {len(tuning_table)}')
    # Print the total number of steps
    print(f'Total number of steps: {tuning_table["nb_scan"].sum()}')

    # ================================ Result for each seed and each diagram
    result_by_diagram = tuning_table.groupby(['dataset', 'model', 'use_uncertainty', 'diagram', 'seed']).agg(
        model_test_acc=('model_test_acc', 'mean'),  # The accuracy should be the same for each diagram (same model)
        model_test_f1=('model_test_f1', 'mean'),  # The score should be the same for each diagram (same model)
        nb_scan=('nb_scan', 'sum'),
        mean_scan=('nb_scan', 'mean'),
        nb_good=('nb_good_inference', 'sum'),
        tuning_success=('tuning_success', lambda x: x.sum() / x.count()),
    )
    result_by_diagram['model_tuning_acc'] = result_by_diagram['nb_good'] / result_by_diagram['nb_scan']

    print('\n\n----------------------------------------------------\n')
    print('Result by diagrams\n')
    print(result_by_diagram[['model_test_acc', 'model_test_f1', 'mean_scan', 'tuning_success']].to_string())

    # ================================ Result for each seed (average on diagrams)
    result_by_seed = tuning_table.groupby(['dataset', 'model', 'use_uncertainty', 'seed']).agg(
        model_test_acc=('model_test_acc', 'mean'),
        model_test_acc_std=('model_test_acc', 'std'),
        model_test_f1=('model_test_f1', 'mean'),
        model_test_f1_std=('model_test_f1', 'std'),
        nb_scan=('nb_scan', 'sum'),
        mean_scan=('nb_scan', 'mean'),
        nb_good=('nb_good_inference', 'sum'),
        tuning_success_count=('tuning_success', 'sum'),
        tuning_count=('tuning_success', 'count'),
        tuning_success=('tuning_success', lambda x: x.sum() / x.count()),
    )
    result_by_seed['model_tuning_acc'] = result_by_seed['nb_good'] / result_by_seed['nb_scan']

    print('\n\n----------------------------------------------------\n')
    print('Result by seed\n')
    print(result_by_seed[['model_test_acc', 'model_test_acc_std', 'mean_scan', 'tuning_success']].to_string())

    # ================================ Result grouped by tuning method (variability by seed)
    results = result_by_seed.groupby(['dataset', 'model', 'use_uncertainty']).agg({
        'mean_scan': ['mean', 'std'],  # Number of scans during the tuning
        'model_test_acc': ['mean', 'std'],  # Model accuracy on testset
        'model_test_f1': ['mean', 'std'],  # Model f1 score on testset
        'model_tuning_acc': ['mean', 'std'],  # Model accuracy during the tuning procedure
        'tuning_count': 'sum',  # Number of tuning procedures
        'tuning_success_count': 'sum',  # Number of successful tuning procedures
        'tuning_success': ['mean', 'std']  # Tuning procedure that successfully found the 1-electron regime
    })

    # Sorting row
    results.sort_values(by=['dataset', 'model', 'use_uncertainty'], ascending=[False, True, False], inplace=True)

    print('\n\n----------------------------------------------------\n')
    print('Result by method (variability by seed)\n')
    print(results.to_string())

    # Remove the 'group by' index and compact rename columns
    results.reset_index(inplace=True)
    results.columns = [f'{i}|{j}' if j != '' else f'{i}' for i, j in results.columns]

    # Print the average success rate difference with and without using the uncertainty
    tuning_no_uncertainty_fail_rates = 1 - results[results['use_uncertainty'] == False]['tuning_success|mean'].values
    tuning_uncertainty_fail_rates = 1 - results[results['use_uncertainty'] == True]['tuning_success|mean'].values
    tuning_improvements = ((tuning_no_uncertainty_fail_rates - tuning_uncertainty_fail_rates) /
                           tuning_no_uncertainty_fail_rates)

    print('\n\n----------------------------------------------------')
    print(f"\nAverage improvement rates (global average: {tuning_improvements.mean():.2%})")
    print(', '.join(f'{t:.2%}' for t in tuning_improvements))

    # Print the average difference of number of steps with and without using the uncertainty
    tuning_no_uncertainty_steps = results[results['use_uncertainty'] == False]['mean_scan|mean'].values
    tuning_uncertainty_steps = results[results['use_uncertainty'] == True]['mean_scan|mean'].values
    tuning_steps_diff = tuning_uncertainty_steps - tuning_no_uncertainty_steps
    tuning_steps_rates = tuning_steps_diff / tuning_no_uncertainty_steps

    print(f"\nAverage steps difference (global average: {tuning_steps_diff.mean():.1f})")
    print(', '.join(f'{t:.1f}' for t in tuning_steps_diff))

    print(f"\nAverage step increase (global average: {tuning_steps_rates.mean():.2%})")
    print(', '.join(f'{t:.2%}' for t in tuning_steps_rates))

    # Convert boolean values
    results['use_uncertainty'] = results['use_uncertainty'].map({True: 'Yes', False: 'No'})

    # Merge the mean and std in the same column as string
    for metric_name in ['model_test_acc', 'tuning_success']:
        results[metric_name] = (results[metric_name + '|mean'].map('{:.1%}'.format) +
                                results[metric_name + '|std'].map(lambda s: ' ±{:2.1f}'.format(s * 100)))
        del results[metric_name + '|mean']
        del results[metric_name + '|std']

    # Filter and rename columns
    results = results[['dataset', 'model', 'model_test_acc', 'use_uncertainty', 'mean_scan|mean', 'tuning_success']]
    results.columns = ['Dataset', 'Model', 'Line detection\naccuracy',
                       'Uncertainty-based\nTuning', 'Average steps', 'Tuning success']

    # Show latex version for paper
    print('\n\n----------------------------------------------------\n\n')
    print(tabulate(results, headers='keys', tablefmt='latex', showindex=False, floatfmt='.0f'))

    # Show the same in pretty table
    print('\n\n----------------------------------------------------\n\n')
    print(tabulate(results, headers='keys', tablefmt='fancy_grid', showindex=False, floatfmt='.0f'))


def process_online_experiment_results(offline_diagram: str):
    data = load_runs(['experiment-*'])

    # Load the scan of the diagram used for the online experiment
    diagram = DiagramOffline.load_diagrams(settings.pixel_size,
                                           settings.research_group,
                                           Path('data', 'interpolated_csv.zip'),
                                           Path('data', 'labels.json'),
                                           True,
                                           white_list=[offline_diagram])[0]

    # Make a dataframe with a line per tuning result
    tuning_table = []
    target = ChargeRegime.ELECTRON_1
    for _, row in data.iterrows():
        for tuning_result in row['results.tuning_results']:
            # Infer the number of electrons, based on the offline diagram
            final_x, final_y = tuning_result['final_coord']
            final_charge = diagram.get_charge(final_x, final_y)

            tuning_table.append([
                row['settings.run_name'],
                row['settings.model_type'],  # Model
                'Uncertainty' in tuning_result['procedure_name'],  # Use uncertainty
                tuning_result['nb_steps'],  # Nb scan
                final_charge,  # Charge area
                tuning_result['final_volt_coord'],  # Final coord
                final_charge == target,  # Autotuning success
            ])

    # Convert to dataframe for convenance
    tuning_table = pd.DataFrame(tuning_table,
                                columns=['run_name', 'model', 'use_uncertainty', 'nb_scan', 'charge',
                                         'final_volt_coord', 'tuning_success'])

    # Filter final coordinates by category (model and uncertainty), for result plotting
    final_coords = [
        (
            'CNN',
            'lightskyblue',
            map(tuple, tuning_table[
                (tuning_table['model'] == 'CNN') & (tuning_table['use_uncertainty'] == False)
                ]['final_volt_coord'].values.tolist())
        ),
        (
            'CNN Uncertainty',
            'royalblue',
            map(tuple, tuning_table[(tuning_table['model'] == 'CNN') & (tuning_table['use_uncertainty'] == True)][
                'final_volt_coord'].values.tolist())
        ),
        (
            'BCNN',
            'palegreen',
            map(tuple, tuning_table[(tuning_table['model'] == 'BCNN') & (tuning_table['use_uncertainty'] == False)][
                'final_volt_coord'].values.tolist())
        ),
        (
            'BCNN Uncertainty',
            'limegreen',
            map(tuple, tuning_table[(tuning_table['model'] == 'BCNN') & (tuning_table['use_uncertainty'] == True)][
                'final_volt_coord'].values.tolist())
        ),
    ]

    diagram.plot_results(final_coords)

    tuning_table.sort_values(by=['model', 'use_uncertainty', 'run_name'], ascending=[True, False, True], inplace=True)

    # Full table
    print(tabulate(tuning_table, headers='keys', tablefmt='fancy_grid', showindex=False))

    result_by_method = tuning_table.groupby(['model', 'use_uncertainty']).agg(
        mean_scan=('nb_scan', 'mean'),
        std_scan=('nb_scan', 'std'),
        tuning_success=('tuning_success', lambda x: x.sum() / x.count()),
    )

    # Remove the 'group by' index and compact rename columns
    result_by_method.reset_index(inplace=True)

    # Convert boolean values
    result_by_method['use_uncertainty'] = result_by_method['use_uncertainty'].map({True: 'Yes', False: 'No'})

    # Rename columns
    result_by_method.columns = ['Model', 'Tuning with uncertainty', 'Average steps', 'STD steps', 'Tuning success']

    # Summary table
    print(tabulate(result_by_method, headers='keys', tablefmt='fancy_grid', showindex=False,
                   floatfmt=(None, None, '.0f', '.0f', '.1%')))


if __name__ == '__main__':
    # Set plot style
    set_plot_style()

    results_table()
