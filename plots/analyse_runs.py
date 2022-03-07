import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame

from datasets.diagram import ChargeRegime
from utils.output import load_runs, set_plot_style


def compare_autotuning():
    data = load_runs([
        'tmp_1',
        'tmp_2',
        'tmp_3',
        'tmp_4',
        'tmp_5',
    ])

    # Compute accuracy and number of steps for each autotuning run
    grouped_results = []
    target = str(ChargeRegime.ELECTRON_1)
    for _, row in data.iterrows():
        results = row['results.tuning_results']
        procedure = row['settings.autotuning_procedure']
        use_oracle = row['settings.autotuning_use_oracle']
        model = row['settings.model_type']

        if procedure == 'random':
            name = 'random'
        else:
            name = f'{procedure}\n' + ('Oracle' if use_oracle else model)

        for diagram, stats in results.items():
            nb_good_regime = sum(1 for s in stats if s['charge_area'] == target)
            nb_steps = sum(s['nb_steps'] for s in stats)
            nb_tuning = len(stats)
            grouped_results.append([diagram, name, nb_good_regime / nb_tuning, nb_steps / nb_tuning])

    # Convert to dataframe
    grouped_results = DataFrame(grouped_results, columns=['diagram', 'procedure_model', 'success', 'mean_steps'])

    # Order and colors
    order = grouped_results.groupby(['procedure_model'])['success'].mean().reset_index().sort_values('success')
    color_map = {'random': 'lightgrey', '\nCNN': 'tab:blue', '\nBCNN': 'tab:purple', '\nOracle': 'grey'}
    colors = [color for c_filter, color in color_map.items() for procedure in order['procedure_model']
              if c_filter in procedure]

    # Plot accuracy
    sns.barplot(x="procedure_model", y="success", data=grouped_results, order=order['procedure_model'],
                palette=colors, ci='sd')
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.ylabel('Success rate')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures success to reach {target} electron regime')
    plt.tight_layout()
    plt.show(block=False)

    # Plot steps
    sns.barplot(x="procedure_model", y="mean_steps", data=grouped_results, order=order['procedure_model'],
                palette=colors, ci='sd')
    plt.ylabel('Average steps')
    plt.xlabel('Procedure and model')
    plt.title(f'Autotuning procedures number of steps')
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

    layers_size_analyse()
