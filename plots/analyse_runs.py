import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from classes.diagram import ChargeRegime
from utils.output import load_runs, set_plot_style


def compare_autotuning():
    data = load_runs([
        'tuning_michel_random',
        'tuning_michel_czischek_oracle',
        'tuning_michel_czischek_cnn',
        'tuning_michel_czischek_bcnn',
        'tuning_michel_czischek_bcnn',
        'tuning_michel_bczischek_cnn',
        'tuning_michel_bczischek_bcnn',
    ])

    # Compute accuracy for each autotuning run
    scores = {}
    target = str(ChargeRegime.ELECTRON_1)
    for _, row in data.iterrows():
        results = row['results.final_regimes']
        procedure = row['settings.autotuning_procedure']
        use_oracle = row['settings.autotuning_use_oracle']
        model = row['settings.model_type']

        nb_good = 0
        nb_total = 0
        for diagram, stats in results.items():
            nb_good += stats[target] if target in stats else 0
            nb_total += sum(stats.values())

        if procedure == 'random':
            name = 'random'
        else:
            name = f'{procedure}\n' + ('Oracle' if use_oracle else model)

        scores[name] = nb_good / nb_total if nb_total > 0 else 0

    # Sort by accuracy
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))
    color_map = {'random': 'lightgrey', '\nCNN': 'tab:blue', '\nBCNN': 'tab:purple', '\nOracle': 'grey'}
    colors = [color for c_filter, color in color_map.items() for procedure in scores.keys() if c_filter in procedure]

    plt.bar(range(len(scores)), scores.values(), tick_label=list(scores.keys()), color=colors)
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    plt.ylabel('Accuracy')
    plt.title(f'Autotuning procedures success to reach {target} electron regime')
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
    data = load_runs('layers_size*')

    # print(data['settings.hidden_layers_size'])

    # Hidden size -> Accuracy
    plt.axhline(y=data['results.baseline_std_test_accuracy'][0], label='STD Baseline', color='r')
    sns.lineplot(data=data, x='network_info.total_params', y='results.final_accuracy', label='Feed Forward')
    plt.title('Evolution of the accuracy in function of number of parameters')
    plt.xlabel('Total number of parameters')
    plt.ylabel('Classification accuracy')

    # plt.xlim(right=10_000)
    # plt.xlim(left=0)

    plt.show(block=False)


def patch_size_analyse():
    # Load selected runs' files
    data = load_runs('patch_size_cnn*')

    # Patch size -> Accuracy
    sns.lineplot(data=data, x='settings.patch_size_x', y='results.final_accuracy', label='CNN')
    sns.lineplot(data=data, x='settings.patch_size_x', y='results.baseline_std_test_accuracy', label='STD baseline')
    plt.title('Evolution of the accuracy in function of patch size')
    plt.xlabel('Patch size in pixel')
    plt.ylabel('Classification accuracy')
    plt.show(block=False)

    # Patch size -> Train size
    uniq_seed = data.drop_duplicates(subset='settings.patch_size_x').copy()
    uniq_seed['total_train'] = uniq_seed['results.train_dataset_size'] + uniq_seed['results.train_dataset_augmentation']
    sns.lineplot(data=uniq_seed, x='settings.patch_size_x', y='total_train')
    plt.title('Size of training dataset in function of patch size')
    plt.xlabel('Patch size in pixel')
    plt.ylabel('Number of training patch')
    plt.show(block=False)


if __name__ == '__main__':
    # Set plot style
    set_plot_style()

    compare_autotuning()
