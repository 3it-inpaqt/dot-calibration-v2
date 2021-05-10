import matplotlib.pyplot as plt
import seaborn as sns

from utils.output import load_runs, set_plot_style


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

    repeat_analyse()
