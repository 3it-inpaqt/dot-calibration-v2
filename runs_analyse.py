import matplotlib.pyplot as plt
import seaborn as sns

from utils.output import load_runs, set_plot_style

if __name__ == '__main__':
    # Set plot style
    set_plot_style()

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
