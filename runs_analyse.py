import matplotlib.pyplot as plt
import seaborn as sns

from utils.output import load_runs, set_plot_style

# TODO create a notebook to compare several runs
if __name__ == '__main__':
    # Set plot style
    set_plot_style()
    # Load selected runs' files
    data = load_runs('train-size-*')

    # Effect of the number of train point per class
    sns.lineplot(data=data, x='settings.train_point_per_class', y='results.accuracy')
    plt.title('Evolution of the accuracy depending of the size of the dataset')
    plt.xlabel('Train point per class')
    plt.ylabel('Accuracy')
    plt.show(block=False)
