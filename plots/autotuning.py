import seaborn as sns
from pandas import DataFrame

from datasets.diagram_offline import ChargeRegime
from utils.output import save_plot


def plot_autotuning_results(results, overall) -> None:
    # Convert to dataframe to please seaborn
    df = DataFrame(results[1:], columns=results[0])
    # Compute steps stats
    avg_steps = df['Steps'].mean()
    model_success_rate = overall['good'] / overall['steps'] if overall['steps'] > 0 else 0
    # Remove useless columns
    df = df.loc[:, ['Diagram'] + [str(r) for r in ChargeRegime]]
    # One row per regime and diagram
    df = df.melt(id_vars=['Diagram'], var_name='Number of electrons', value_name='Quantity')
    # Plot
    plot = sns.barplot(hue='Number of electrons', y='Quantity', data=df, saturation=.7, legend=False,
                       palette=['grey', 'tab:red', 'tab:green', 'tab:red', 'tab:red', 'tab:red'])
    plot.set_title(f'Final dot regime after autotuning.\nAverage steps: {avg_steps:.1f} '
                   f'({model_success_rate:.1%} line detection success)')

    save_plot(f'autotuning_results')
