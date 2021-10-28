import seaborn as sns
from pandas import DataFrame

from utils.output import save_plot


def plot_autotuning_results(results) -> None:
    # Convert to dataframe to please seaborn
    df = DataFrame(results[1:], columns=results[0])
    # Remove useless columns
    df = df.iloc[:, :7]
    # One row per regime and diagram
    df = df.melt(id_vars=["Diagram"], var_name="Number of electrons", value_name="Quantity")
    # Plot
    sns.barplot(x='Number of electrons', y='Quantity', data=df,
                palette=['grey', 'tab:red', 'tab:green', 'tab:red', 'tab:red', 'tab:red'])

    save_plot(f'autotuning_results')
