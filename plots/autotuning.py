import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import DataFrame

from utils.output import save_plot
from utils.settings import settings


def plot_autotuning_results(results, overall) -> None:
    # Convert to dataframe to please seaborn
    df = DataFrame(results[1:], columns=results[0])
    # average step
    ave_step = np.array([int(i[1]) for i in results[1:]])
    ave_step = ave_step.mean()
    ave_line_success = np.array([int(i[2]) for i in results[1:]])
    ave_line_success = ave_line_success.mean()
    # Remove useless columns
    df = df.loc[:, ['Diagram'] + [str(r) for r in area_legend()]]
    # One row per regime and diagram
    df = df.melt(id_vars=['Diagram'], var_name='Number of electrons', value_name='Quantity')
    # Plot
    plot = sns.barplot(x='Number of electrons', y='Quantity', data=df, saturation=.7,
                       palette=['grey', 'tab:red', 'tab:green', 'tab:red', 'tab:red', 'tab:red'])
    plot.set_title(f'Final dot regime after autotuning.\nAverage steps: {ave_step:.1f} '
                   f'({ave_line_success:.1%} line detection success)')

    save_plot(f'autotuning_results')


def plot_autotuning_results_NDots(results) -> None:
    # Convert to dataframe to please seaborn
    df_log = DataFrame(results[1:], columns=results[0])

    # average step
    ave_step = np.array([int(i[1]) for i in results[1:]])
    ave_step = ave_step.mean()
    ave_line_success = np.array([int(i[2]) for i in results[1:]])
    ave_line_success = ave_line_success.mean()

    df_log = df_log.loc[:, ['Diagram'] + area_legend()]
    # One row per regime and diagram
    df_log = df_log.melt(id_vars=['Diagram'], var_name='Number of electrons', value_name='Quantity')
    # Plot
    palette = ['grey' if tag == 'UNKNOWN' else 'tab:green' if tag == '(1, 1)' else 'tab:red' for tag in
               results[0][6:]]

    plot = sns.barplot(x='Number of electrons', y='Quantity', data=df_log, saturation=.7, palette=palette)
    plot.set_title(f'Final dot regime after autotuning.\nAverage steps: {ave_step:.1f} '
                   f'({ave_line_success:.1%} line detection success)')

    save_plot(f'autotuning_results')

    for result in results[1:]:
        ### table ###
        data, legend = decompose_axis([results[0], result])
        columns = legend[1]
        rows = legend[0]
        # print('data = ', data, '\ncolumns = ', columns, '\nline = ', rows)
        plot, ax = plt.subplots(1, 1)
        ax.set_title(f'Final dot regime after autotuning.\nAverage steps: {results[1][1]} '
                     f'({results[1][2]:.1%} line detection success), Success: {results[1][5]:.1%}%')
        ax.axis('tight')
        ax.axis('off')
        df = pd.DataFrame(data, columns=columns, index=rows)
        df = df.astype(int)
        row_colors = ["grey"] * len(df.index)
        col_colors = ["grey"] * len(df.columns)
        cell_width = 1.0 / (len(df.columns) + 1)
        cell_height = 1.0 / (len(df.index) + 1)

        table = ax.table(cellText=df.values,
                         colLabels=df.columns,
                         rowLabels=df.index,
                         rowColours=row_colors,
                         colColours=col_colors,
                         loc="center",
                         cellLoc="center",
                         cellColours=[[plt.cm.Greys(0.3)] * len(df.columns)] * len(df.index))
        for key, cell in table.get_celld().items():
            cell.set_height(cell_height)
            cell.set_width(cell_width)

        save_plot(f'autotuning_results-table_{result[0]}')

def decompose_axis(result):
    data = result[1][6:]
    legend = result[0][6:]
    legend_axis = [['0', '1', '2', '3+', 'Unknown'],
                   ['0', '1', '2', '3+']]
    data_tab = [[0] * (len(legend_axis[1])) for _ in range(len(legend_axis[0]) - 1)]
    for k, tag in enumerate(legend):
        if data[k] == '-' or data[k] == '_':
            continue
        elif tag == 'UNKNOWN':
            unk = [int(data[k])] * len(legend_axis[1])
            data_tab.append(unk)
        else:
            i = int(tag[1])
            j = int(tag[4]) if tag[4] != ' ' else int(tag[5])
            val = int(data[k])
            data_tab[i][j] = val
    return data_tab, legend_axis


def area_legend():
    if settings.dot_number == 1:
        return ['UNKNOWN', '0', '1', '2', '3', '4+']
    else:
        import itertools
        charge_areas = ["UNKNOWN"]
        regimes = ["0", "1", "2", "3+"]
        couples = tuple(itertools.product(regimes, repeat=settings.dot_number))
        for couple in couples:
            charge_areas.append(couple)
        return [str(x).replace("'", "") for x in charge_areas]


def corresponding_legend(result: tuple):
    legend = [1] * settings.dot_number
    for i, area in enumerate(result):
        if area == "unknown":
            return "UNKNOWN"
        else:
            if int(area[0]) >= 3:
                legend[i] = "3+"
            else:
                legend[i] = area[0]
    return str(tuple(legend)).replace("'", "")
