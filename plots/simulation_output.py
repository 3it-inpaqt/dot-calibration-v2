from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from utils.output import save_plot


@dataclass
class SimOut:
    color: str
    label: str
    line_style: str
    y_prefix: str
    polarized: bool = False  # True if this node exist in +/- flavors


NODE_MAP = {
    'I_': SimOut('tab:blue', 'Data input', '-', 'Input'),
    'B_': SimOut('tab:blue', 'Bias input', '--', 'Bias'),
    'TIA_H_OUT_': SimOut('tab:olive', 'Line sum', '-', 'Neuron', polarized=True),
    'HIDDEN_ACTIV_OUT_H': SimOut('tab:purple', 'Activation out', '-', 'Neuron'),
    'SUM_H': SimOut('tab:pink', 'Activation input', '--', 'Neuron'),
    'SUM_H_OUT_': SimOut('tab:orange', 'Activation in', '--', 'Neuron'),
}


def node_v_str(prefix: str, index: int, polarity: bool = True) -> str:
    """
    Create a string that represent the voltage at a specific node.

    Args:
        prefix: The node prefix.
        index: The node index (ignored if the prefix finish by '_' and the index is 0)
        polarity: If the node is polarized, add '+' at the end for True or '-' for False

    Returns:
        The string that represent the voltage at a specific node.
    """
    suffix = ''
    mapped_prefix = prefix[:-4] if prefix[-2].isdigit() else prefix
    if NODE_MAP[mapped_prefix].polarized:
        suffix = '+' if polarity else '-'

    if not prefix.endswith('_') and index == 0:
        # If the prefix doesn't end with '_', we assume there is no index
        return f'V({prefix}{suffix})'
    else:
        return f'V({prefix}{index + 1:03}{suffix})'


def iterate_nodes(prefixes: List[str]) -> Generator[Tuple[str, Optional[bool]], None, None]:
    """
    A generator which return the prefix with each polarity of needed

    Args:
        prefixes: A list of node prefixes

    Returns:
        A tuple generator: (the prefix, the polarity as boolean or None if not applicable)
    """
    for prefix in prefixes:
        mapped_prefix = prefix[:-4] if prefix[-2].isdigit() else prefix
        if NODE_MAP[mapped_prefix].polarized:
            yield prefix, True  # +
            yield prefix, False  # -
        yield prefix, None  # Polarity will be ignored


def plot_digital_vs_analog_outputs(results: pd.DataFrame, log_dir_path: Path, before_threshold: bool = False):
    """
    Plot output of digital and analog model side by side.

    Args:
        results: Dataframe that contains all inference result.
        log_dir_path: The current directory used by Lightning to save the log and the files related to this run.
        before_threshold: If True plot the output before the threshold (far analog) or sigmoid (for digital).
    """

    # Plot on the same figure, with 2 y-axis
    fig, ax1 = plt.subplots()
    sns.lineplot(x='ho', y='digital_output' + ('_before_thr' if before_threshold else ''),
                 color='orange', data=results, ax=ax1)
    ax2 = plt.twinx()
    sns.scatterplot(x='ho', y='sim_output' + ('_before_thr' if before_threshold else ''),
                    color='blue', data=results, ax=ax2)

    # Axes labels and color
    ax1.set_xlabel('Weight Hidden -> Output')
    ax1.set_ylabel('Digital', color='orange', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylabel('Analog simulation (V)', color='blue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(False)

    w_ih = results.iloc[0]['ih']
    input_value = results.iloc[0]['input']
    plt.title(f"Model output{' before threshold' if before_threshold else ''}\n"
              f"W Input -> Hidden = {w_ih} and Input = {input_value}")

    save_plot(log_dir_path, 'output_' + 'before' if before_threshold else 'after' + '_threshold')


# noinspection PyUnboundLocalVariable
def plot_v_evolution(results: pd.DataFrame, node_prefixes: List[str], title: str, file_name: str,
                     max_subplot: int = 20, mode: str = 'merge'):
    """
    Plot the electric tension evolution for one node of the simulation.
    Note: I would like to apologize for this over-complicated plot function.

    Args:
        results: A list af result to plot.
        log_dir_path: The current directory used by Lightning to save the log and the files related to this run.
        node_prefixes: The name prefix of the node to plot (before the index). Should match with the node name in the
            netlist.
        title: The title of the plot.
        file_name: The name of the file.
        max_subplot: The maximum amount of subplots to show.
        mode: The type of layout:
            * merge: Every node measurements with the same index are show on the subplot (1 column, nb row = nb index)
            * split: Every node measurements with the same index are show side by side on different subplots
                (2 columns, nb row = nb index)
            * concat: Every node measurements are show on a new subplot (1 column, nb row = nb index * nb node)
    """
    # Count the number of node with each prefix within the subplot limit
    nb = dict()
    for prefix in node_prefixes:
        nb[prefix] = sum(1 for i in range(max_subplot) if node_v_str(prefix, i) in results)
    nb_total = sum(nb.values())

    if nb_total == 0:
        return  # Nothing to plot

    if mode == 'split' and len(node_prefixes) == 1:
        mode = 'merge'

    if mode == 'merge':
        nb_row = max(nb.values())
        nb_col = 1
    elif mode == 'concat':
        nb_row = nb_total
        nb_col = 1
    elif mode == 'split':
        nb_row = max(nb.values())
        nb_col = len(node_prefixes)
    else:
        raise ValueError(f'Invalid plot mode: "{mode}"')

    # Count the max number to know how many we hide
    nb_max = copy(nb)
    for prefix in node_prefixes:
        while True:
            if node_v_str(prefix, nb_max[prefix]) not in results:
                break
            nb_max[prefix] += 1

    # noinspection PyTypeChecker
    fig, axes = plt.subplots(nb_row, nb_col, sharex=True, figsize=(8 + (nb_col * 2), 2 + (2 * nb_row)))

    # Iterate rows
    for i_row in range(nb_row):
        # Iterate columns
        for i_col in range(nb_col):
            # Select the axis to use
            if nb_row == 1 and nb_col == 1:
                ax = axes
            elif mode == 'merge' or mode == 'concat':
                ax = axes[i_row]
            elif mode == 'split':
                if nb_row > 1:
                    ax = axes[i_row][i_col]
                else:
                    ax = axes[i_col]

            # Select data to plot on this axis
            if mode == 'merge':
                current_prefixes = node_prefixes
                index = i_row
            elif mode == 'concat':
                index = i_row
                current_pre_i = 0
                # Select the good prefix and index for this row
                for count in nb.values():
                    if index - count < 0:
                        break
                    index -= count
                    current_pre_i += 1
                current_prefixes = [node_prefixes[current_pre_i]]
            elif mode == 'split':
                current_prefixes = [node_prefixes[i_col]]
                index = i_row

            # Iterate node to show for each subplot
            for prefix, polarity in iterate_nodes(current_prefixes):
                mapped_prefix = prefix[:-4] if prefix[-2].isdigit() else prefix
                node_str = node_v_str(prefix, index, polarity)
                polarity_str = '' if polarity is None else ('+' if polarity else '–')
                show_label = ((len(current_prefixes) > 1 and mode == 'merge') or polarity is not None) and i_row == 0
                linestyle = NODE_MAP[mapped_prefix].line_style
                if polarity is False:
                    linestyle = 'dotted'
                sns.lineplot(data=results, x='TIME', y=node_str,
                             color=NODE_MAP[mapped_prefix].color,
                             linestyle=linestyle,
                             label=NODE_MAP[mapped_prefix].label + polarity_str if show_label else None,
                             ax=ax)
                if i_col == 0:
                    ax.set_ylabel(f'{NODE_MAP[mapped_prefix].y_prefix} {index + 1}' if len(node_prefixes) > 1
                                  else f'{index + 1}{polarity_str}')
                else:
                    ax.set_ylabel(None)  # y-label only on the first column, because it should always be the same

            ax.set_xlabel(None)  # Global x-label only
            # Make sure that the y scale is not too small
            bottom, top = ax.get_ylim()
            ax.set_ylim(min(bottom, -0.05), max(top, 0.05))

    # Build a fancy global y-label
    global_y_labels = set()
    for label, value in nb.items():
        mapped_prefix = label[:-4] if label[-2].isdigit() else label
        node_prefix = NODE_MAP[mapped_prefix].y_prefix
        y_label = str(value) if value == nb_max[label] else f'{value}/{nb_max[label]}'
        y_label += ' ' + node_prefix + ('s' if value > 1 and node_prefix[-1] != 's' else '')
        global_y_labels.add(y_label)

    if len(global_y_labels) == 1 and nb == nb_max:
        fig.supylabel('(V)')
    else:
        fig.supylabel(' - '.join(global_y_labels) + ' (V)')
    fig.supxlabel('Simulation time (s)')
    fig.suptitle(title, fontsize=14)

    save_plot(file_name)


def plot_simulation_state_evolution(results: pd.DataFrame, nb_layers: int):
    """
    Plot the evolution of physical variables measured during the simulation.

    Args:
        results: A list af result to plot.
    """

    plot_v_evolution(results, ['I_', 'B_'], 'Inputs pulses', 'sim_inputs', mode='concat')
    for i in range(nb_layers):
        plot_v_evolution(results, [F'SUM_H_OUT_{i+1:03}_', F'HIDDEN_ACTIV_OUT_H{i+1:03}_'],
                         'Signal before and after activation', 'sim_activation')
    for i in range(nb_layers):
        plot_v_evolution(results, [f'TIA_H_OUT_{i+1:03}_', f'SUM_H_OUT_{i+1:03}_'],
                         'Difference at the input of the output layer', 'sim_diff_output', mode='split')
