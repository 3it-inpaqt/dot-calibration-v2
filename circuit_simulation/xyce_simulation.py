import logging
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
import pandas as pd

from classes.classifier_nn import ClassifierNN
from utils.settings import settings
from utils.output import save_netlist, save_xyce_output, save_xyce_results
from circuit_simulation.generate_netlist import generate_netlist_from_model
from plots.simulation_output import plot_simulation_state_evolution


def run_circuit_simulation(model: ClassifierNN, inputs: List, is_first_run: bool):
    """
    Generate a netlist according to the model size and parameters, then run the circuit simulation with Xyce as a
    subprocess.

    Args:
        model: The model to convert as a physical circuit
        inputs: List of inputs on which we want to do inference (e.g. if we have a 18x18 patch then the list contains
                324 elements)

    Returns:
        A pandas dataframe that represent the measurements defined in the simulation. One column for each
         variable defined in the line ".PRINT" of the netlist.
         The normalised output value before threshold.
         The binary output value before threshold.
    """
    # Generate netlist for circuit simulation
    netlist = generate_netlist_from_model(model, inputs)

    if is_first_run:
        # Save the netlist
        save_netlist(netlist)

    # Create netlist in a temporary file
    netlist_file = NamedTemporaryFile(mode='w', delete=False, prefix='xyce-netlist_', suffix='.CIR')
    netlist_file.write(netlist)
    netlist_file.flush()

    # We assume "Xyce" binary to be in the user $PATH
    # TODO add seed from global config
    bash_command = f'Xyce -randseed 42 -delim , {netlist_file.name}'
    logging.debug(f'Run command: "{bash_command}"')

    try:
        process = subprocess.run(bash_command, cwd='./circuit_simulation', shell=True, check=True, capture_output=True,
                                 encoding='utf-8')

        if is_first_run:
            save_xyce_output(process.stdout)

        # Load output csv file that contains measurements for each variable defined in ".PRINT" section of the netlist
        sim_results = pd.read_csv(netlist_file.name + '.prn', index_col=0, skipfooter=1, engine='python')
        # Add variability numbers if they exist
        if Path(netlist_file.name + '.ES.csv').exists():
            var_results = pd.read_csv(netlist_file.name + '.ES.csv', skipfooter=1, engine='python')
            sim_results = sim_results.merge(var_results, on='TIME')

        # Remove some escape character in the output column names
        sim_results = sim_results.rename(lambda x: x.replace('{', '').replace('}', ''), axis='columns')

        if is_first_run:
            save_xyce_results(sim_results)

        # Detect output pulses
        model_output_before_threshold, model_output = detect_model_output(sim_results)

    except subprocess.CalledProcessError as e:
        # If Xyce process fail, the output is printed and the whole program is stopped by an error.
        print(e.output)
        raise e

    finally:
        # Remove the temporary netlist file
        netlist_file.close()
        Path(netlist_file.name).unlink()
        # Remove the possible generated outputs
        Path(netlist_file.name + '.prn').unlink(missing_ok=True)
        Path(netlist_file.name + '.ES.prn').unlink(missing_ok=True)
        Path(netlist_file.name + '.ES.csv').unlink(missing_ok=True)
        Path(netlist_file.name + '_ensemble.dat').unlink(missing_ok=True)
        Path(netlist_file.name + '.mt0').unlink(missing_ok=True)

    # Create some plots to visualise the results
    # plot_simulation_state_evolution(sim_results)

    return sim_results, model_output_before_threshold, model_output


def detect_model_output(sim_results: pd.DataFrame) -> (float, int):
    """
    Detect the model output before and after the threshold, according to the pulse shape measured during Xyce simulation
    and stored in variables 'V(SUM_H_OUT_001)' and 'V(HIDDEN_ACTIV_OUT_H_001)'.

    Args:
        sim_results: A pandas dataframe that represent the measurements defined in the simulation. One column for each
         variable defined in the line ".PRINT" of the netlist.

    Returns:
        The analog output value before threshold (V)
        The binary output value after threshold (0 or 1)

    """
    output_signal = sim_results['V(SUM_H_OUT__001)']
    output_threshold = sim_results['V(HIDDEN_ACTIV_OUT_H_001)']
    t = sim_results['TIME']

    window_mean = 1e-8
    v_out_threshold = 4.9
    i_min = np.where(t > settings.xyce_init_latency)[0][0]
    # Search the index of the end of this inference
    i_max = len(t) - 1
    if np.max(output_threshold[i_min:i_max]) > v_out_threshold:
        prediction = 1
    else:
        prediction = 0
    i_abs = np.argmax(np.abs(output_signal[i_min:i_max]))
    output = np.mean(output_signal[np.where(t > t[i_min + i_abs] - window_mean)[0][0]:i_abs + i_min])
    return output, prediction
