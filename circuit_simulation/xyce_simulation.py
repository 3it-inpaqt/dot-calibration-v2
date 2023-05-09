import logging
import subprocess
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from classes.classifier_nn import ClassifierNN
from circuit_simulation.generate_netlist import generate_netlist_from_model


def run_circuit_simulation(model: ClassifierNN, inputs):
    """
    Generate a netlist according to the model size and parameters, then run the circuit simulation with Xyce as a
    subprocess.

    Args:
        model: The model to convert as a physical circuit
        inputs_sequences: The binary sequence for each input

    Returns:
        A pandas dataframe that represent the measurements defined in the simulation. One column for each
         variable defined in the line ".PRINT" of the netlist.
         The normalised output value before threshold.
         The binary output value before threshold.
    """
    # Generate netlist for circuit simulation
    netlist = generate_netlist_from_model(model, inputs)

    # if out_path:
    #     # Save the netlist
    #     save_netlist(out_path, netlist)

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

        # if out_path:
        #     save_xyce_output(out_path, process.stdout)

        # Load output csv file that contains measurements for each variable defined in ".PRINT" section of the netlist
        sim_results = pd.read_csv(netlist_file.name + '.prn', index_col=0, skipfooter=1, engine='python')
        # Add variability numbers if they exist
        if Path(netlist_file.name + '.ES.csv').exists():
            var_results = pd.read_csv(netlist_file.name + '.ES.csv', skipfooter=1, engine='python')
            sim_results = sim_results.merge(var_results, on='TIME')

        # Remove some escape character in the output column names
        sim_results = sim_results.rename(lambda x: x.replace('{', '').replace('}', ''), axis='columns')

        # if out_path:
        #     save_xyce_results(out_path, sim_results)

        # Detect output pulses
        model_output_before_threshold, model_output = detect_model_output(sim_results)

    except subprocess.CalledProcessError as e:
        # If Xyce process fail, the output is printed and the whole program is stopped by an error.
        print(e.output)
        raise e

    finally:
        # Remove the temporary netlist file
        Path(netlist_file.name).unlink()
        # Remove the possible generated outputs
        Path(netlist_file.name + '.prn').unlink(missing_ok=True)
        Path(netlist_file.name + '.ES.prn').unlink(missing_ok=True)
        Path(netlist_file.name + '.ES.csv').unlink(missing_ok=True)
        Path(netlist_file.name + '_ensemble.dat').unlink(missing_ok=True)
        Path(netlist_file.name + '.mt0').unlink(missing_ok=True)

    # # Create some plots to visualise the results
    # if out_path:
    #     plot_simulation_state_evolution(sim_results, out_path)

    return sim_results, model_output_before_threshold, model_output


def detect_model_output(sim_results: pd.DataFrame) -> (float, int):
    """
    Detect the model output before and after the threshold, according to the pulse shape measured during Xyce simulation
    and stored in variables 'V(sum_h_out_001)' and 'V(Vout_001)'.

    Args:
        sim_results: A pandas dataframe that represent the measurements defined in the simulation. One column for each
         variable defined in the line ".PRINT" of the netlist.
        xyce_config: Program configuration related to Xyce (run with --help to see the configuration options).

    Returns:
        The analog output value before threshold (V)
        The binary output value after threshold (0 or 1)

    """
    output_signal = sim_results['V(SUM_H_OUT_001)']
    output_threshold = sim_results['V(VOUT_001)']
    t = sim_results['TIME']

    window_mean = 1e-8
    v_out_threshold = 4.9
    inference_duration = xyce_config.pulse_width + xyce_config.resting_time
    nb_inference = round((t.iloc[-1] - xyce_config.init_latency) / inference_duration)
    i_min = np.where(t > xyce_config.init_latency)[0][0]

    list_output = []
    list_thr = []
    for i in range(nb_inference):
        # Search the index of the end of this inference
        search_i_max = np.where(t >= xyce_config.init_latency + (i + 1) * inference_duration)
        i_max = search_i_max[0][0] if len(search_i_max[0]) > 0 else len(t) - 1
        if np.max(output_threshold[i_min:i_max]) > v_out_threshold:
            list_thr.append(1)
        else:
            list_thr.append(0)
        i_abs = np.argmax(np.abs(output_signal[i_min:i_max]))
        list_output.append(
            np.mean(output_signal[np.where(t > t[i_min + i_abs] - window_mean)[0][0]:i_abs + i_min]))
        i_min = i_max
    return list_output[-1], list_thr[-1]


def save_netlist(log_dir_path: Path, netlist: str) -> bool:
    """
    Save the netlist if it is the first one generated during this run.
    Only the input should change in the next netlists.

    Args:
        log_dir_path: The current directory used by Lightning to save the log and the files related to this run.
        netlist: The generated netlist to save as a string.

    Returns:
        True if the netlist has been successfully saved. False if this netlist saving is skipped of failed.
    """
    # Check if the logging directory is valid
    if log_dir_path is not None and log_dir_path.exists():
        netlist_path = log_dir_path / 'netlist.CIR'

        # Save only one example of Netlist per run
        if not netlist_path.exists():
            logging.debug(f'Xyce netlist saved in "{netlist_path}"')
            with open(netlist_path, 'w') as f:
                f.write(netlist)
            return True
        return False
    else:
        logging.warning(f'Impossible to save the Netlist because the log directory is not valid ({log_dir_path})')
        return False


def save_xyce_output(log_dir_path: Path, xyce_output: str) -> bool:
    """
    Save Xyce process output.

    Args:
        log_dir_path: The current directory used by Lightning to save the log and the files related to this run.
        xyce_output: Xyce process output as a string.

    Returns:
        True if the output has been successfully saved. False if output saving is skipped of failed.
    """
    # Check if the logging directory is valid
    if log_dir_path is not None and log_dir_path.exists():
        output_path = log_dir_path / 'xyce_output.txt'

        # Save only one example of output per run
        if not output_path.exists():
            logging.debug(f'Xyce simulation output saved in "{output_path}"')
            with open(output_path, 'w') as f:
                f.write(xyce_output)
            return True
        return False
    else:
        logging.error(f'Impossible to save Xyce output because the log directory is not valid ({log_dir_path})')
        return False


def save_xyce_results(log_dir_path: Path, xyce_results: pd.DataFrame):
    """
    Save Xyce measurements as a CSV file.
    There is one column per variable listed after ".PRINT TRAN" in the Netlist.

    Args:
        log_dir_path: The current directory used by Lightning to save the log and the files related to this run.
        xyce_results: Xyce measurements as a pandas dataframe.

    Returns:
        True if the results has been successfully saved. False if results saving is skipped of failed.
    """
    if log_dir_path is not None and log_dir_path.exists():
        results_path = log_dir_path / 'xyce_results.csv'

        # Save only one example of results per run
        if not results_path.exists():
            logging.debug(f'Xyce simulation results saved in "{results_path}"')
            xyce_results.to_csv(results_path, index=False)
            return True
        return False
    else:
        logging.error(f'Impossible to save Xyce results because the log directory is not valid ({log_dir_path})')
        return False
