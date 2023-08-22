import logging
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
import pandas as pd
import torch

from classes.classifier import Classifier
from classes.classifier_nn import ClassifierNN
from utils.settings import settings
from utils.output import save_netlist, save_xyce_output, save_xyce_results
from circuit_simulation.generate_netlist import NetlistGenerator
from plots.simulation_output import plot_simulation_state_evolution


class CircuitSimulator(Classifier):

    def __init__(self, network : ClassifierNN):
        self.network = network
        self.netlist_generator = NetlistGenerator(network)
        self.confidence_thresholds = network.confidence_thresholds

    def infer(self, patches, nb_samples = 0) -> (List[bool], List[float]):
        """
        Simulate circuit inferences on a set of input patches.

        :param patches: The patches to classify.
        :param nb_sample: Not used here, just added for compatibility with Bayesian models.
        :return: The class inferred by this method and the confidence it this result (between 0 and 1).
        """
        outputs_before_thr = []
        predictions = []
        for patch in patches:
            inputs = torch.flatten(patch).tolist()
            if settings.use_xyce:
                sim_results, sim_output_before_thr, sim_output = self.run_xyce_simulation(inputs, False)
            else:
                sim_results, sim_output_before_thr, sim_output = self.run_ltspice_simulation(inputs, False)
            outputs_before_thr.append(sim_output_before_thr)
            predictions.append(sim_output)

        predictions = torch.tensor(predictions)
        outputs_before_thr = torch.tensor(outputs_before_thr)

        # Use sigmoid to convert the output into probability
        sigmoid_outputs = torch.sigmoid(outputs_before_thr/settings.sim_pulse_amplitude)

        # We assume that a value far from 0 or 1 mean low confidence (e.g. output:0.25 => class 0 with 50% confidence)
        confidences = torch.abs(0.5 - sigmoid_outputs) * 2

        return predictions, confidences


    def run_xyce_simulation(self, inputs: List, is_first_run: bool):
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
        netlist, nb_layers = self.netlist_generator.generate_netlist_from_model(inputs, is_first_run)

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
            model_output_before_threshold, model_output = self.detect_model_output(sim_results, nb_layers)

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
        if is_first_run:
            plot_simulation_state_evolution(sim_results, nb_layers)

        return sim_results, model_output_before_threshold, model_output


    def run_ltspice_simulation(self, inputs: List, is_first_run: bool):
        """
        Generate a netlist according to the model size and parameters, then run the circuit simulation with LTspice as a
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
        netlist, nb_layers = self.netlist_generator.generate_netlist_from_model(inputs, is_first_run)

        if is_first_run:
            # Save the netlist in the output directory
            save_netlist(netlist)
        # Save the netlist in the circuit simulation directory
        save_path = Path('./circuit_simulation', 'netlist.CIR')
        with open(save_path, 'w') as f:
            f.write(netlist)

        runcmd = f'{settings.ltspice_executable_path} -b -ascii netlist.CIR'
        logging.debug(f'Run command: "{runcmd}"')

        try:
            subprocess.run(runcmd, cwd='./circuit_simulation', check=True, capture_output=True,
                                     encoding='utf-8')
            sim_results = self.parse_raw_file(".\\circuit_simulation\\netlist")
            if is_first_run:
                save_xyce_results(sim_results)

            # Detect output pulses
            model_output_before_threshold, model_output = self.detect_model_output(sim_results, nb_layers)

        except subprocess.CalledProcessError as e:
            # If LTspice process fail, the output is printed and the whole program is stopped by an error.
            print(e.output)
            raise e

        # Create some plots to visualise the results
        if is_first_run:
            plot_simulation_state_evolution(sim_results, nb_layers)

        return sim_results, model_output_before_threshold, model_output


    @staticmethod
    def parse_raw_file(file_path):
        """
        Parse the raw data file created by LTspice and create a pandas DataFrame from it.

        The function reads the raw data file, extracts the header information, variable names,
        and data values, and returns a pandas DataFrame with the extracted data.

        Parameters:
            file_path (str): The path to the raw data file (without the '.raw' extension).

        Returns:
            pandas.DataFrame: A DataFrame containing the parsed data from the raw file.
                              The columns are named after the extracted variable names, and
                              the rows represent the data points for each variable.

        Raises:
            IOError: If the specified raw data file is not found.
        """
        reading_header = True
        reading_variables = False
        data = []
        variables = []
        data_line = []
        try:
            f = open(file_path + '.raw', 'r')
            for line_num, line in enumerate(f):

                if reading_header:
                    if line_num == 4:
                        number_of_vars = int(line.split(' ')[-1])
                    if line_num == 5:
                        number_of_points = int(line.split(' ')[-1])
                    if line[:10] == 'Variables:':
                        reading_header = False
                        reading_variables = True
                        continue
                elif reading_variables:
                    if line[:7] == 'Values:':
                        reading_variables = False
                        header_length = line_num + 1
                        continue
                    else:
                        variable_name = line.split('\t')[-2]
                        variables.append(variable_name.upper())
                else:
                    data_line_num = (line_num - header_length) % number_of_vars
                    data_line.append(line.split('\t')[-1].split('\n')[0])
                    if data_line_num == number_of_vars - 1:
                        data.append(data_line)
                        data_line = []

            f.close()
        except IOError:
            print('File not found: ' + file_path + '.raw')

        sim_results = pd.DataFrame(data=data, columns=variables)
        return sim_results.applymap(float)

    @staticmethod
    def detect_model_output(sim_results: pd.DataFrame, nb_layers: int) -> (float, int):
        """
        Detect the model output before and after the threshold, according to the pulse shape measured during Xyce simulation
        and stored in variables 'V(SUM_H_OUT_001)' and 'V(HIDDEN_ACTIV_OUT_H_001)'.

        Args:
            sim_results: A pandas dataframe that represent the measurements defined in the simulation. One column for each
             variable defined in the line ".PRINT" of the netlist.
            nb_layers: nb of hidden layers in the model

        Returns:
            The analog output value before threshold (V)
            The binary output value after threshold (0 or 1)

        """
        output_signal = sim_results[f'V(SUM_H_OUT_{nb_layers:03}_001)']
        output_threshold = sim_results[f'V(HIDDEN_ACTIV_OUT_H{nb_layers:03}_001)']
        t = sim_results['TIME']

        t_min_v_out = nb_layers * settings.sim_layer_latency + settings.sim_init_latency + settings.sim_pulse_rise_delay
        # Make sure that we let the chance to the current to stabilize (that's why we add 8e-8)
        t_min_v_out = t_min_v_out + 8e-8
        # Make sure we don't overshoot the time window width (that's why we subtract 1.2e-7)
        window_width = settings.sim_pulse_width - settings.sim_pulse_rise_delay - settings.sim_pulse_fall_delay - 1.2e-7
        t_max_v_out = t_min_v_out + window_width
        i_min = np.where(t > t_min_v_out)[0][0]
        i_max = np.where(t > t_max_v_out)[0][0]

        output = output_signal[i_min:i_max].min()
        v_out_threshold = 0
        if output > v_out_threshold:
            prediction = 1
        else:
            prediction = 0
        return output, prediction
