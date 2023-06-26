import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import List, Tuple, Union

import numpy as np
from jinja2 import Environment, FileSystemLoader

from classes.classifier_nn import ClassifierNN
from utils.settings import settings
from plots.parameters import plot_resistance_distribution


def convert_param_to_resistances(param: float, r_min: int, r_max: int) -> (int, int):
    """
    Convert normalised model parameter to a couple of resistance values (R+, R-).
    This is an idealist conversion algorithm with no consideration for hardware non-idealities.
    It maximises the total resistance value.

    Args:
        param: The model parameter (weight or bias) to convert.
        r_max: The maximal memristor resistance value to consider for the parameter mapping.
        r_min: The minimal memristor resistance value to consider for the parameter mapping.

    Returns:
        A couple of resistance values (R+, R-) equivalent to the input parameter. In Ohm, rounded as integer.
    """
    if param > 0:
        return round(1 / (1 / r_max + param * (1 / r_min - 1 / r_max))), r_max
    elif param < 0:
        return r_max, round(1 / (1 / r_max - param * (1 / r_min - 1 / r_max)))
    else:  # weight == 0
        return r_max, r_max


def compute_gain_matching(parameters_max_value: float) -> (float, float):
    """
    Compute the gain of transimpedance amplifiers (TIAs) and differential amplifiers to match the output amplitude.

    Args:
        parameters_max_value: The maximum absolute parameter in the neural network.

    Returns:
        The two gains for the TIA and the differential amplifier (A_tia, A_diff) in Ohm.
    """
    # Computing the conductance associated with a logical 1
    g_log_1 = (1 / settings.xyce_r_min + (parameters_max_value - 1) * 1 / settings.xyce_r_max) / parameters_max_value
    total_gain = 1 / (g_log_1 - 1 / settings.xyce_r_max)
    if total_gain == 7500:
        a_tia, a_diff = 2500, 780  # Set the optimal gain configuration for the OPA684/MAX4223 OpAmps
    else:
        a_tia = total_gain / 2.5
        if a_tia > 6000:  # Upper bound for the TIA gain, higher gain leads to an insufficient slew rate
            a_tia = 6000
        elif a_tia < 1000:  # Lower bound for the TIA gain, lower gain leads to oscillations
            a_tia = 1000
        a_diff = total_gain / a_tia * 260
        if a_diff > 975 or a_diff < 260:  # Lower and upper bound for the differential amplifier gain
            logging.warning('The gain matching is not optimal for current resistance range and parameter values.')
    return a_tia, a_diff


def convert_inputs_to_pulses(inputs: Union[List[List[float]], List[float]],
                             offset: float = 0) -> List[List[Tuple[float, float]]]:
    """
    Convert multiple inputs sequences into multiple pulse trains.

    Args:
        inputs: The flattened inputs to convert
        offset: A delay for each time in the sequence (s)

    Returns:
        A list of pulse train. Each pulse train is defined by a list of pair (simulation time (s), voltage (V))
        Dimension: Input size (D) × Nb tension change × 2 (simulation time (s), voltage (V))
    """
    if settings.xyce_pulse_fall_delay + settings.xyce_pulse_rise_delay >= settings.xyce_pulse_width:
        raise ValueError('The sum of pulse delays is longer than the pulse width.')

    pulses = []
    for input_value in inputs:
        time = offset + settings.xyce_init_latency
        pulse = [(0, 0)]  # Start at 0V
        pulse.extend(
            [
                # Start pulse
                (time, 0),
                (time + settings.xyce_pulse_rise_delay, input_value * settings.xyce_pulse_amplitude),
                # End pulse
                (time + settings.xyce_pulse_width - settings.xyce_pulse_fall_delay,
                 input_value * settings.xyce_pulse_amplitude),
                (time + settings.xyce_pulse_width, 0)
            ])

        pulses.append(pulse)

    return pulses


def generate_netlist_from_model(model: ClassifierNN, inputs: List, is_first_run: bool) -> tuple[str, int]:
    """
    Convert a neural network model to its equivalent analog circuit netlist.
    The netlist is formatted to fit Xyce (https://xyce.sandia.gov/) syntax.
    The model is currently restricted to 2 layers feed forward neural networks with binary inputs and one binary output.
    The size of the input and the hidden layer is flexible.

    Args:
        model: The PyTorch neural network model to convert as a netlist.
        inputs: The inputs for the inference task, to convert as pulses

    Returns:
        The Xyce netlist as a string.
    """
    # Get the absolute max values of the parameters
    parameters_max = max(layer.data.abs().max() for layer in model.parameters()).item()

    layers = defaultdict(list)
    nb_layers = 0
    # Iterate parameters layer by layer
    for layer_name, layer_parameters in model.named_parameters():
        # Normalise parameters between -1 and 1
        layer_parameters = layer_parameters / parameters_max

        for neuron_params in layer_parameters.tolist():
            # Weight conversion
            if 'weight' in layer_name:
                neuron_weights = []
                for weight in neuron_params:
                    neuron_weights.append(convert_param_to_resistances(weight, settings.xyce_r_min, settings.xyce_r_max))
                layers['weight_' + str(nb_layers)].append(neuron_weights)  # Add neuron to the list
            # Bias conversion
            elif 'bias' in layer_name:
                layers['bias_' + str(nb_layers - 1)].append(
                    convert_param_to_resistances(neuron_params, settings.xyce_r_min, settings.xyce_r_max))

        if 'weight' in layer_name:
            nb_layers += 1

    # Convert input pulses
    inputs_pulses = convert_inputs_to_pulses(inputs)
    # Create bias pulse train (will be used only if model have bias enable)
    bias_pulses = {}
    for i in range(nb_layers):
        if layers['bias_' + str(i)] is not None:
            bias_pulses['bias_' + str(i)] = convert_inputs_to_pulses([1], offset=i * settings.xyce_layer_latency)[0]

    # xyce_matrix_mul(layers, inputs_pulses, bias_pulses['bias_0'])

    # Compute gain matching for TIA and sum according to the parameters max value and the physical configuration
    gain_tia, gain_sum = compute_gain_matching(parameters_max)

    # Total simulation duration
    simulation_duration = settings.xyce_init_latency + settings.xyce_pulse_width \
                          + settings.xyce_layer_latency * nb_layers

    # Compute the clipping voltage for the Hard Tanh
    # Logical value hardcoded to 1 for now
    tanh_upper_bound = 1 * settings.xyce_pulse_amplitude - 0.6

    # Plot a histogram of the resistances
    if is_first_run:
        plot_resistance_distribution(layers, 'Resistance values of the memristors\nrepresenting the model parameters',
                                     'parameters_resistances')

    # Creat Jinja template environment and process the netlist with model layers
    environment = Environment(loader=FileSystemLoader("./circuit_simulation"))
    # Formatting functions
    environment.filters["i"] = lambda x: f'{x:03}'  # Index formatting as "001"
    environment.filters["s"] = lambda x: f'{x * 1e9:.2f}ns'  # Time formatting from second to nanosecond
    environment.filters["v"] = lambda x: f'{x:.3f}V'  # Voltage formatting
    environment.filters["c"] = lambda x: f'{x * 1e12:.3f}p'  # Capacitance formatting
    environment.filters["l"] = lambda x: f'{x * 1e9:.3f}n'  # Inductance formatting

    netlist = environment.get_template("netlist_template.CIR").render(
        layers=layers,
        nb_layers=nb_layers,
        pulses_sequences=inputs_pulses,
        bias_pulses=bias_pulses,
        tanh_upper_bound=tanh_upper_bound,
        step_size=settings.xyce_step_size,
        simulation_duration=simulation_duration,
        read_std=settings.xyce_memristor_read_std,
        var_sample_size=settings.xyce_var_sample_size,
        gain_tia=gain_tia,
        gain_sum=gain_sum
    )

    return netlist, nb_layers

def xyce_matrix_mul(layers, input_pulses, bias_pulse):
    # Generate the matrix of resistances for layer 1
    layer_1_R = []
    for neuron in layers['weight_0']:
        # we put one line for + resistances and one line for - resistances
        for polarity in range(2):
            neuron_weights = []
            for weight in neuron:
                neuron_weights.append(weight[polarity])
            layer_1_R.append(neuron_weights)

    # Since each row represents a neuron we add the biases at the end of each row.
    for i,neuron_bias in enumerate(layers['bias_0']):
        for polarity in range(2):
            layer_1_R[2*i + polarity].append(neuron_bias[polarity])

    layer_1_R = np.array(layer_1_R)
    # Have each column represent a neuron
    layer_1_R = layer_1_R.T

    # Calculate conductance
    layer_1_G = 1 / layer_1_R

    time = 0
    while time != 5:
        # Create the matrix of voltages
        layer_1_V = []
        # Start with the inputs
        nb_inputs = len(input_pulses)
        for pulse_nb in range(nb_inputs):
            wire = []
            for neuron in layer_1_R[0]:
                # append the voltage at time time
                if pulse_nb != nb_inputs:
                    wire.append(input_pulses[pulse_nb][time][1] - input_pulses[pulse_nb][time][1])
                else:
                    wire.append(input_pulses[pulse_nb][time][1] - bias_pulse[time][1])
            layer_1_V.append(wire)

        # Do the same for the bias
        wire = []
        for neuron in layer_1_R[0]:
            # append the voltage at time time
            wire.append(bias_pulse[time][1])
        layer_1_V.append(wire)

        # Transpose the voltage matrix
        layer_1_V = np.array(layer_1_V)
        layer_1_V = layer_1_V.T

        # Calculate the current with I = V*G
        layer_1_I = np.matmul(layer_1_V, layer_1_G)
        I_det = np.linalg.det(layer_1_I)
        print(f'the det is ' + str(I_det))

        print(time)
        time = time + 1