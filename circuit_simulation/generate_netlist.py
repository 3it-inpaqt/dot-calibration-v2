import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from typing import List, Tuple, Union

from jinja2 import Environment, FileSystemLoader

from classes.classifier_nn import ClassifierNN


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


def compute_gain_matching(xyce_config: Namespace, parameters_max_value: float) -> (float, float):
    """
    Compute the gain of transimpedance amplifiers (TIAs) and differential amplifiers to match the output amplitude.

    Args:
        xyce_config: Program configuration related to Xyce (run with --help to see the configuration options).
        parameters_max_value: The maximum absolute parameter in the neural network.

    Returns:
        The two gains for the TIA and the differential amplifier (A_tia, A_diff) in Ohm.
    """
    # Computing the conductance associated with a logical 1
    g_log_1 = (1 / xyce_config.r_min + (parameters_max_value - 1) * 1 / xyce_config.r_max) / parameters_max_value
    total_gain = 1 / (g_log_1 - 1 / xyce_config.r_max)
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


def convert_inputs_to_pulses(inputs_sequences: Union[List[List[float]], List[float]],
                             xyce_config: Namespace, offset: float = 0) -> List[List[Tuple[float, float]]]:
    """
    Convert multiple inputs sequences into multiple pulse trains.

    Args:
        inputs_sequences: The binary inputs sequences to convert (dimension: Sequence length × Input size)
        xyce_config: Program configuration related to Xyce (run with --help to see the configuration options).
        offset: A delay for each time in the sequence (s)

    Returns:
        A list of pulse train. Each pulse train is defined by a list of pair (simulation time (s), voltage (V))
        Dimension: Input size (D) × Nb tension change × 2 (simulation time (s), voltage (V))
    """
    if xyce_config.pulse_fall_delay + xyce_config.pulse_rise_delay >= xyce_config.pulse_width:
        raise ValueError('The sum of pulse delays is longer than the pulse width.')

    # Turn scalar value into 1 element list if necessary
    inputs_sequences = inputs_sequences if isinstance(inputs_sequences, Iterable) else [inputs_sequences]

    if isinstance(inputs_sequences[0], Iterable):
        # Transpose the 2d array to have (Input size × Sequence length)
        inputs_sequences = [*zip(*inputs_sequences)]

    pulses_sequences = []
    inference_duration = xyce_config.pulse_width + xyce_config.resting_time
    for inputs_sequence in inputs_sequences:
        pulses = [(0, 0)]  # Start at 0V
        # Turn scalar value into 1 element list if necessary
        inputs_sequence = inputs_sequence if isinstance(inputs_sequence, Iterable) else [inputs_sequence]
        for i, input_value in enumerate(inputs_sequence):
            time = offset + xyce_config.init_latency + (i * inference_duration)
            # We have a pulse only if the input is 1
            if input_value == 1:
                pulses.extend(
                    [
                        # Start pulse
                        (time, 0),
                        (time + xyce_config.pulse_rise_delay, xyce_config.pulse_amplitude),
                        # End pulse
                        (time + xyce_config.pulse_width - xyce_config.pulse_fall_delay, xyce_config.pulse_amplitude),
                        (time + xyce_config.pulse_width, 0)
                    ])

        # Add a final 0V point at the end of the sequence
        pulses.append((xyce_config.init_latency + (len(inputs_sequence) * inference_duration), 0))
        pulses_sequences.append(pulses)

    return pulses_sequences


def generate_netlist_from_model(model: ClassifierNN, inputs) -> str:
    """
    Convert a neural network model to its equivalent analog circuit netlist.
    The netlist is formatted to fit Xyce (https://xyce.sandia.gov/) syntax.
    The model is currently restricted to 2 layers feed forward neural networks with binary inputs and one binary output.
    The size of the input and the hidden layer is flexible.

    Args:
        xyce_config:
        model: The PyTorch neural network model to convert as a netlist.
        inputs_sequences: The binary sequence for each input, to convert as pulses

    Returns:
        The Xyce netlist as a string.
    """
    # Get the absolute max values of the parameters
    parameters_max = max(layer.data.abs().max() for layer in model.parameters()).item()

    layers = defaultdict(list)
    # Iterate parameters layer by layer
    for layer_name, layer_parameters in model.get_clean_params():
        # Normalise parameters between -1 and 1
        layer_parameters = layer_parameters / parameters_max

        for neuron_params in layer_parameters.tolist():
            # Weight conversion
            if layer_name.startswith('weight'):
                neuron_weights = []
                for weight in neuron_params:
                    neuron_weights.append(convert_param_to_resistances(weight, xyce_config.r_min, xyce_config.r_max))
                layers[layer_name].append(neuron_weights)  # Add neuron to the list
            # Bias conversion
            elif layer_name.startswith('bias'):
                layers[layer_name].append(
                    convert_param_to_resistances(neuron_params, xyce_config.r_min, xyce_config.r_max))

    sequence_length = len(inputs_sequences)
    if isinstance(model, DecoderFF):
        # Flatten the sequence to send everything in one time since we don't have recurrence (for demo only)
        inputs_sequences = [item for sublist in inputs_sequences for item in sublist]
        sequence_length = 1

    model_input_size = len(layers['weight_ih'][0])
    # Convert input pulses
    pulses_sequences = convert_inputs_to_pulses(inputs_sequences, xyce_config)
    # Create bias pulse train (will be used only if model have bias enable)
    bias_pulses = {
        'bias_ih': convert_inputs_to_pulses([[1]] * sequence_length, xyce_config)[0],
        # Add a time offset for the second bias pulses to compensate the first layer latency
        'bias_ho': convert_inputs_to_pulses([[1]] * sequence_length, xyce_config, offset=xyce_config.relu_latency)[0]
    }

    # Compute gain matching for TIA and sum according to the parameters max value and the physical configuration
    gain_tia, gain_sum = compute_gain_matching(xyce_config, parameters_max)

    # Check dimensions
    input_size = len(pulses_sequences)
    if model_input_size != input_size:
        raise ValueError(f"The model input size ({model_input_size}) doesn't fit "
                         f"with the pulse input size ({input_size})")

    # Take the last time of the pulse sequence as the total simulation duration
    simulation_duration, _ = pulses_sequences[0][-1]

    # Compute the clipping voltage for the Hard Tanh
    # Logical value hardcoded to 1 for now
    tanh_upper_bound = 1 * xyce_config.pulse_amplitude - 0.6

    # Compute physical values for LC circuit
    lc_capacitance = (xyce_config.pulse_width + xyce_config.resting_time - xyce_config.relu_latency) / (50 * 120)
    lc_inductance = (xyce_config.pulse_width + xyce_config.resting_time - xyce_config.relu_latency) * 50 / 120

    # Plot some stuff
    plot_resistance_distribution(layers, 'Resistance values of the memristors\nrepresenting the model parameters',
                                 model.get_log_dir_path(), 'parameters_resistances')

    # Creat Jinji template environment and process the netlist with model layers
    environment = Environment(loader=FileSystemLoader("./circuit_simulation"))
    # Formatting functions
    environment.filters["i"] = lambda x: f'{x:03}'  # Index formatting as "001"
    environment.filters["s"] = lambda x: f'{x * 1e9:.2f}ns'  # Time formatting from second to nanosecond
    environment.filters["v"] = lambda x: f'{x:.3f}V'  # Voltage formatting
    environment.filters["c"] = lambda x: f'{x * 1e12:.3f}p'  # Capacitance formatting
    environment.filters["l"] = lambda x: f'{x * 1e9:.3f}n'  # Inductance formatting
    return environment.get_template("netlist_template.CIR").render(
        layers=layers,
        pulses_sequences=pulses_sequences,
        bias_pulses=bias_pulses,
        threshold=xyce_config.pulse_amplitude * 0.5,  # threshold_analog = pulse_amplitude * threshold_digital
        non_linearity=model.hidden_activation.analog_subcircuit_name(),
        tanh_upper_bound=tanh_upper_bound,
        step_size=xyce_config.step_size,
        simulation_duration=simulation_duration,
        read_std=xyce_config.memristor_read_std,
        var_sample_size=xyce_config.var_sample_size,
        recurrence_lc_capacitance=lc_capacitance,
        recurrence_lc_inductance=lc_inductance,
        gain_tia=gain_tia,
        gain_sum=gain_sum
    )
