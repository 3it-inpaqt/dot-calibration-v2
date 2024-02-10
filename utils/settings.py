import argparse
from ast import List
import re
from dataclasses import asdict, dataclass
from math import isnan
from typing import Sequence, Union

import configargparse
from numpy.distutils.misc_util import is_sequence

from utils.logger import logger


@dataclass(init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default path: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)
    """

    # ==================================================================================================================
    # ==================================================== General =====================================================
    # ==================================================================================================================

    # Name of the run to save the result ('tmp' for temporary files).
    # If empty or None thing is saved.
    run_name: str = ''

    # The seed to use for all random number generators during this run.
    # Forcing reproducibility could lead to a performance lost.
    seed: int = 42

    # If true every baseline are run before the real training. If false this step is skipped.
    evaluate_baselines: bool = False

    # The metric to use for plotting, logging and model performance evaluation.
    # See https://yonvictor.notion.site/Classification-Metrics-2074032f927847c0885918eb9ddc508c
    # Possible values: 'precision', 'recall', 'f1'.
    main_metric: str = 'f1'

    # ==================================================================================================================
    # ============================================== Logging and Outputs ===============================================
    # ==================================================================================================================

    # The minimal logging level to show in the console (see https://docs.python.org/3/library/logging.html#levels).
    logger_console_level: Union[str, int] = 'INFO'

    # The minimal logging level to write in the log file (see https://docs.python.org/3/library/logging.html#levels).
    logger_file_level: Union[str, int] = 'DEBUG'

    # If True, a log file is created for each run with a valid run_name.
    # The console logger could be enabled at the same time.
    # If False, the logging will only be in the console.
    logger_file_enable: bool = True

    # If True uses a visual progress bar in the console during training and loading.
    # Should be used with a logger_console_level as INFO or more for better output.
    visual_progress_bar: bool = True

    # If True, add color for pretty console output.
    # Should be disabled on Windows.
    console_color: bool = True

    # If True, show matplotlib images when they are ready.
    show_images: bool = False

    # If True and the run have a valid name, save matplotlib images in the run directory
    save_images: bool = True

    # If True and the run have a valid name, save the experimental patches measured in the run directory
    save_measurements: bool = True

    # If True, plot the measurement. It is then saved or shown depending on the other settings.
    plot_measurements: bool = True

    # If True and the run have a valid name, save animated GIF in the run directory
    save_gif: bool = False

    # If True and the run have a valid name, save video in the run directory
    save_video: bool = False

    # If True and the run have a valid name, save the neural network parameters in the run directory at the end of the
    # training. Saved before applying early stopping if enabled.
    # The file will be at the root of run directory, under then name: "final_network.pt"
    save_network: bool = True

    # If True, the diagrams will be plotted when they are loaded.
    # Always skipped if patches are loaded from cache.
    plot_diagrams: bool = True

    # If True, images are also saved in a format adapted to latex format (vectorial and transparent background).
    image_latex_format: bool = False

    # The number of results to plot for each procedure.
    nb_plot_tuning_fail: int = 5
    nb_plot_tuning_success: int = 2

    # Push phone notification for important updates if the topic is not empty.
    # Send notification for: end of training, end of online tuning
    # See https://docs.ntfy.sh/ for more information.
    ntfy_topic: str = ''

    # The ntfy server to use for push notification (public official server by default).
    ntfy_server: str = 'https://ntfy.sh/'

    # ==================================================================================================================
    # ==================================================== Dataset =====================================================
    # ==================================================================================================================

    # If true, the data will be loaded from the cache if possible (maybe create bug with some settings).
    use_data_cache: bool = False

    # The sizes of a diagram patch send to the network input (number of pixels)
    patch_size_x: int = 18
    patch_size_y: int = 18

    # The patch overlapping (number of pixels)
    patch_overlap_x: int = 10
    patch_overlap_y: int = 10

    # The width of the border to ignore during the patch labeling (number of pixels)
    # E.g.: If one line touch only 1 pixel at the right of the patch and the label_offset_x is >1 then the patch will be
    # labeled as "no_line"
    label_offset_x: int = 6
    label_offset_y: int = 6

    # The size of the interpolated pixel in Volt.
    # Should be available in the dataset folder.
    pixel_size: float = 0.001

    # If True, the parasitic lines will be considered as a line labels.
    # Otherwise, the labels of parasitic line will be ignored.
    load_parasitic_lines: bool = False

    # The name of the research group who provide the data.
    # currently: 'louis_gaudreau' or 'michel_pioro_ladriere' or 'eva_dupont_ferrier' or 'stefanie_czischek'
    # Should be available in the dataset folder.
    research_group: str = 'michel_pioro_ladriere'

    # The percentage of data kept for testing only.
    # If test_diagram is set, this value should be 0.
    test_ratio: float = 0.2

    # The base name (no extension) of the diagram file to use as test for the line and tuning task.
    # To use for cross-validation.
    # If test_ratio != 0, this value should be empty string.
    test_diagram: str = ''

    # The percentage of data kept for validation only. Set tot 0 to disable validation.
    validation_ratio: float = 0.2

    # If True, data augmentation methods will be applied to increase the size of the train dataset.
    train_data_augmentation: bool = True

    # The number of data loader workers, to take advantage of multithreading. Always disable with CUDA.
    # 0 means automatic setting (using cpu count).
    nb_loader_workers: int = 0

    # If True, the training dataset is balanced using weighted random sampling.
    # see https://github.com/ufoym/imbalanced-dataset-sampler
    balance_class_sampling: bool = True

    # The normalization method to use.
    # Could be: None, 'patch', 'train-set'.
    # See ../documentation/normalization.drawio.svg for more information.
    normalization: str = 'patch'

    # The path to yaml file containing the normalization values (min and max).
    # Use to consistant normalization of the data after the training.
    normalization_values_path: str = ''

    # The percentage of gaussian noise to add in the test set.
    # Used to test uncertainty.
    test_noise: float = 0.0

    # Whether an exponentially weighted moving average (EWMA) method should be used to preprocess the patches.
    # See this paper: Moras, M. (2023). Outils d’identification du régime à un électron pour les boîtes quantiques
    # semiconductrices.
    use_ewma = False

    # This is the r parameter in the following equation: Z_i = (1 - r) * Z_(i-1) + r * X_i
    ewma_parameter = 0.15

    # If True, we take the absolute value after subtracting the EWMA from the derivative of the patch and ewma_threshold
    # is not used.
    is_ewma_with_abs = True

    # This is the k used in the EWMA method when we look if a value is outside this range: mean +/- k * sigma
    ewma_threshold = 3

    # ==================================================================================================================
    # ===================================================== Model ======================================================
    # ==================================================================================================================

    # The type of model to use (could be a neural network).
    # Have to be in the implemented list: FF, BFF, CNN, BCNN.
    model_type: str = 'CNN'

    # The number of fully connected hidden layers and their respective number of neurons.
    hidden_layers_size: Sequence = (200, 100)

    # Whether there should be a bias in the hidden layer or not (currently only implemented in FF and CNN)
    bias_in_hidden_layer = True

    # The number of convolution layers and their respective properties (for CNN models only).
    conv_layers_kernel: Sequence = (4, 4)
    conv_layers_channel: Sequence = (12, 24)

    # Define if there is a max pooling layer after each convolution layer (True = max pooling)
    # Have to match the convolution layers size
    max_pooling_layers: Sequence = (False, False)

    # Define if there is a batch normalization layer after each layer (True = batch normalization)
    # Have to match the number of layers (convolution + linear)
    batch_norm_layers: Sequence = (False, False, False, False)

    # ==================================================================================================================
    # ==================================================== Training ====================================================
    # ==================================================================================================================

    # If a valid path to a file containing neural network parameters is set, they will be loaded in the current neural
    # network and the training step will be skipped.
    trained_network_cache_path: str = ''

    # The pytorch device to use for training and testing. Can be 'cpu', 'cuda' or 'auto'.
    # The automatic setting will use CUDA is a compatible hardware is detected.
    device: str = 'auto'

    # The learning rate value used by the SGD for parameters update.
    learning_rate: float = 0.001

    # Dropout rate for every dropout layer defined in networks.
    # If a network model doesn't have a dropout layer, this setting will have no effect.
    # Set to 0 to skip dropout layers
    dropout: int = 0.4

    # Whether dropconnect should be used during training or not. This is only implemented for FF models right now.
    use_dropconnect = False

    # If use_dropconnect is True, specifies the probability of setting a weight to 0 during the training. Should be
    # between 0 and 1
    dropconnect_prob = 0.1

    # The size of the mini-batch for the training and testing.
    batch_size: int = 512

    # The number of training epochs.
    # Can't be set as the same time as nb_train_update, since it indirectly defines nb_epoch.
    # Set to 0 to disable (nb_train_update must be > 0)
    nb_epoch: int = 0

    # The number of updates before to stop the training.
    # This is just a convenant way to define the number of epochs with variable batch and dataset size.
    # The final value will be a multiple of the batch number in 1 epoch (rounded to the higher number of epochs).
    # Can't be set as the same time as nb_epoch, since it indirectly defines it.
    # Set to 0 to disable (nb_epoch must me > 0)
    nb_train_update: int = 10_000

    # Save the best network state during the training based on the test main metric.
    # Then load it when the training is complete.
    # The file will be at the root of run directory, under the name: "best_network.pt"
    # Required checkpoints_per_epoch > 0 and checkpoint_validation = True
    early_stopping: bool = True

    # Threshold to consider the model inference good enough.
    # Under this limit, we consider that we don't know the answer.
    # Negative threshold means automatic value selection using tau.
    confidence_threshold: float = -1.0

    # Relative importance of model error compares to model uncertainty for automatic confidence threshold tuning.
    # The Confidence threshold is optimized by minimizing the following score: nb error + (nb unknown * tau)
    # Used only if the confidence threshold is not defined (<0)
    auto_confidence_threshold_tau: float = 0.2

    # The number of samples used to compute the loss of bayesian networks.
    bayesian_nb_sample_train: int = 3

    # The number of samples used to compute model inference during the validation.
    bayesian_nb_sample_valid: int = 3

    # The number of samples used to compute model inference during the testing.
    bayesian_nb_sample_test: int = 10

    # The metric to use to compute the model inference confidence.
    # Should be in: 'std', 'norm_std', 'entropy', 'norm_entropy'
    bayesian_confidence_metric: str = 'norm_std'

    # The weight of complexity cost part when computing the loss of bayesian networks.
    bayesian_complexity_cost_weight: float = 1 / 50_000

    # ==================================================================================================================
    # ================================================== Checkpoints ===================================================
    # ==================================================================================================================

    # The number of checkpoints per training epoch.
    # Can be combined with updates_per_checkpoints.
    # Set to 0 to disable.
    checkpoints_per_epoch: int = 0

    # The number of model updates (back propagation) before to start a checkpoint.
    # Can be combined with checkpoints_per_epoch.
    # Set to 0 to disable.
    checkpoints_after_updates: int = 200

    # The number of data in the checkpoint training subset.
    # Set to 0 to don't compute the train metrics during checkpoints.
    checkpoint_train_size: int = 640

    # If the inference metrics of the validation dataset should be computed, or not, during checkpoint.
    # The validation ratio has to be higher than 0.
    checkpoint_validation: bool = True

    # If the inference metrics of the testing dataset should be computed, or not, during checkpoint.
    checkpoint_test: bool = False

    # If True and the run have a valid name, save the neural network parameters in the run directory at each checkpoint.
    checkpoint_save_network: bool = False

    # ==================================================================================================================
    # =================================================== Autotuning ===================================================
    # ==================================================================================================================

    # List of autotuning procedure names to use.
    # Have to be in the implemented list: random, shift, shift_u, jump, jump_u, full, sanity_check
    autotuning_procedures: Sequence = ('jump_u',)

    # If True, the line classification model cheat by using the diagram labels (no neural network loaded).
    # Used for baselines.
    autotuning_use_oracle: bool = False

    # If True, the tuning algorithm will try to detect the transition line slope. If not, will always use the prior
    # slope knowledge.
    # This feature is only available on jump algorithms.
    auto_detect_slope: bool = True

    # If True, the Jump algorithm will validate the leftmost line at different Y-position to avoid mistake in the case
    # of fading lines.
    validate_left_line: bool = True

    # If the oracle is enabled, these numbers corrupt its precision.
    # 0.0 = all predictions are based on the ground truth (labels), means 100% precision
    # 0.5 = half of the predictions are random, means 75% precision for binary classification.
    # 1.0 = all the predictions are random, means 50% precision for binary classification.
    # TODO: implement that (and maybe create an Oracle class)
    autotuning_oracle_line_random: float = 0
    autotuning_oracle_no_line_random: float = 0

    # Number of iterations per diagram for the autotuning test.
    # For the 'full' procedure, this number is override to 1.
    autotuning_nb_iteration: int = 50

    # ==================================================================================================================
    # ==================================================== Connector ===================================================
    # ==================================================================================================================

    # The name if the connector to use to capture online diagrams.
    # Possible values: 'mock', 'py_hegel'
    connector_name: str = 'mock'

    # The automation level of the connector.
    # 'auto': the connector will automatically send the command to the measurement device.
    # 'semi-auto': the connector will show the command to the user before to send it to the measurement device.
    # 'manual': the connector will only show the command to the user, and will not send it to the measurement device.
    interaction_mode: str = 'semi-auto'

    # The maximum and minimum voltage that we can request from the connector.
    # This needs to be explicitly defined before to tune an online diagram with a connector.
    range_voltage_x: Sequence = (float('nan'), float('nan'))
    range_voltage_y: Sequence = (float('nan'), float('nan'))

    # The voltage range in which we can choose a random starting point, for each gate.
    start_range_voltage_x: Sequence = (float('nan'), float('nan'))
    start_range_voltage_y: Sequence = (float('nan'), float('nan'))

    # ==================================================================================================================
    # ============================================= Circuit Simulation =================================================
    # ==================================================================================================================

    # Whether the inference of the ML model should be simulated on a circuit or not.
    simulate_circuit: bool = False

    # If simulate_circuit is True, then should the simulation be done with Xyce?
    # Only one simulation engine should be selected.
    use_xyce: bool = True

    # If simulate_circuit is True, then should the simulation be done with LTspice?
    # Only one simulation engine should be selected.
    use_ltspice: bool = False

    # File path of the LTspice program installed on the system
    ltspice_executable_path: str = ''


    ########## Coded by HA ##########
    # The name of the circuit to be used in simulations
    # It is important to set this parameter
    # The best practice is to keep its default value  
    LTSpice_asc_filetype : str = 'asc'
    LTspice_working_directory: str = 'data/ltspice/'
    LTspice_out_directory: str = LTspice_working_directory + 'spice/'
    LTspice_data_directory: str = LTspice_working_directory + 'data/'
    LTspice_wd_prefix: str = 'activ_v'   # Specifies the name prefix for the WD
    LTSpice_asc_filename: str = LTspice_out_directory + 'complete_circuit.asc'   # Specifies the circuit file name
    # required components directory
    LTspice_required_directory: str = 'ltspice/spice/components/requiredSubCells'
    LTspice_avail_activation_fn = ['column_activ_cmos', 'column_activ_comparator', 'column_activ_relu', 'column_activ_tia_relu', 'column_activ_cmos_v2', 'column_activ_cmos_v3']
    LTspice_activation_fn_select: int = 5
    LTspice_final_activation_fn_select: int = 5 # # Specifies the AF for the final stage
    # Layers parameters 
    LTspice_num_of_layers: int = 3
    LTspice_layer_dims = [[65,20], [21,10],[11,1]] # Sizes with biases 
    # Setting the minimum vertical spacing between blocks in the circuit
    LTspice_block_vspacing: int = 480 # should be multiple of 16
    # The measurements to be saved from the simulation. The appropriate numbers can be found in the .raw file generated at simulation.
    LTspice_variable_numbering = {'time': 0, 'final_out': 45}
    # The ordering of the measurements to be saved in the output csv files can be changed below, by changing how the numbers are ordered.
    # E.g. switch place of 0 and 1 if you want V_c to be placed left of time in the output csv files.
    LTspice_preffered_sorting = [0, 1]
    # Leave blank if output should be writtten to root folder. The folder specified must be created manually if it doesn't exist.
    LTspice_output_data_path = 'ltspice/simulation_results/'
    # Naming convention for output files. Can be numerical names, increasing in value as output is created.
    # Parameter names gives the files the name of the parameter that it is run with.
    # Assumes number by default. Set to 'parameter' or 'number'.
    LTspice_output_data_naming_convention = 'parameter'
    #################################

    # Whether we should run the inference over the test-set with the simulated circuit.
    # Warning: the simulations can be long, you might want to avoid testing the circuits sometimes.
    test_circuit: bool = True

    # Maximum number of test inferences to simulate on the circuit (0 means the whole test set).
    sim_max_test_inference: int = 1000

    # If set and greater than 0, the model's parameters will be clipped between
    # [-parameters_clipping, parameters_clipping] after each training batch. Set to None if no parameter clipping
    # should be used
    # TODO: Move this setting to the training section and make it non-restrictive to the circuit simulation
    parameters_clipping: float = 2

    # If set to True, the training will take into account that memristors may be blocked with
    # xyce_memristor_blocked_prob.
    # TODO: Currently only implemented for FF NNs.
    hardware_aware_training: bool = False

    # The simulation step size for transient analysis (s)
    sim_step_size: float = 3e-10

    # The minimal resistance value that we consider for memristor programming (ohm)
    sim_r_min: int = 5000

    # The maximal resistance value that we consider for memristor programming (ohm)
    sim_r_max: int = 15000

    # The read standard deviation of the memristor resistance (% [0,1])
    # Should be around 0.2%
    # TODO: Verify that this is correctly implemented before using it
    sim_memristor_read_std: float = 0.0

    # The number of sample for the read variability study.
    # This setting have no effect if memristor_read_std is 0.
    # TODO: Verify that this is correctly implemented before using it
    sim_var_sample_size: int = 0

    # The write standard deviation of the memristor resistance (% [0,1])
    # Should be around 0.8%
    sim_memristor_write_std: float = 0.008

    # The probability that a memristor will be blocked to r_max when we try to write a value to it (% [0,1])
    ratio_failure_HRS: float = 0.05

    # The probability that a memristor will be blocked to r_min when we try to write a value to it (% [0,1])
    # ratio_failure_HRS + ratio_failure_LRS should be around 10%
    ratio_failure_LRS: float = 0.05

    # The pulse amplitude for the input encoding (V)
    sim_pulse_amplitude: float = 0.2

    # The pulse duration for the input encoding (s)
    sim_pulse_width: float = 3e-7

    # The resting time after a pulse (s)
    sim_resting_time: float = 5e-8

    # Pulse delay from 0V to pulse_amplitude (s)
    sim_pulse_rise_delay: float = 1e-9

    # Pulse delay from pulse_amplitude to 0V (s)
    sim_pulse_fall_delay: float = 1e-9

    # Simulation initial latency before to start the first pulse (s)
    sim_init_latency: float = 1e-9

    # The estimated delay between the input and the output of the activation function analog bloc.
    # Used to synchronize the pulse between layers.
    sim_layer_latency: float = 1e-8

    # Number of parallel processes to run (0 means the number of cpu cores)
    sim_nb_process: int = 1

    # Voltage OFFSET added to the pulses
    sim_voltage_offset: float = 0.95

    def is_named_run(self) -> bool:
        """ Return True only if the name of the run is set (could be a temporary name). """
        return len(self.run_name) > 0

    def is_unnamed_run(self) -> bool:
        """ Return True only if the name of the run is NOT set. """
        return len(self.run_name) == 0

    def is_temporary_run(self) -> bool:
        """ Return True only if the name of the run is set and is temporary name. """
        return self.run_name == 'tmp'

    def is_saved_run(self) -> bool:
        """ Return True only if the name of the run is set and is NOT temporary name. """
        return self.is_named_run() and not self.is_temporary_run()

    def validate(self):
        """
        Validate settings.
        """

        # General
        assert self.run_name is None or not re.search('[/:"*?<>|\\\\]+', self.run_name), \
            'Invalid character in run name (should be a valid directory name)'
        assert self.main_metric in ['precision', 'recall', 'f1'], f'Unknown metric "{self.main_metric}"'

        # Logging and Outputs
        possible_log_levels = ('CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')
        assert self.logger_console_level.upper() in possible_log_levels or isinstance(self.logger_console_level, int), \
            f"Invalid console log level '{self.logger_console_level}'"
        assert self.logger_file_level.upper() in possible_log_levels or isinstance(self.logger_file_level, int), \
            f"Invalid file log level '{self.logger_file_level}'"

        # Dataset
        assert self.research_group in ['louis_gaudreau', 'michel_pioro_ladriere', 'eva_dupont_ferrier',
                                       'stefanie_czischek'], f'Unknown dataset research group: "{self.research_group}"'
        assert self.patch_size_x > 0, 'Patch size should be higher than 0'
        assert self.patch_size_y > 0, 'Patch size should be higher than 0'
        assert self.patch_overlap_x >= 0, 'Patch overlapping should be 0 or more'
        assert self.patch_overlap_y >= 0, 'Patch overlapping should be 0 or more'
        assert self.patch_overlap_x < self.patch_size_x, 'Patch overlapping should be lower than the patch size'
        assert self.patch_overlap_y < self.patch_size_y, 'Patch overlapping should be lower than the patch size'
        assert self.label_offset_x < (self.patch_size_x // 2), 'Label offset should be lower than patch size // 2'
        assert self.label_offset_y < (self.patch_size_y // 2), 'Label offset should be lower than patch size // 2'
        assert self.test_ratio > 0 or len(self.test_diagram) > 0, 'Test data ratio or test diagram should be set'
        assert not (self.test_ratio > 0 and len(self.test_diagram) > 0), 'Only one between "test ratio" and ' \
                                                                         '"test diagram" should be set'
        assert self.test_ratio + self.validation_ratio < 1, 'test_ratio + validation_ratio should be less than 1 to' \
                                                            ' have training data'
        assert self.normalization in [None, 'patch', 'train-set'], f'Unknown normalization method: {self.normalization}'

        # Networks
        assert isinstance(self.model_type, str) and self.model_type.upper() in ['FF', 'BFF', 'CNN', 'BCNN'], \
            f'Invalid network type {self.model_type}'
        assert all((a > 0 for a in self.hidden_layers_size)), 'Hidden layer size should be more than 0'
        assert len(self.conv_layers_channel) == len(self.conv_layers_kernel) == len(self.max_pooling_layers), \
            'All convolution meta parameters should have the same size (channel, kernels and max pooling)'
        if self.model_type.upper() in ['CNN', 'BCNN']:
            assert len(self.conv_layers_channel) + len(self.hidden_layers_size) == len(self.batch_norm_layers), \
                'The batch normalisation meta parameters should be define for each layer (convolution and linear)'
        if self.model_type.upper() in ['FF', 'BFF']:
            assert len(self.hidden_layers_size) == len(self.batch_norm_layers), \
                'The batch normalisation meta parameters should be define for each linear layer'
        assert all((a > 0 for a in self.conv_layers_channel)), 'Conv layer nb channel should be more than 0'
        assert all((a > 1 for a in self.conv_layers_kernel)), 'Conv layer kernel size should be more than 1'

        # Training
        # TODO should also accept "cuda:1" format
        assert self.device in ('auto', 'cpu', 'cuda'), f'Not valid torch device name: {self.device}'
        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0 or self.nb_train_update > 0, 'Number of epoch or number of train step ' \
                                                              'should be at least 1'
        assert not (self.nb_epoch > 0 and self.nb_train_update > 0), 'Exactly one should be set between' \
                                                                     ' number of epoch and number of train step'
        assert self.bayesian_nb_sample_train > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_nb_sample_valid > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_nb_sample_test > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_confidence_metric in ['std', 'norm_std', 'entropy', 'norm_entropy'], \
            f'Invalid bayesian confidence metric value "{self.bayesian_confidence_metric}"'
        if self.use_dropconnect:
            assert 0 < self.dropconnect_prob < 1, 'The probability used for dropconnect should be between 0 and 1.'

        # Checkpoints
        assert self.checkpoints_per_epoch >= 0, 'The number of checkpoints per epoch should be >= 0'
        assert self.checkpoints_after_updates >= 0, 'The number of updates per checkpoints should be >= 0'

        # Autotuning
        procedures_allow = ('random', 'shift', 'shift_u', 'jump', 'jump_u', 'full', 'sanity_check')
        for procedure in self.autotuning_procedures:
            assert isinstance(procedure, str) and procedure.lower() in procedures_allow, \
                f'Invalid autotuning procedure name {procedure}'
        assert self.autotuning_nb_iteration >= 1, 'At least 1 autotuning iteration required'

        # Connector
        assert len(self.start_range_voltage_x) == 2 and len(self.start_range_voltage_y) == 2, \
            'The start_range of voltage should be a list of 2 values (min, max)'
        assert len(self.range_voltage_x) == 2 and len(self.range_voltage_y) == 2, \
            'The range of voltage should be a list of 2 values (min, max)'
        assert self.interaction_mode.lower().strip() in ('auto', 'semi-auto', 'manual'), \
            f'Invalid connector interaction mode: {self.interaction_mode}'

        # If at least one of the voltage ranges is set, check every value consistency
        if not (isnan(self.range_voltage_x[0]) and isnan(self.range_voltage_x[1]) and
                isnan(self.range_voltage_y[0]) and isnan(self.range_voltage_y[1])):
            assert self.range_voltage_x[0] < self.range_voltage_x[1], \
                'The first value of the range_voltage should be lower than the second one (min, max)'
            assert self.range_voltage_y[0] < self.range_voltage_y[1], \
                'The first value of the range_voltage should be lower than the second one (min, max)'
            assert self.start_range_voltage_x[0] < self.start_range_voltage_x[1], \
                'The first value of the start_range_voltage should be lower than the second one (min, max)'
            assert self.start_range_voltage_y[0] < self.start_range_voltage_y[1], \
                'The first value of the start_range_voltage should be lower than the second one (min, max)'
            assert (self.start_range_voltage_x[0] >= self.range_voltage_x[0] and
                    self.start_range_voltage_x[1] <= self.range_voltage_x[1] and
                    self.start_range_voltage_y[0] >= self.range_voltage_y[0] and
                    self.start_range_voltage_y[1] <= self.range_voltage_y[1]), \
                'The start_range_voltage should be inside the range_voltage'

        # Circuit Simulation
        if self.simulate_circuit:
            assert self.use_xyce or self.use_ltspice, \
                'Specify if Xyce or LTspice should be used to run the circuit simulation'
            assert (self.use_xyce and self.use_ltspice) == False, \
                'Specify if Xyce or LTspice should be used to run the circuit simulation, but you can\'t choose both ' \
                'at the same time'
        if self.parameters_clipping is not None:
            assert self.parameters_clipping > 0, 'Parameter clipping should be greater than 0'
        assert self.sim_step_size > 0, 'Simulation step size should be greater than 0'
        assert self.sim_step_size < self.sim_pulse_width + self.sim_pulse_rise_delay + self.sim_pulse_fall_delay + \
               self.sim_init_latency + self.sim_resting_time, 'Simulation step size is too big'
        assert self.sim_r_min < self.sim_r_max, 'LRS value of memristors should be smaller than the HRS value of ' \
                                                'memristors'
        assert self.sim_memristor_read_std >= 0, 'Memristors\' read variability should be greater or equal than 0'
        assert self.sim_memristor_write_std >= 0, 'Memristors\' programming variability should be greater or equal ' \
                                                  'than 0'
        assert self.ratio_failure_HRS >= 0, 'There should be more or equal to 0% of memristors that are in a HRS'
        assert self.ratio_failure_LRS >= 0, 'There should be more or equal to 0% of memristors that are in a LRS'
        assert self.ratio_failure_HRS + self.ratio_failure_LRS <= 1, 'There can\'t be more than 100% of memristors ' \
                                                                     'that are stuck-at fault'
        assert self.sim_pulse_width > 0, 'Simulated pulses width should be greater than 0'
        assert self.sim_pulse_rise_delay > 0, 'Simulated pulses rise delay should be greater than 0'
        assert self.sim_pulse_fall_delay > 0, 'Simulated pulses fall delay should be greater than 0'
        assert self.sim_init_latency > 0, 'Simulation\'s initial latency should be greater than 0'
        assert self.sim_resting_time > 0, 'Simulation\'s resting time should be greater than 0'
        assert self.sim_layer_latency > 0, 'Simulation\'s layer latency should be greater than 0'
        assert self.sim_max_test_inference >= 0, 'Maximum number of test inferences on the circuit should be set ' \
                                                 'to a number greater or equal to 0'
        assert self.sim_nb_process >= 0, 'The number of parallel processes running circuit simulations should ' \
                                         'be set to a number greater or equal to 0'
        if self.simulate_circuit and self.use_ltspice:
            assert self.ltspice_executable_path != '', 'Set the LTspice executable path, it is probably ' \
                                                       'C:\Program Files\ADI\LTspice\LTspice.exe on Windows'
        if self.hardware_aware_training and self.model_type != 'FF':
            raise NotImplementedError("Hardware-aware training is only implemented for FF NNs.")
        if self.use_dropconnect and self.model_type != 'FF':
            raise NotImplementedError("Training with dropconnect is only implemented for FF NNs.")

    def __init__(self):
        """
        Create the setting object.
        """
        self._load_file_and_cmd()

    def _load_file_and_cmd(self) -> None:
        """
        Load settings from local file and arguments of the command line.
        """

        def str_to_bool(arg_value: str) -> bool:
            """
            Used to handle boolean settings.
            If not the 'bool' type convert all not empty string as true.

            :param arg_value: The boolean value as a string.
            :return: The value parsed as a string.
            """
            if isinstance(arg_value, bool):
                return arg_value
            if arg_value.lower() in {'false', 'f', '0', 'no', 'n'}:
                return False
            elif arg_value.lower() in {'true', 't', '1', 'yes', 'y'}:
                return True
            raise argparse.ArgumentTypeError(f'{arg_value} is not a valid boolean value')

        def type_mapping(arg_value):
            if type(arg_value) == bool:
                return str_to_bool
            if is_sequence(arg_value):
                if len(arg_value) == 0:
                    return str
                else:
                    return type_mapping(arg_value[0])

            # Default same as current value
            return type(arg_value)

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        # Spacial argument
        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        # Create argument for each attribute of this class
        for name, value in asdict(self).items():
            p.add_argument(f'--{name.replace("_", "-")}',
                           f'--{name}',
                           dest=name,
                           required=False,
                           action='append' if is_sequence(value) else 'store',
                           type=type_mapping(value))

        # Load arguments form file, environment and command line to override the defaults
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valide the new value.

        :param name: The name of the attribut
        :param value: The value of the attribut
        """
        if name not in self.__dict__ or self.__dict__[name] != value:
            logger.debug(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
            self.__dict__[name] = value

    def __delattr__(self, name):
        raise AttributeError('Removing a setting is forbidden for the sake of consistency.')

    def __str__(self) -> str:
        """
        :return: Human readable description of the settings.
        """
        return 'Settings:\n\t' + \
            '\n\t'.join([f'{name}: {str(value)}' for name, value in asdict(self).items()])


# Singleton setting object
settings = Settings()
