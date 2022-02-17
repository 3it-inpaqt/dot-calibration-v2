import argparse
import re
from dataclasses import asdict, dataclass
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

    # The seed to use for all random number generator during this run.
    # Forcing reproducibility could lead to a performance lost.
    seed: int = 42

    # If true every baseline are run before the real training. If false this step is skipped.
    evaluate_baselines: bool = True

    # The metric to use for plotting, logging and model performance evaluation.
    # See https://yonvictor.notion.site/Classification-Metrics-2074032f927847c0885918eb9ddc508c
    # Possible values: 'precision', 'recall', 'f1'.
    main_metric: str = 'precision'

    # ==================================================================================================================
    # ============================================== Logging and Outputs ===============================================
    # ==================================================================================================================

    # The minimal logging level to show in the console (see https://docs.python.org/3/library/logging.html#levels).
    logger_console_level: Union[str, int] = 'INFO'

    # The minimal logging level to write in the log file (see https://docs.python.org/3/library/logging.html#levels).
    logger_file_level: Union[str, int] = 'DEBUG'

    # If True a log file is created for each run with a valid run_name.
    # The console logger could be enabled at the same time.
    # If False the logging will only be in console.
    logger_file_enable: bool = True

    # If True uses a visual progress bar in the console during training and loading.
    # Should be use with a logger_console_level as INFO or more for better output.
    visual_progress_bar: bool = True

    # If True show matplotlib images when they are ready.
    show_images: bool = True

    # If True and the run have a valid name, save matplotlib images in the run directory
    save_images: bool = True

    # If True and the run have a valid name, save animated GIF in the run directory
    save_gif: bool = False

    # If True and the run have a valid name, save the neural network parameters in the run directory at the end of the
    # training. Saved before applying early stopping if enabled.
    # The file will be at the root of run directory, under then name: "final_network.pt"
    save_network: bool = True

    # If True the diagrams will be plotted when they are loaded.
    # Always skipped if patches are loaded from cache.
    plot_diagrams: bool = False

    # ==================================================================================================================
    # ==================================================== Dataset =====================================================
    # ==================================================================================================================

    # If true the data will be loaded from cache if possible.
    use_data_cache: bool = True

    # The size of a diagram patch send to the network input (number of pixel)
    patch_size_x: int = 10
    patch_size_y: int = 10

    # The patch overlapping (number of pixel)
    patch_overlap_x: int = 5
    patch_overlap_y: int = 5

    # The width of the border to ignore during the patch labeling (number of pixel)
    # Eg.: If one line touch only 1 pixel at the right of the patch and the label_offset_x is >1 then the patch will be
    # labeled as "no_line"
    label_offset_x: int = 0
    label_offset_y: int = 0

    # The size of the interpolated pixel in Volt.
    # Should be available in the dataset folder.
    pixel_size: float = 0.0025

    # The name of the research group who provide the data (currently: 'louis_gaudreau' or 'michel_pioro_ladriere').
    # Should be available in the dataset folder.
    research_group: str = 'louis_gaudreau'

    # The percentage of data kept for testing only
    test_ratio: float = 0.2

    # The percentage of data kept for testing only
    validation_ratio: float = 0.1

    # If True, data augmentation methods will be applied to increase the size of the train dataset.
    train_data_augmentation: bool = True

    # The number of data loader workers, to take advantage of multithreading. Always disable with CUDA.
    # 0 means automatic setting (using cpu count).
    nb_loader_workers: int = 0

    # If True, the training dataset is balanced using weighted random sampling.
    # see https://github.com/ufoym/imbalanced-dataset-sampler
    balance_class_sampling: bool = False

    # The path to yaml file containing the normalization values (min and max).
    # Use to consistant normalization of the data after the training.
    normalization_values_path: str = ''

    # ==================================================================================================================
    # ===================================================== Model ======================================================
    # ==================================================================================================================

    # The type of model to use (could be a neural network).
    # Have to be in the implemented list: FF, BFF, CNN, BCNN.
    model_type: str = 'FF'

    # The number hidden layer and their respective number of neurons.
    hidden_layers_size: Sequence = (200, 200)

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

    # The momentum value used by the SGD for parameters update.
    momentum: float = 0.9

    # Dropout rate for every dropout layers defined in networks.
    # If a notwork model doesn't have a dropout layer this setting will have no effect.
    # 0 skip dropout layers
    dropout: int = 0.2

    # The size of the mini-batch for the training and testing.
    batch_size: int = 32

    # The number of training epoch.
    nb_epoch: int = 20

    # Save the best network state during the training based on the test main metric.
    # Then load it when the training is complet.
    # The file will be at the root of run directory, under then name: "best_network.pt"
    # Required checkpoints_per_epoch > 0 and checkpoint_validation = True
    early_stopping: bool = False

    # The number of sample used to compute the loss of bayesian networks.
    bayesian_nb_sample_train: int = 3

    # The number of sample used to compute model inference during the validation.
    bayesian_nb_sample_valid: int = 10

    # The number of sample used to compute model inference during the testing.
    bayesian_nb_sample_test: int = 100

    # The metric to use to compute the model inference confidence.
    # Should be in: 'std', 'norm_std', 'entropy', 'norm_entropy'
    bayesian_confidence_metric: str = 'norm_std'

    # The weight of complexity cost part when computing the loss of bayesian networks.
    bayesian_complexity_cost_weight: float = 1 / 50_000

    # ==================================================================================================================
    # ================================================== Checkpoints ===================================================
    # ==================================================================================================================

    # The number of checkpoints per training epoch, if 0 no checkpoint is processed
    checkpoints_per_epoch: int = 0

    # The number of data in the checkpoint training subset.
    # Set to 0 to don't compute the train metrics during checkpoints.
    checkpoint_train_size: int = 640

    # If the inference metrics of the validation dataset should be computed, or not, during checkpoint.
    checkpoint_validation: bool = True

    # If True and the run have a valid name, save the neural network parameters in the run directory at each checkpoint.
    checkpoint_save_network: bool = False

    # ==================================================================================================================
    # =================================================== Autotuning ===================================================
    # ==================================================================================================================

    # The name of the autotuning procedure to use.
    # Have to be in the implemented list: random, shifting, shifting_b, jump, jump_b, full
    autotuning_procedure: str = 'random'

    # If True the line classification model cheat by using the diagram labels (no neural network loaded).
    # Used for baselines.
    autotuning_use_oracle: bool = False

    # If the oracle is enabled, these numbers corrupt its precision.
    # 0.0 = all predictions are based on the ground truth (labels), means 100% precision
    # 0.5 = half of the predictions are random, means 75% precision for binary classification.
    # 1.0 = all the predictions are random, means 50% precision for binary classification.
    # TODO: implement that (and maybe create an Oracle class)
    autotuning_oracle_line_random: float = 0
    autotuning_oracle_no_line_random: float = 0

    # Number of iteration per diagram for the autotuning test
    autotuning_nb_iteration: int = 100

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
        assert self.patch_size_x > 0, 'Patch size should be higher than 0'
        assert self.patch_size_y > 0, 'Patch size should be higher than 0'
        assert self.patch_overlap_x >= 0, 'Patch overlapping should be 0 or more'
        assert self.patch_overlap_y >= 0, 'Patch overlapping should be 0 or more'
        assert self.patch_overlap_x < self.patch_size_x, 'Patch overlapping should be lower than the patch size'
        assert self.patch_overlap_y < self.patch_size_y, 'Patch overlapping should be lower than the patch size'
        assert self.label_offset_x < (self.patch_size_x // 2), 'Label offset should be lower than patch size // 2'
        assert self.label_offset_y < (self.patch_size_y // 2), 'Label offset should be lower than patch size // 2'
        assert self.test_ratio > 0, 'Test data ratio should be more than 0'
        assert self.test_ratio + self.validation_ratio < 1, 'test_ratio + validation_ratio should be less than 1 to' \
                                                            ' have training data'

        # Networks
        assert isinstance(self.model_type, str) and self.model_type.upper() in ['FF', 'BFF', 'CNN', 'BCNN'], \
            f'Invalid network type {self.model_type}'
        assert all((a > 0 for a in self.hidden_layers_size)), 'Hidden layer size should be more than 0'

        # Training
        # TODO should also accept "cuda:1" format
        assert self.device in ('auto', 'cpu', 'cuda'), f'Not valid torch device name: {self.device}'
        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0, 'Number of epoch should be at least 1'
        assert self.bayesian_nb_sample_train > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_nb_sample_valid > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_nb_sample_test > 0, 'The number of bayesian sample should be at least 1'
        assert self.bayesian_confidence_metric in ['std', 'norm_std', 'entropy', 'norm_entropy'], \
            f'Invalid bayesian confidence metric value "{self.bayesian_confidence_metric}"'

        # Checkpoints
        assert self.checkpoints_per_epoch >= 0, 'The number of checkpoints should be >= 0'

        # Autotuning
        procedures_allow = ['random', 'shifting', 'shifting_b', 'jump', 'jump_b', 'full']
        assert isinstance(self.autotuning_procedure, str) and self.autotuning_procedure.lower() in procedures_allow, \
            f'Invalid autotuning procedure name {self.autotuning_procedure}'
        assert self.autotuning_nb_iteration >= 1, 'At least 1 autotuning iteration required'

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
