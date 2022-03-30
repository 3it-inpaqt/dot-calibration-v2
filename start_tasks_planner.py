from itertools import chain

from runs.run_line_task import clean_up, fix_seed
from start_lines import start_line_task
from start_tuning import start_tuning_task
from utils.logger import logger
from utils.output import ExistingRunName, init_out_directory, set_plot_style
from utils.planner import BasePlanner, CombinatorPlanner, ParallelPlanner, Planner, SequencePlanner
from utils.settings import settings


def run_tasks_planner(runs_planner: BasePlanner,
                      tuning_task: bool = False,
                      skip_validation: bool = False,
                      skip_existing_runs: bool = False) -> None:
    """
    Start a sequence of task following the runs planner.

    :param runs_planner: The run planner to follow.
    :param tuning_task: If True start the tuning task, if False start the line task.
    :param skip_validation: If True skip the settings validation.
    :param skip_existing_runs: If True, skipp run for which an out directory already exists.
    """
    # Setup logger before to start
    logger.set_console_level(settings.logger_console_level)
    logger.set_formatter(settings.console_color)

    # Typical settings override for planner runs
    settings.run_name = None  # The name should be overridden during the runs
    settings.show_images = False
    settings.use_data_cache = True

    # Set plot style
    set_plot_style()

    if not skip_validation:
        logger.info('Start validation of the runs planner...')

        try:
            # Iterate through all values to make sure all settings names and values are correct.
            # This way it's less likely that an error occurs during the real processing.
            for _ in runs_planner:
                settings.validate()
        except AssertionError as exc:
            raise RuntimeError('Invalid planned settings, runs aborted before to start') from exc

        BasePlanner.reset_names()
        logger.info('Completed successful validation of the runs planner')

    nb_runs = len(runs_planner)
    logger.info(f'Starting a set of {nb_runs} runs with a planner')

    skipped_runs = list()

    # At every iteration of the loop the settings will be update according to the planner current state
    for i, run_name in enumerate(runs_planner):
        logger.info(f'Runs competed: {i}/{nb_runs} ({i / nb_runs:05.1%}) - Next run: {run_name}')
        # Set the name of this run according to the planner
        # All other settings are already set during the "next" operation
        settings.run_name = run_name

        try:
            # Create the output directory to save results and plots
            init_out_directory()
        except ExistingRunName:
            if skip_existing_runs:
                logger.info(f'Skip existing run {run_name}')
                skipped_runs.append(run_name)
                continue
            else:
                logger.critical(f'Existing run directory: "{run_name}"', exc_info=True)
                break

        # Fix seed
        if settings.seed is not None:
            fix_seed()

        # noinspection PyBroadException
        try:
            # Start the run
            if tuning_task:
                start_tuning_task()
            else:
                start_line_task()

        except KeyboardInterrupt:
            logger.error('Line task interrupted by the user.', exc_info=True)
            break
        except Exception:
            logger.critical('Line task interrupted by an unexpected error.', exc_info=True)
        finally:
            # Clean up the environment, ready for a new run
            clean_up()

    if len(skipped_runs) == nb_runs:
        logger.error(f'All {nb_runs} runs skipped')
    elif len(skipped_runs) == 1:
        logger.warning(f'1 existing run skipped: {skipped_runs[0]}')
    elif len(skipped_runs) > 1:
        logger.warning(f'{len(skipped_runs)} existing runs skipped')


if __name__ == '__main__':
    # Same run several times with the default settings but different seed
    repeat = Planner('seed', range(3), runs_basename='tmp')

    # Default settings for each dataset (for code factorisation)
    datasets_planner = ParallelPlanner([
        Planner('research_group', ['michel_pioro_ladriere', 'louis_gaudreau']),
        Planner('pixel_size', [0.001, 0.0025]),
        Planner('nb_epoch', [0, 0]),  # Nb epoch defined by nb_train_update
        Planner('label_offset_x', [6, 7]),
        Planner('label_offset_y', [6, 7]),
        Planner('conv_layers_channel', [[12, 24], [6, 12]]),
    ])

    # Default settings for each NN
    ffs_hidden_size = [200, 100]
    cnns_hidden_size = [100, 50]
    ffs_lr = 0.0005
    cnns_lr = 0.001
    ff_update = 30_000
    cnn_update = 20_000
    bcnn_update = 25_000

    # Patch size study
    size_range = range(8, 28)
    overlap_range = [s // 2 for s in size_range]
    offset_range = [s // 3 for s in size_range]
    patch_size = CombinatorPlanner([
        Planner('model_type', ['CNN']),  # No Bayesian to save time
        Planner('evaluate_baselines', [True]),
        Planner('hidden_layers_size', [cnns_hidden_size]),
        Planner('learning_rate', [cnns_lr]),
        Planner('nb_train_update', [cnn_update]),
        datasets_planner,
        ParallelPlanner([
            Planner('patch_overlap_x', overlap_range),
            Planner('patch_overlap_y', overlap_range),
            Planner('label_offset_x', offset_range),
            Planner('label_offset_y', offset_range),
            Planner('patch_size_x', size_range),
            Planner('patch_size_y', size_range),
        ])
    ], runs_name='patch_size_cnn')

    # Hidden layer size study
    layers_size = CombinatorPlanner([
        Planner('model_type', ['FF', 'CNN']),  # No Bayesian to save time
        Planner('learning_rate', [ffs_lr, cnns_lr]),
        Planner('nb_train_update', [ff_update, cnn_update]),
        datasets_planner,
        SequencePlanner([
            # 1 Layer
            Planner('hidden_layers_size', [[a] for a in range(5, 50, 5)]),
            Planner('hidden_layers_size', [[a] for a in range(50, 400, 25)]),
            # 2 Layers
            Planner('hidden_layers_size', [[a * 2, a] for a in range(5, 50, 5)]),
            Planner('hidden_layers_size', [[a * 2, a] for a in range(50, 400, 25)]),
        ]),
    ], runs_name='layers_size')

    # Batch size study
    train_batch_size = CombinatorPlanner([
        Planner('model_type', ['FF', 'CNN']),  # No Bayesian to save time
        Planner('hidden_layers_size', [ffs_hidden_size, cnns_hidden_size]),
        Planner('learning_rate', [ffs_lr, cnns_lr]),
        Planner('nb_train_update', [ff_update, cnn_update]),
        datasets_planner,
        Planner('batch_size', list(chain(range(25, 150, 25), range(150, 500, 50), range(500, 2000, 100))))
    ], runs_name='train_batch_size')

    # Make full scan plots
    full_scan_all = CombinatorPlanner([
        Planner('autotuning_procedure', ['full']),
        Planner('autotuning_nb_iteration', [1]),
        Planner('autotuning_use_oracle', [False]),
        Planner('save_images', [True]),
        Planner('plot_diagrams', [True]),
        ParallelPlanner([
            Planner('model_type', ['FF', 'CNN', 'BCNN']),
            Planner('hidden_layers_size', [ffs_hidden_size, cnns_hidden_size, cnns_hidden_size]),
            Planner('learning_rate', [ffs_lr, cnns_lr, cnns_lr]),
            Planner('nb_train_update', [ff_update, cnn_update, bcnn_update]),
        ]),
        datasets_planner
    ], runs_name='full_scan')

    # Train all networks with all datasets
    train_all_networks = CombinatorPlanner([
        CombinatorPlanner([
            Planner('evaluate_baselines', [True]),
            ParallelPlanner([
                Planner('model_type', ['FF', 'CNN', 'BCNN']),
                Planner('hidden_layers_size', [ffs_hidden_size, cnns_hidden_size, cnns_hidden_size]),
                Planner('learning_rate', [ffs_lr, cnns_lr, cnns_lr]),
                Planner('nb_train_update', [ff_update, cnn_update, bcnn_update]),
            ]),
            datasets_planner
        ]),
        Planner('seed', range(10))
    ], runs_name='ref')

    # Run tuning on all datasets and procedures
    tune_all_diagrams = CombinatorPlanner([
        Planner('autotuning_nb_iteration', [50]),
        ParallelPlanner([
            CombinatorPlanner([
                ParallelPlanner([
                    Planner('model_type', ['CNN', 'CNN', 'BCNN']),
                    Planner('hidden_layers_size', [cnns_hidden_size, cnns_hidden_size, cnns_hidden_size]),
                    Planner('learning_rate', [cnns_lr, cnns_lr, cnns_lr]),
                    Planner('nb_train_update', [1, cnn_update, bcnn_update]),
                    Planner('autotuning_use_oracle', [True, False, False]),
                ]),
                datasets_planner
            ]),
        ]),
        Planner('autotuning_procedure', ['shifting', 'shifting_b', 'jump', 'jump_b']),
        # TODO turn the autotuning_procedure setting into a list to avoid multiple training
    ], runs_name='tuning')

    run_tasks_planner(train_all_networks, skip_existing_runs=True, tuning_task=False)
