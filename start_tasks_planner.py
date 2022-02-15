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

    # Typical settings override for planner runs
    settings.run_name = None  # The name should be override during the runs
    settings.visual_progress_bar = True
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

        logger.info('Completed successful validation of the runs planner')

    nb_runs = len(runs_planner)
    logger.info(f'Starting a set of {nb_runs} runs with a planner')

    skipped_runs = list()

    # At every iteration of the loop the settings will be update according to the planner current state
    for i, run_name in enumerate(runs_planner):
        logger.info(f'Run {i:02n}/{nb_runs:n} ({i / nb_runs:05.1%}) - {run_name}')
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

    # Patch size study
    size_range = range(7, 32)
    overlap_range = [s // 2 for s in size_range]
    offset_range = [s // 7 for s in size_range]
    patch_size = CombinatorPlanner([
        ParallelPlanner([
            Planner('patch_overlap_x', overlap_range),
            Planner('patch_overlap_y', overlap_range),
            Planner('label_offset_x', offset_range),
            Planner('label_offset_y', offset_range),
            Planner('patch_size_x', size_range),
            Planner('patch_size_y', size_range),
        ]),
        Planner('seed', range(5, 7))
    ], runs_basename='patch_size_cnn')

    # Hidden layer size study
    layers_size = CombinatorPlanner([
        SequencePlanner([
            Planner('hidden_layers_size', [[a] for a in range(5, 50, 5)]),
            Planner('hidden_layers_size', [[a] for a in range(50, 100, 10)]),
            # Planner('hidden_layers_size', [[a] for a in range(200, 1000, 100)]),
            # Planner('hidden_layers_size', [[a] for a in range(1000, 5000, 250)])
        ]),
        Planner('seed', range(2, 4))
    ], runs_basename='layers_size-seed_2')

    run_tasks_planner(repeat, skip_existing_runs=True)
