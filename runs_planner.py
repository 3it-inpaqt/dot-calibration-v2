from main import main
from utils.logger import logger
from utils.planner import BasePlanner, CombinatorPlanner, ParallelPlanner, Planner
from utils.settings import settings


def start_planner(runs_planner: BasePlanner, skip_validation: bool = False):

    # Typical settings override for planner runs
    settings.run_name = None  # The name should be override during the runs
    settings.visual_progress_bar = False
    settings.show_images = False
    settings.use_data_cache = False

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

    # At every iteration of the loop the settings will be update according to the planner current state
    for i, run_name in enumerate(runs_planner):
        logger.info(f'Run {i:02n}/{nb_runs:n} ({i / nb_runs:05.1%})')
        # Set the name of this run according to the planner
        # All other settings are already set during the "next" operation
        settings.run_name = run_name
        # Start the run
        main()


if __name__ == '__main__':
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
        Planner('seed', range(3, 4))
    ], runs_basename='patch_size-seed-2')

    start_planner(patch_size)
