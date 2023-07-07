from pathlib import Path

from datasets.diagram_ndot_offline import DiagramOfflineNDot
from datasets.diagram_offline import DiagramOffline
from datasets.qdsd import DATA_DIR
from runs.run_line_task import clean_up, preparation
from runs.run_tuning_task import run_autotuning
from start_lines import start_line_task
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def start_tuning_offline_task() -> None:
    """
    Start the charge tuning task.
    Train a model if necessary, then test it.
    """
    nb_rep = 1
    while nb_rep > 0:

        # Start with line task: train model if necessary, then test it
        model = None if settings.autotuning_use_oracle else start_line_task()
        model_diagram = DiagramOffline if settings.dot_number == 1 else DiagramOfflineNDot
        with SectionTimer('datasets loading', 'debug'):
            # Load diagrams from files (line and area)
            diagrams = model_diagram.load_diagrams(pixel_size=settings.pixel_size,
                                                   research_group=settings.research_group,
                                                   diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                                   labels_path=Path(DATA_DIR, 'labels.json'),
                                                   single_dot=True if settings.dot_number == 1 else False,
                                                   load_lines=True,
                                                   load_areas=True,
                                                   white_list=[settings.test_diagram] if settings.test_diagram else None)

            # Normalize the diagram with the same min/max value used during the training.
            # The values are fetch via the "normalization_values_path" setting or in the current run directory.
            model_diagram.normalize_diagrams(diagrams)

        # Run the autotuning task
        run_autotuning(model, diagrams)
        nb_rep -= 1

if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        start_tuning_offline_task()

    except KeyboardInterrupt:
        logger.error('Tuning task interrupted by the user.', exc_info=True)
    except Exception:
        logger.critical('Tuning task interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()
