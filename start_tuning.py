from pathlib import Path

from datasets.diagram import Diagram
from datasets.qdsd import DATA_DIR
from runs.run_line_task import clean_up, preparation
from runs.run_tuning_task import run_autotuning
from start_lines import start_line_task
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def start_tuning_task() -> None:
    """
    Start the charge tuning task.
    Train a model if necessary, then test it.
    """

    # Start with line task: train model if necessary, then test it
    model = start_line_task()

    with SectionTimer('datasets loading', 'debug'):
        # Load diagrams from files (line and area)
        diagrams = Diagram.load_diagrams(pixel_size=settings.pixel_size,
                                         research_group=settings.research_group,
                                         diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                         labels_path=Path(DATA_DIR, 'labels.json'),
                                         single_dot=True,
                                         load_lines=True,
                                         load_areas=True)

        # Normalize the diagram with the same min/max value used during the training.
        # The values are fetch via the "normalization_values_path" setting or in the current run directory.
        Diagram.normalize(diagrams)

    # Run the autotuning task
    run_autotuning(model, diagrams)


if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        start_tuning_task()

    except KeyboardInterrupt:
        logger.error('Tuning task interrupted by the user.')
        raise  # Let it go to stop the task planner if needed
    except Exception:
        logger.critical('Tuning task interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()
