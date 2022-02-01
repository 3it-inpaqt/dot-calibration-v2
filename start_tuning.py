from runs.run_line_task import clean_up, preparation
from runs.run_tuning_task import run_autotuning
from utils.logger import logger


def main():
    """
    Start the charge tuning task.
    Train a model if necessary, then test it.
    """

    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        run_autotuning()
    except KeyboardInterrupt:
        logger.error('Tuning task interrupted by the user.')
        raise  # Let it go to stop the task planner if needed
    except Exception:
        logger.critical('Tuning task interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()


if __name__ == '__main__':
    main()
