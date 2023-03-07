from connectors.connector import Connector
from runs.run_line_task import clean_up, get_cuda_device, init_model, preparation
from runs.run_tuning_task import run_autotuning
from utils.logger import logger
from utils.output import load_network_
from utils.settings import settings


def start_online_tuning_task() -> None:
    """
    Start an online tuning task. The model has to be pretrained.
    """

    # Check some settings incompatible with the online tuning task
    if settings.autotuning_use_oracle:
        raise ValueError('The online tuning task is not compatible with the Oracle ("autotuning_use_oracle" setting).')
    if settings.trained_network_cache_path is None:
        raise ValueError('A pre-trained model has to be defined for the online tuning task '
                         '("trained_network_cache_path" setting).')

    # Instantiate the model according to the settings
    model = init_model()
    # Load the pretrained model
    if not load_network_(model, settings.trained_network_cache_path, get_cuda_device()):
        raise ValueError(f'Could not load the pretrained model from "{settings.trained_network_cache_path}".')

    # Get the connector to measure online data from experimental setup
    # Then instantiate an online diagram with this connector.
    diagram = Connector.get_connector().get_diagram()

    # Run the autotuning task with one online diagram
    run_autotuning(model, [diagram])


if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        start_online_tuning_task()

    except KeyboardInterrupt:
        logger.error('Online tuning task interrupted by the user.', exc_info=True)
    except Exception:
        logger.critical('Online tuning task interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()
