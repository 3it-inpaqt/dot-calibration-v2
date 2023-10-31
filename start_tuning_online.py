from math import isnan

from connectors.connector import Connector
from runs.run_line_task import clean_up, get_cuda_device, init_model, preparation
from runs.run_tuning_task import run_autotuning
from utils.logger import logger
from utils.output import load_network_, push_notification
from utils.settings import settings
from utils.timer import SectionTimer


def start_tuning_online_task() -> None:
    """
    Start an online tuning task. The model has to be pretrained.
    """
    check_settings()

    # Instantiate the model according to the settings
    device = get_cuda_device()
    model = init_model().to(device)
    # Load the pretrained model
    if not load_network_(model, settings.trained_network_cache_path, device, load_thresholds=True):
        raise ValueError(f'Could not load the pretrained model from "{settings.trained_network_cache_path}".')

    with Connector.get_connector() as connector:
        # Get the connector to measure online data from experimental setup.
        # Then instantiate an online diagram with this connector.
        with SectionTimer('setup connection'):
            diagram = connector.get_diagram()

        # Run the autotuning task with one online diagram
        run_autotuning(model, [diagram])

    # Send a push notification when the tuning is finished
    push_notification('Online tuning finished', f'Online tuning "{settings.run_name}" finished')


def check_settings() -> None:
    """
    Check some settings incompatible with the online tuning task before to start.
    """
    if settings.autotuning_use_oracle:
        raise ValueError('The online tuning task is not compatible with the Oracle ("autotuning_use_oracle" setting).')
    if settings.trained_network_cache_path is None:
        raise ValueError('A pre-trained model has to be defined for the online tuning task '
                         '("trained_network_cache_path" setting).')
    if (isnan(settings.range_voltage_x[0]) or isnan(settings.range_voltage_x[1]) or
            isnan(settings.range_voltage_y[0]) or isnan(settings.range_voltage_y[1])):
        raise ValueError('The min and max voltage have to be defined before to start an online tuning task '
                         '("range_voltage_x" and "range_voltage_y" settings).')
    if (settings.range_voltage_x[1] - settings.range_voltage_x[0] > 4 or
            settings.range_voltage_y[1] - settings.range_voltage_y[0] > 4):
        raise ValueError('The voltage range is too large for the online tuning task (>4V).')


if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        start_tuning_online_task()

    except KeyboardInterrupt:
        logger.error('Online tuning task interrupted by the user.', exc_info=True)
    except Exception as e:
        logger.critical('Online tuning task interrupted by an unexpected error.', exc_info=True)

        # Send a push notification if the tuning failed
        push_notification('Online tuning failed', f'Online tuning "{settings.run_name}" failed with '
                                                  f'an unexpected error: {e.__class__.__name__}')
    finally:
        # Clean up the environment, ready for a new run
        clean_up()
