from datasets.qdsd import QDSDLines
from networks.simple_classifier import SimpleClassifier
from run import clean_up, preparation, run
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def main():
    # Prepare the environment
    preparation()

    # Catch and log every exception during the runtime
    # noinspection PyBroadException
    try:
        with SectionTimer('datasets loading', 'debug'):
            # Load the training dataset
            train_set = QDSDLines(test=False, patch_size=(settings.patch_size_x, settings.patch_size_y),
                                  overlap=(settings.patch_overlap_x, settings.patch_overlap_y))

            # Load test testing dataset
            test_set = QDSDLines(test=True, patch_size=(settings.patch_size_x, settings.patch_size_y),
                                 overlap=(settings.patch_overlap_x, settings.patch_overlap_y))

        # Build the network
        net = SimpleClassifier(input_size=100, nb_classes=2)

        # Run the training and the test
        run(train_set, test_set, net)
    except KeyboardInterrupt:
        logger.error('Run interrupted by the user.')
        raise  # Let it go to stop the runs planner if needed
    except Exception:
        logger.critical('Run interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        clean_up()


if __name__ == '__main__':
    main()
