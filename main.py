from datasets.mock_classification_dataset import MockClassificationDataset
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
            train_set = MockClassificationDataset(settings.nb_classes, settings.train_point_per_class)
            train_set.show_plot()  # Plot and show the data

            # Load test testing dataset
            test_set = MockClassificationDataset(settings.nb_classes, settings.test_point_per_class)

        # Build the network
        net = SimpleClassifier(input_size=2, nb_classes=len(train_set.classes))

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
