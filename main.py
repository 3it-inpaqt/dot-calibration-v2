from datasets.qdsd import QDSDLines
from networks.feed_forward import FeedForward
from plots.data import plot_patch_sample
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
            train_set, test_set = QDSDLines.build_split_datasets(test_ratio=settings.test_ratio,
                                                                 patch_size=(settings.patch_size_x,
                                                                             settings.patch_size_y),
                                                                 overlap=(settings.patch_overlap_x,
                                                                          settings.patch_overlap_y),
                                                                 label_offset=(settings.label_offset_x,
                                                                               settings.label_offset_y))
            plot_patch_sample(test_set, 8)

        # Build the network
        net = FeedForward(input_shape=(settings.patch_size_x, settings.patch_size_y))

        # Run the training and the test
        run(train_set, test_set, net)
    except KeyboardInterrupt:
        logger.error('Run interrupted by the user.')
        raise  # Let it go to stop the runs planner if needed
    except Exception:
        logger.critical('Run interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        del train_set, test_set, net
        clean_up()


if __name__ == '__main__':
    main()
