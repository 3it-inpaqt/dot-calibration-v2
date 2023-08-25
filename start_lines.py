from classes.classifier import Classifier
from classes.classifier_nn import ClassifierNN
from datasets.qdsd import QDSDLines
from plots.data import plot_patch_sample
from runs.run_line_task import clean_up, init_model, preparation, run_train_test
from utils.logger import logger
from utils.settings import settings
from utils.timer import SectionTimer


def start_line_task() -> Classifier:
    """
    Start the line classification task.
    Train a model if necessary, then test it.

    :return: The trained classifier model.
    """

    with SectionTimer('datasets loading', 'debug'):
        # If some test diagrams are defined in settings, use them for the test set, otherwise use the test ratio
        test_ratio_or_names = [settings.test_diagram] if len(settings.test_diagram) > 0 else settings.test_ratio
        # Load datasets for line classification task (only line labels)
        train_set, test_set, valid_set = QDSDLines.build_split_datasets(test_ratio_or_names=test_ratio_or_names,
                                                                        validation_ratio=settings.validation_ratio,
                                                                        patch_size=(settings.patch_size_x,
                                                                                    settings.patch_size_y),
                                                                        overlap=(settings.patch_overlap_x,
                                                                                 settings.patch_overlap_y),
                                                                        label_offset=(settings.label_offset_x,
                                                                                      settings.label_offset_y),
                                                                        pixel_size=settings.pixel_size,
                                                                        research_group=settings.research_group)
    # Plot a sample of test data
    plot_patch_sample(test_set, 8)

    # Instantiate the model according to the settings
    model = init_model()

    # Run the training and the test (train skipped if 'trained_network_cache_path' setting defined)
    if issubclass(type(model), ClassifierNN):
        model = run_train_test(train_set, test_set, valid_set, model)
    else:
        # TODO start test here
        raise NotImplemented(f'Not implemented run for model type "{type(model)}"')

    return model


if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # Catch and log every exception during runtime
    # noinspection PyBroadException
    try:
        trained_model = start_line_task()
    except KeyboardInterrupt:
        logger.error('Line task interrupted by the user.', exc_info=True)
    except Exception:
        logger.critical('Line task interrupted by an unexpected error.', exc_info=True)
    finally:
        # Clean up the environment, ready for a new run
        if 'trained_model' in locals():
            del trained_model
        clean_up()