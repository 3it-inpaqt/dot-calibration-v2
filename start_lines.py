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
        run_train_test(train_set, test_set, valid_set, model)
    else:
        # TODO start test here
        raise NotImplemented(f'Not implemented run for model type "{type(model)}"')

    return model


import shutil

def move_directory(source_dir, destination_dir):
    try:
        shutil.move(source_dir, destination_dir)
        print(f"Directory '{source_dir}' moved to '{destination_dir}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def move_and_rename(new_folder, new_name):
    old_directory_name = './out/tmp'
    new_directory_name = './out/' + new_name
    destination_directory = './résultats/article/' + new_folder

    move_directory(old_directory_name, new_directory_name)
    move_directory(new_directory_name, destination_directory)


def main():
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


if __name__ == '__main__':
    main()
    # research_groups = ['louis_gaudreau']
    # new_folders = ['louis']
    # hidden_layers_sizes = [(5,), (10,), (15,), (5,10), (10,5)]
    # hidden_layers_names = ['5', '10', '15', '5_10', '10_5']
    # batch_norm_layers_list = [(False,), (False,), (False,), (False,False), (False,False)]
    # write_stds = [0.0, 0.008]
    # HRS_list = [0.0, 0.05, 0.1]
    # HRS_names = ['hrs0', 'hrs5', 'hrs10']
    # LRS_list = [0.1, 0.05, 0.0]
    # LRS_names = ['lrs10', 'lrs5', 'lrs0']
    # ha_list = [False, True]
    #
    # for research_group, new_folder in zip(research_groups, new_folders):
    #     settings.research_group = research_group
    #     for hidden_layers_size, hidden_layers_name, batch_norm_layers in zip(hidden_layers_sizes, hidden_layers_names,
    #                                                                          batch_norm_layers_list):
    #         settings.hidden_layers_size = hidden_layers_size
    #         settings.batch_norm_layers = batch_norm_layers
    #         settings.trained_network_cache_path = ''
    #         for write_std in write_stds:
    #             settings.sim_memristor_write_std = write_std
    #             if write_std == write_stds[0]:
    #                 settings.hardware_aware_training = False
    #                 settings.ratio_failure_HRS = 0.0
    #                 settings.ratio_failure_LRS = 0.0
    #                 main()
    #                 try:
    #                     shutil.copyfile('./out/tmp/best_network.pt', './best_network.pt')
    #                     settings.trained_network_cache_path = 'C:\\Users\\Yohan\\Documents\\doc 1\\UdeS\\Stage 3IT\\dot-calibration-memristor-simulation\\best_network.pt'
    #                 except Exception as e:
    #                     print(e)
    #                 new_name = hidden_layers_name + '_std0_hrs0_lrs0'
    #                 move_and_rename(new_folder, new_name)
    #             else:
    #                 for ha in ha_list:
    #                     settings.hardware_aware_training = ha
    #                     if ha:
    #                         settings.trained_network_cache_path = ''
    #                     for hrs, hrs_name, lrs, lrs_name in zip(HRS_list, HRS_names, LRS_list, LRS_names):
    #                         settings.ratio_failure_HRS = hrs
    #                         settings.ratio_failure_LRS = lrs
    #                         settings.hardware_aware_training = ha
    #                         main()
    #                         new_name = hidden_layers_name + '_std08_' + hrs_name + '_' + lrs_name
    #                         if ha:
    #                             new_name = new_name + '_ha'
    #                         move_and_rename(new_folder, new_name)