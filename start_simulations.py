from runs.run_line_task import clean_up, preparation
from start_lines import start_line_task
from utils.logger import logger
from utils.settings import settings
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
    destination_directory = './results/' + new_folder

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
    """
    This is like running start_lines.py by systematically varying settings related to neural network architectures,
    memristor non-idealities and training strategies. It iterates through different configurations specified by lists
    such as `research_groups`, `new_folders`, `hidden_layers_sizes`, etc. It executes the `main()` function with each
    configuration and stores the results accordingly.
    """
    research_groups = ['louis_gaudreau']
    new_folders = ['louis']
    hidden_layers_sizes = [(5,), (10,), (15,), (5,10), (10,5)]
    hidden_layers_names = ['5', '10', '15', '5_10', '10_5']
    batch_norm_layers_list = [(False,), (False,), (False,), (False,False), (False,False)]
    write_stds = [0.0, 0.008]
    HRS_list = [0.0, 0.05, 0.1]
    HRS_names = ['hrs0', 'hrs5', 'hrs10']
    LRS_list = [0.1, 0.05, 0.0]
    LRS_names = ['lrs10', 'lrs5', 'lrs0']
    ha_list = [False, True]

    for research_group, new_folder in zip(research_groups, new_folders):
        settings.research_group = research_group
        for hidden_layers_size, hidden_layers_name, batch_norm_layers in zip(hidden_layers_sizes, hidden_layers_names,
                                                                             batch_norm_layers_list):
            settings.hidden_layers_size = hidden_layers_size
            settings.batch_norm_layers = batch_norm_layers
            settings.trained_network_cache_path = ''
            for write_std in write_stds:
                settings.sim_memristor_write_std = write_std
                if write_std == write_stds[0]:
                    settings.hardware_aware_training = False
                    settings.ratio_failure_HRS = 0.0
                    settings.ratio_failure_LRS = 0.0
                    main()
                    try:
                        shutil.copyfile('./out/tmp/best_network.pt', './best_network.pt')
                        settings.trained_network_cache_path = 'best_network.pt'
                    except Exception as e:
                        print(e)
                    new_name = hidden_layers_name + '_std0_hrs0_lrs0'
                    move_and_rename(new_folder, new_name)
                else:
                    for ha in ha_list:
                        settings.hardware_aware_training = ha
                        if ha:
                            settings.trained_network_cache_path = ''
                        for hrs, hrs_name, lrs, lrs_name in zip(HRS_list, HRS_names, LRS_list, LRS_names):
                            settings.ratio_failure_HRS = hrs
                            settings.ratio_failure_LRS = lrs
                            settings.hardware_aware_training = ha
                            main()
                            new_name = hidden_layers_name + '_std08_' + hrs_name + '_' + lrs_name
                            if ha:
                                new_name = new_name + '_ha'
                            move_and_rename(new_folder, new_name)