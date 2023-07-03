import torch
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryConfusionMatrix
from pandas import DataFrame
import multiprocessing
import time

from classes.classifier_nn import ClassifierNN
from circuit_simulation.xyce_simulation import run_circuit_simulation
from circuit_simulation.generate_netlist import NetlistGenerator
from utils.logger import logger
from plots.model_performance import plot_confusion_matrix, plot_analog_vs_digital_before_threshold, \
    plot_analog_before_threshold_hist
from utils.settings import settings
from utils.timer import duration_to_str
from utils.output import save_inferences
from utils.progress_bar import ProgressBar


def test_analog(model: ClassifierNN, test_dataset: Dataset):
    """
    Convert the model an analog circuit, and run the simulation for each input of the test set.

    Args:
        model: The neural network model to convert.
        test_dataset: the test set.
    """
    logger.info('Start network testing on circuit simulation...')

    test_loader = DataLoader(test_dataset)

    # Limit the number of simulation (0 means the whole test set)
    hard_limit = settings.xyce_max_test_inference if settings.xyce_max_test_inference > 0 else len(test_loader)

    # Define the number of parallel process to run (0 means the number of cpu cores)
    nb_process = settings.xyce_nb_process if settings.xyce_nb_process > 0 else multiprocessing.cpu_count()
    nb_process = min(nb_process, hard_limit)

    logger.info(f'Start {hard_limit} Xyce simulations ' +
                 ('' if nb_process == 1 else f'({nb_process} parallel processes) ') + '...')
    job_args = []
    outputs_table = []
    start_time = time.perf_counter()

    netlist_generator = NetlistGenerator(model)

    with ProgressBar(hard_limit, task_name='Inferences on circuit simulation', auto_display=True) as progress_bar:
        for i, (input_values, label) in enumerate(test_loader):

            if i >= hard_limit:
                break

            if nb_process > 1:
                # Save arguments list to run on the multiprocess pool later
                job_args.append((model, input_values, label, netlist_generator, i==0))
            else:
                # Single thread mode, so just run it now
                outputs_table.append(inference_job(model, input_values, label, netlist_generator, i==0))
                progress_bar.incr()

        # Start multi-thread pool
        if nb_process > 1:
            # Spawn process to save memory and avoid inheriting too many file descriptors
            context = multiprocessing.get_context('spawn')
            with context.Pool(processes=nb_process) as pool:
                outputs_table = pool.starmap(inference_job, job_args)

    sim_run_time = time.perf_counter() - start_time
    logger.info(f'Xyce simulations completed in {duration_to_str(sim_run_time)} '
                 f'({duration_to_str(sim_run_time / hard_limit)} / sim)')

    outputs_table = DataFrame(outputs_table)

    # Confusion matrix for analog model on test set
    test_analog_cm = BinaryConfusionMatrix()
    test_analog_cm(torch.tensor(outputs_table['analog_output']), torch.tensor(outputs_table['label']))

    # Confusion matrix between digital and analog model on test set
    analog_fidelity_cm = BinaryConfusionMatrix()
    analog_fidelity_cm(torch.tensor(outputs_table['analog_output']), torch.tensor(outputs_table['digital_output']))

    # Plot model performances after the analog test
    plot_confusion_matrix(test_analog_cm, 'Confusion matrix for analog model\non test set',
                          'confusion_matrix_test_analog')
    plot_confusion_matrix(analog_fidelity_cm,
                          'Confusion matrix between digital and analog model\non test set',
                          'confusion_matrix_fidelity', x_label='Analog',
                          y_label='Digital')

    plot_analog_vs_digital_before_threshold(DataFrame(outputs_table))

    plot_analog_before_threshold_hist(outputs_table['analog_before_th'])

    save_inferences(outputs_table)


def inference_job(model: ClassifierNN, inputs: torch.tensor, label: torch.tensor, netlist_generator: NetlistGenerator, is_first_run: bool):
    """
    Run an independent inference job.

    Args:
        model: The NN model to convert in a netlist.
        input: The input for this inference.
        is_first_run: True if this is the first inference
    """

    digital_output_before_thr = model(inputs)
    digital_output = (digital_output_before_thr > 0).int()

    inputs = torch.flatten(inputs).tolist()

    # Run the inference on the simulated circuit
    sim_results, sim_output_before_thr, sim_output = run_circuit_simulation(netlist_generator, inputs, is_first_run)

    return {
        'input': inputs,
        'label': label.int().item(),
        'analog_before_th': sim_output_before_thr,
        'analog_logic_before_th': sim_output_before_thr / settings.xyce_pulse_amplitude,
        'analog_output': sim_output,
        'digital_before_th': digital_output_before_thr.item(),
        'digital_output': digital_output.item()
    }
