from collections import Counter
from typing import Dict, List

from tabulate import tabulate

from autotuning.autotuning_procedure import AutotuningProcedure
from autotuning.full_scan import FullScan
from autotuning.jump_shifting import JumpShifting
from autotuning.jump_shifting_bayes import JumpShiftingBayes
from autotuning.random_baseline import RandomBaseline
from autotuning.shifting import Shifting
from autotuning.shifting_bayes import ShiftingBayes
from classes.classifier import Classifier
from classes.classifier_nn import ClassifierNN
from classes.data_structures import AutotuningResult
from datasets.diagram import ChargeRegime, Diagram
from plots.autotuning import plot_autotuning_results
from runs.run_line_task import get_cuda_device
from utils.logger import logger
from utils.output import save_results
from utils.progress_bar import ProgressBar
from utils.settings import settings
from utils.timer import SectionTimer


@SectionTimer('tuning task')
def run_autotuning(model: Classifier, diagrams: List[Diagram]) -> None:
    """
    Run the autotuning simulation.

    :param model: The classifier model used by the tuning procedure.
    :param diagrams: The list of diagrams to run on the tuning procedure.
    """

    # Automatically chooses the device according to the settings, and move diagram data to it.
    device = get_cuda_device()
    logger.debug(f'pyTorch device selected: {device}')
    for diagram in diagrams:
        diagram.to(device)

    # Set up the autotuning procedure according to the settings
    procedure = init_procedure(model)

    # Variables to store stats and results
    autotuning_results = {d.file_basename: [] for d in diagrams}
    nb_iterations = settings.autotuning_nb_iteration * len(diagrams)
    nb_error_to_plot = 10

    # Start the autotuning testing
    logger.info(f'{len(diagrams)} diagram(s) will be process {settings.autotuning_nb_iteration} times '
                f'with the "{procedure}" autotuning procedure')
    with SectionTimer('autotuning simulation'), ProgressBarTuning(nb_iterations) as progress:
        for i in range(settings.autotuning_nb_iteration):
            for diagram in diagrams:
                procedure.reset_procedure()

                # Start the procedure
                procedure.setup_next_tuning(diagram)  # Give diagram and set starting coordinates randomly
                logger.debug(f'Start tuning diagram {diagram.file_basename} '
                             f'(size: {len(diagram.x_axes)}x{len(diagram.y_axes)})')
                result = procedure.run_tuning()

                # Save result and log
                autotuning_results[diagram.file_basename].append(result)
                nb_error_to_plot = save_show_results(result, procedure, i == 0, nb_error_to_plot)

                progress.incr()

    # Log and plot results
    save_show_final_results(autotuning_results)


def init_procedure(model: Classifier) -> AutotuningProcedure:
    """
    Set up the autotuning procedure based on the current settings.
    :return: The procedure.
    """
    patch_size = (settings.patch_size_x, settings.patch_size_y)
    label_offsets = (settings.label_offset_x, settings.label_offset_y)

    # Load model
    if issubclass(type(model), ClassifierNN):
        model.eval()  # Turn off training features (eg. dropout)

    # Load procedure
    procedure_name = settings.autotuning_procedure.lower()
    if procedure_name == 'random':
        return RandomBaseline((settings.patch_size_x, settings.patch_size_y))
    elif procedure_name == 'shifting':
        return Shifting(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'shifting_b':
        return ShiftingBayes(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'jump':
        return JumpShifting(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'jump_b':
        return JumpShiftingBayes(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'full':
        return FullScan(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    else:
        raise ValueError(f'Unknown autotuning procedure name "{settings.autotuning_procedure}".')


def save_show_results(autotuning_result: AutotuningResult, procedure: AutotuningProcedure, is_first_tuning: bool,
                      nb_error_to_plot: int) -> int:
    """
    Save the current autotuning procedure results.

    :param procedure: The procedure which contains tuning stats.
    :param autotuning_result: The autotuning procedure results.
    :param is_first_tuning: True if this is currently the first tuning of this diagram.
    :param nb_error_to_plot: The remaining number of error procedure that we want to plot.
    :return: The remaining number of error procedure that we want to plot after this one.
    """

    success = autotuning_result.is_success_tuning

    # Log information
    logger.debug(f'End tuning {procedure.diagram.file_basename} in {autotuning_result.nb_steps} steps '
                 f'({autotuning_result.success_rate:.1%} success). '
                 f'Final coordinates: {autotuning_result.final_coord} => {autotuning_result.charge_area} e '
                 f'{"[Good]" if success else "[Bad]"}')

    # Plot tuning steps for the first round and some error samples
    if is_first_tuning:
        procedure.plot_step_history(autotuning_result.final_coord, success)
        procedure.plot_step_history_animation(autotuning_result.final_coord, success)
    elif nb_error_to_plot > 0 and not success:
        procedure.plot_step_history(autotuning_result.final_coord, success, plot_vanilla=False)
        procedure.plot_step_history_animation(autotuning_result.final_coord, success)
        nb_error_to_plot -= 1

    return nb_error_to_plot


def save_show_final_results(autotuning_results: Dict[str, List[AutotuningResult]]) -> None:
    """
    Show autotuning results in text output and plots.

    :param autotuning_results: The charge tuning result dictionary.
    :param line_detection_results: The Line classification result dictionary
    """
    overall = Counter()
    headers = ['Diagram', 'Steps', 'Model Success'] + list(map(str, ChargeRegime)) + ['Good', 'Bad', 'Tuning Success']
    results_table = [headers]

    # Process counter of each diagram
    for diagram_name, tuning_results in autotuning_results.items():
        # Count total final regimes
        regimes = Counter()
        for result in tuning_results:
            regimes[result.charge_area] += 1
        overall += regimes

        # Count total steps
        nb_steps = sum(r.nb_steps for r in tuning_results)
        nb_good_inference = sum(r.nb_classification_success for r in tuning_results)
        if nb_steps > 0:
            model_success = nb_good_inference / nb_steps
        else:
            model_success = 0
        overall['steps'] += nb_steps
        overall['good'] += nb_good_inference

        nb_good_regime = regimes[ChargeRegime.ELECTRON_1]
        nb_total = len(tuning_results)
        nb_bad_regime = nb_total - nb_good_regime

        # 'Diagram', 'Steps', 'Model Success'
        results_row = [diagram_name, nb_steps, model_success]
        # Charge Regimes
        results_row += [regimes[regime] for regime in ChargeRegime]
        # 'Good', 'Bad', 'Tuning Success'
        results_row += [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
        results_table.append(results_row)

    plot_autotuning_results(results_table, overall)

    # Overall row
    nb_good_regime = overall[ChargeRegime.ELECTRON_1]
    nb_total = sum((overall.get(charge, 0) for charge in ChargeRegime))
    total_success_rate = nb_good_regime / nb_total if nb_total > 0 else 0
    if len(autotuning_results) > 1:
        nb_bad_regime = nb_total - nb_good_regime
        nb_total_steps = overall['steps']
        nb_total_model_success = overall['good'] / overall['steps'] if overall['steps'] > 0 else 0

        results_row = [f'Sum ({len(autotuning_results)})', nb_total_steps, nb_total_model_success]
        results_row += [overall[regime] for regime in ChargeRegime]
        results_row += [nb_good_regime, nb_bad_regime, total_success_rate]
        results_table.append(results_row)
        overall += overall

    # Print
    logger.info('Autotuning results:\n' +
                tabulate(results_table, headers="firstrow", tablefmt='fancy_grid', floatfmt='.2%'))

    # Save results in yaml file
    save_results(tuning_results=autotuning_results, final_tuning_result=total_success_rate)


class ProgressBarTuning(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to tuning. """

    def __init__(self, nb_iterations: int):
        super().__init__(nb_iterations, task_name='Tuning',
                         enable_color=settings.console_color,
                         boring_mode=not settings.visual_progress_bar,
                         refresh_time=0.5 if settings.visual_progress_bar else 10)
