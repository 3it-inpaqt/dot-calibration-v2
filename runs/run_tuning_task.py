from collections import Counter, defaultdict
from itertools import chain
from typing import Dict, List, Optional, Tuple

from tabulate import tabulate

from autotuning.autotuning_procedure import AutotuningProcedure
from autotuning.full_scan import FullScan
from autotuning.jump import Jump
from autotuning.jump_uncertainty import JumpUncertainty
from autotuning.random_baseline import RandomBaseline
from autotuning.sanity_check import SanityCheck
from autotuning.shift import Shift
from autotuning.shift_uncertainty import ShiftUncertainty
from classes.classifier import Classifier
from classes.classifier_nn import ClassifierNN
from classes.data_structures import AutotuningResult
from datasets.diagram import Diagram
from datasets.diagram_offline import ChargeRegime
from datasets.diagram_online import DiagramOnline
from plots.autotuning import plot_autotuning_results
from runs.run_line_task import get_cuda_device
from utils.logger import logger
from utils.output import save_results
from utils.progress_bar import ProgressBar
from utils.settings import settings
from utils.timer import SectionTimer


@SectionTimer('tuning task')
def run_autotuning(model: Optional[Classifier], diagrams: List[Diagram]) -> None:
    """
    Run the autotuning simulation.

    :param model: The classifier model used by the tuning procedure. If the model is None, the Oracle option should be
        enabled.
    :param diagrams: The list of diagrams to run on the tuning procedure. It could be offline or online diagrams.
    """
    if len(diagrams) == 0:
        raise ValueError('No diagram provided to the tuning run.')

    if settings.autotuning_use_oracle:
        logger.warning('Oracle option enabled. The labels are used instead of a classification model. '
                       'For baseline only.')

    # Automatically chooses the device according to the settings, and move diagram data to it.
    device = get_cuda_device()
    logger.debug(f'pyTorch device selected: {device}')
    for diagram in diagrams:
        diagram.to(device)

    # Variables to store stats and results
    autotuning_results = defaultdict(list)

    # Start the autotuning testing
    is_online = ' online' if len(diagrams) == 1 and isinstance(diagrams[0], DiagramOnline) else ''
    logger.info(f'{len(diagrams)}{is_online} diagram(s) will be process {settings.autotuning_nb_iteration} time(s) '
                f'with {len(settings.autotuning_procedures)} autotuning procedure(s)')

    # Only 1 iteration for the 'full' procedure
    nb_iterations = sum(1 if p == 'full' else settings.autotuning_nb_iteration for p in settings.autotuning_procedures)
    nb_iterations *= len(diagrams)
    with SectionTimer('autotuning simulation'), ProgressBarTuning(nb_iterations) as progress:
        for procedure_name in settings.autotuning_procedures:
            # Set up the autotuning procedure_name according to the settings
            procedure = init_procedure(model, procedure_name)
            # Count the amount of success and failure to plot (value decremented inside the function save_show_results)
            nb_to_plot = Counter({False: settings.nb_plot_tuning_fail, True: settings.nb_plot_tuning_success})

            nb_iteration = 1 if procedure_name == 'full' else settings.autotuning_nb_iteration
            for i in range(nb_iteration):
                for diagram in diagrams:
                    procedure.reset_procedure()

                    # Start the procedure
                    procedure.setup_next_tuning(diagram)  # Give diagram and set starting coordinates randomly
                    logger.debug(f'Start tuning diagram: {diagram}')
                    result = procedure.run_tuning()

                    # Save result and log
                    autotuning_results[(procedure_name, diagram.name)].append(result)
                    save_show_results(result, procedure, nb_to_plot)

                    progress.incr()

    # Log and plot results
    save_show_final_results(autotuning_results)


def init_procedure(model: Optional[Classifier], procedure_name: str) -> AutotuningProcedure:
    """
    Set up the autotuning procedure based on the current settings.
    :param model: The model to use for line classification. If the model is None, the Oracle option should be enabled.
    :param procedure_name: The name of the tuning procedure to use.
    :return: The procedure.
    """
    patch_size = (settings.patch_size_x, settings.patch_size_y)
    label_offsets = (settings.label_offset_x, settings.label_offset_y)

    # Load model
    if model is not None and issubclass(type(model), ClassifierNN):
        model.eval()  # Turn off training features (eg. dropout)

    # Load procedure
    procedure_name = procedure_name.lower()
    if procedure_name == 'random':
        return RandomBaseline((settings.patch_size_x, settings.patch_size_y))
    elif procedure_name == 'shift':
        return Shift(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'shift_u':
        return ShiftUncertainty(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'jump':
        return Jump(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'jump_u':
        return JumpUncertainty(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'full':
        return FullScan(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    elif procedure_name == 'sanity_check':
        return SanityCheck(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    else:
        raise ValueError(f'Unknown autotuning procedure name "{procedure_name}".')


@SectionTimer('save results', log_level='debug')
def save_show_results(autotuning_result: AutotuningResult, procedure: AutotuningProcedure,
                      nb_to_plot: Counter[bool]):
    """
    Save the current autotuning procedure results.

    :param procedure: The procedure which contains tuning stats.
    :param autotuning_result: The autotuning procedure results.
    :param nb_to_plot: The remaining number of results that we want to plot (independent count for success and fail).
     The counter is decremented after each plot.
    """
    success = autotuning_result.is_success_tuning

    # Log information
    logger.debug(f'End tuning {procedure.diagram.name} in {autotuning_result.nb_steps} steps '
                 f'({autotuning_result.success_rate:.1%} success). '
                 f'Final coordinates: {autotuning_result.final_coord} => {autotuning_result.charge_area} e '
                 f'{"[Good]" if success else "[Bad]"}')

    if nb_to_plot[success] > 0 or isinstance(procedure, FullScan):
        procedure.plot_step_history(autotuning_result.final_volt_coord, success)
        procedure.plot_step_history_animation(autotuning_result.final_volt_coord, success)
        nb_to_plot[success] -= 1


@SectionTimer('save results', log_level='debug')
def save_show_final_results(autotuning_results: Dict[Tuple[str, str], List[AutotuningResult]]) -> None:
    """
    Show autotuning results in text output and plots.

    :param autotuning_results: The charge tuning result dictionary.
    """
    overall = Counter()
    headers = ['Procedure'] if len(settings.autotuning_procedures) > 1 else []
    headers += ['Diagram', 'Steps', 'Model Success'] + list(map(str, ChargeRegime)) + ['Good', 'Bad', 'Tuning Success']
    results_table = [headers]

    # Process counter of each diagram
    for (procedure_name, diagram_name), tuning_results in autotuning_results.items():
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

        results_row = [procedure_name] if len(settings.autotuning_procedures) > 1 else []
        # 'Diagram', 'Steps', 'Model Success'
        results_row += [diagram_name, nb_steps, model_success]
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
    if len(autotuning_results) > 1 and len(settings.autotuning_procedures) == 1:
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
                tabulate(results_table, headers='firstrow', tablefmt='fancy_grid' if settings.console_color else 'grid',
                         floatfmt='.2%'))

    # Save flatten array results in yaml file
    save_results(tuning_results=chain(*autotuning_results.values()), final_tuning_result=total_success_rate)


class ProgressBarTuning(ProgressBar):
    """ Override the ProgressBar to define print configuration adapted to tuning. """

    def __init__(self, nb_iterations: int):
        super().__init__(nb_iterations, task_name='Tuning',
                         enable_color=settings.console_color,
                         boring_mode=not settings.visual_progress_bar,
                         refresh_time=0.5 if settings.visual_progress_bar else 10)
