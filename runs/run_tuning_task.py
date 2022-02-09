from collections import Counter
from typing import List

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
from classes.diagram import ChargeRegime, Diagram
from plots.autotuning import plot_autotuning_results
from utils.logger import logger
from utils.output import save_results
from utils.progress_bar import ProgressBar
from utils.settings import settings
from utils.timer import SectionTimer


def run_autotuning(model: Classifier, diagrams: List[Diagram]) -> None:
    """
    Run the autotuning simulation.

    :param model: The classifier model used by the tuning procedure.
    :param diagrams: The list of diagrams to run on the tuning procedure.
    """

    # Set up the autotuning procedure according to the settings
    procedure = init_procedure(model)

    # Start the autotuning testing
    logger.info(f'{len(diagrams)} diagram(s) will be process {settings.autotuning_nb_iteration} times '
                f'with the "{procedure}" autotuning procedure')
    autotuning_results = {d.file_basename: Counter() for d in diagrams}
    line_detection_results = {d.file_basename: Counter() for d in diagrams}
    nb_iterations = settings.autotuning_nb_iteration * len(diagrams)
    nb_error_to_plot = 10

    with SectionTimer('autotuning simulation'), \
            ProgressBar(nb_iterations, task_name='Autotuning', auto_display=settings.visual_progress_bar) as progress:
        for i in range(settings.autotuning_nb_iteration):
            for diagram in diagrams:
                procedure.reset_procedure()

                # Start the procedure
                procedure.setup_next_tuning(diagram)  # Give diagram and set starting coordinates randomly
                logger.debug(f'Start tuning diagram {diagram.file_basename} '
                             f'(size: {len(diagram.x_axes)}x{len(diagram.y_axes)})')
                tuned_x, tuned_y = procedure.tune()

                # Save result and log
                nb_steps = procedure.get_nb_steps()
                nb_classification_success = procedure.get_nb_line_detection_success()
                success_rate = nb_classification_success / nb_steps if nb_steps > 0 else 0
                charge_area = diagram.get_charge(tuned_x, tuned_y)
                success_tuning = charge_area is ChargeRegime.ELECTRON_1
                autotuning_results[diagram.file_basename][charge_area] += 1
                line_detection_results[diagram.file_basename].update({'steps': nb_steps,
                                                                      'good': nb_classification_success})
                logger.debug(f'End tuning {diagram.file_basename} in {nb_steps} steps '
                             f'({success_rate:.1%} success). '
                             f'Final coordinates: ({tuned_x}, {tuned_y}) => {charge_area} e '
                             f'{"[Good]" if charge_area is ChargeRegime.ELECTRON_1 else "[Bad]"}')

                progress.incr()
                # Plot tuning steps for the first round and some error samples
                if i == 0:
                    procedure.plot_step_history((tuned_x, tuned_y), success_tuning)
                    procedure.plot_step_history_animation((tuned_x, tuned_y), success_tuning)
                elif nb_error_to_plot > 0 and not success_tuning:
                    procedure.plot_step_history((tuned_x, tuned_y), success_tuning, plot_vanilla=False)
                    procedure.plot_step_history_animation((tuned_x, tuned_y), success_tuning)
                    nb_error_to_plot -= 1

    # Save results in yaml file
    save_results(final_regimes={file: {str(charge): value for charge, value in counter.items()}
                                for file, counter in autotuning_results.items()},
                 line_detection={file: dict(counter) for file, counter in line_detection_results.items()})
    # Log and plot results
    show_results(autotuning_results, line_detection_results)


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


def show_results(autotuning_results: dict, line_detection_results: dict) -> None:
    """
    Show autotuning results in text output and plots.

    :param autotuning_results: The charge tuning result dictionary.
    :param line_detection_results: The Line classification result dictionary
    """
    overall = Counter()
    headers = ['Diagram', 'Steps', 'Model Success'] + list(map(str, ChargeRegime)) + ['Good', 'Bad', 'Tuning Success']
    results_table = [headers]

    # Process counter of each diagram
    for diagram_name, diagram_counter in autotuning_results.items():
        line_detection_result = line_detection_results[diagram_name]
        nb_steps = line_detection_result['steps']
        if line_detection_result['steps'] > 0:
            model_success = line_detection_result['good'] / line_detection_result['steps']
        else:
            model_success = 0

        nb_good_regime = diagram_counter[ChargeRegime.ELECTRON_1]
        nb_total = sum(diagram_counter.values())
        nb_bad_regime = nb_total - nb_good_regime

        # 'Diagram', 'Steps', 'Model Success'
        results_row = [diagram_name, nb_steps, model_success]
        # Charge Regimes
        results_row += [diagram_counter[regime] for regime in ChargeRegime]
        # 'Good', 'Bad', 'Tuning Success'
        results_row += [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
        results_table.append(results_row)
        overall += diagram_counter
        overall += line_detection_result

    plot_autotuning_results(results_table, overall)

    # Overall row
    if len(autotuning_results) > 1:
        nb_good_regime = overall[ChargeRegime.ELECTRON_1]
        nb_total = sum((overall.get(charge, 0) for charge in ChargeRegime))
        nb_bad_regime = nb_total - nb_good_regime
        nb_total_steps = overall['steps']
        nb_total_model_success = overall['good'] / overall['steps'] if overall['steps'] > 0 else 0

        results_row = [f'Sum ({len(autotuning_results)})', nb_total_steps, nb_total_model_success]
        results_row += [overall[regime] for regime in ChargeRegime]
        results_row += [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
        results_table.append(results_row)
        overall += overall

    # Print
    logger.info('Autotuning results:\n' +
                tabulate(results_table, headers="firstrow", tablefmt='fancy_grid', floatfmt='.2%'))
