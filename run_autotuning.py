from collections import Counter
from pathlib import Path
from typing import Optional

from tabulate import tabulate

from autotuning.autotuning_procedure import AutotuningProcedure
from autotuning.czischek_2021 import Czischek2021
from autotuning.full_scan import FullScan
from autotuning.random_baseline import RandomBaseline
from classes.classifier import Classifier
from classes.diagram import ChargeRegime, Diagram
from datasets.qdsd import DATA_DIR
from plots.autotuning import plot_autotuning_results
from run import clean_up, get_cuda_device, init_model, preparation
from utils.logger import logger
from utils.output import load_network_, save_results
from utils.progress_bar import ProgressBar
from utils.settings import settings
from utils.timer import SectionTimer


def run_autotuning() -> None:
    """ Run the autotuning simulation. """
    # Load diagrams from files (line and area)
    diagrams = Diagram.load_diagrams(pixel_size=settings.pixel_size,
                                     research_group=settings.research_group,
                                     diagrams_path=Path(DATA_DIR, 'interpolated_csv.zip'),
                                     labels_path=Path(DATA_DIR, 'labels.json'),
                                     single_dot=True,
                                     load_lines=True,
                                     load_areas=True)

    # Set up the autotuning procedure according to the settings
    procedure = setup_procedure()

    # Start the autotuning testing
    logger.info(f'{len(diagrams)} diagram(s) will be process {settings.autotuning_nb_iteration} times '
                f'with the "{procedure}" autotuning procedure')
    autotuning_results = {d.file_basename: Counter() for d in diagrams}
    line_detection_results = {d.file_basename: Counter() for d in diagrams}
    nb_iterations = settings.autotuning_nb_iteration * len(diagrams)
    with SectionTimer('autotuning simulation'), \
            ProgressBar(nb_iterations, task_name='Autotuning', auto_display=settings.visual_progress_bar) as progress:
        for i in range(settings.autotuning_nb_iteration):
            for diagram in diagrams:
                procedure.reset_procedure()

                # Start the procedure
                start_coord = procedure.get_random_coordinates_in_diagram(diagram)
                logger.debug(f'Start tuning diagram {diagram.file_basename} '
                             f'(size: {len(diagram.x_axes)}x{len(diagram.y_axes)})')
                tuned_x, tuned_y = procedure.tune(diagram, start_coord)

                # Save final result and log
                nb_steps = procedure.get_nb_steps()
                nb_classification_success = procedure.get_nb_line_detection_success()
                charge_area = diagram.get_charge(tuned_x, tuned_y)
                autotuning_results[diagram.file_basename][charge_area] += 1
                line_detection_results[diagram.file_basename].update({'steps': nb_steps,
                                                                      'good': nb_classification_success})
                logger.debug(f'End tuning {diagram.file_basename} in {nb_steps} steps '
                             f'({nb_classification_success / nb_steps:.1%} success). '
                             f'Final coordinates: ({tuned_x}, {tuned_y}) => {charge_area} e '
                             f'{"[Good]" if charge_area is ChargeRegime.ELECTRON_1 else "[Bad]"}')

                progress.incr()
                # Plot tuning steps for the first round
                if i == 0:
                    procedure.plot_step_history(diagram, (tuned_x, tuned_y))

    # Save results in yaml file
    save_results(final_regims={file: {str(charge): value for charge, value in counter.items()}
                               for file, counter in autotuning_results.items()},
                 line_detection={file: dict(counter) for file, counter in line_detection_results.items()})
    # Log and plot results
    show_results(autotuning_results, line_detection_results)


def setup_procedure() -> AutotuningProcedure:
    """
    Set up the autotuning procedure based on the current settings.
    :return: The procedure.
    """
    patch_size = (settings.patch_size_x, settings.patch_size_y)
    label_offsets = (settings.label_offset_x, settings.label_offset_y)

    # Load model
    model: Optional[Classifier] = None
    if not settings.autotuning_use_oracle:
        model = init_model()
        if not load_network_(model, Path(settings.trained_network_cache_path), get_cuda_device()):
            # TODO allow to train network here
            raise RuntimeError(f'Trained parameters not found in: {settings.trained_network_cache_path}')
        model.eval()  # Turn off training features (eg. dropout)

    # Load procedure
    procedure_name = settings.autotuning_procedure.lower()
    if procedure_name == 'random':
        return RandomBaseline((settings.patch_size_x, settings.patch_size_y))
    elif procedure_name == 'czischek':
        return Czischek2021(model, patch_size, label_offsets, settings.autotuning_use_oracle)
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
        model_success = line_detection_result['good'] / line_detection_result['steps']

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
        nb_total_model_success = overall['good'] / overall['steps']

        results_row = [f'Sum ({len(autotuning_results)})', nb_total_steps, nb_total_model_success]
        results_row += [overall[regime] for regime in ChargeRegime]
        results_row += [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
        results_table.append(results_row)
        overall += overall

    # Print
    logger.info('Autotuning results:\n' +
                tabulate(results_table, headers="firstrow", tablefmt='fancy_grid', floatfmt='.2%'))


if __name__ == '__main__':
    # Prepare the environment
    preparation()

    # noinspection PyBroadException
    try:
        run_autotuning()
    except KeyboardInterrupt:
        logger.error('Run interrupted by the user.')
        raise  # Let it go to stop the runs planner if needed
    except Exception:
        logger.critical('Run interrupted by an unexpected error.', exc_info=True)
    finally:
        clean_up()
