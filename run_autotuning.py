from collections import Counter
from pathlib import Path
from typing import Optional

from tabulate import tabulate

from autotuning.autotuning_procedure import AutotuningProcedure
from autotuning.czischek_2021 import Czischek2021
from autotuning.random_baseline import RandomBaseline
from classes.classifier_nn import ClassifierNN
from classes.diagram import ChargeRegime, Diagram
from datasets.qdsd import DATA_DIR
from plots.autotuning import plot_autotuning_results
from run import clean_up, init_neural_network, preparation
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
    results = {d.file_basename: Counter() for d in diagrams}
    nb_iterations = settings.autotuning_nb_iteration * len(diagrams)
    with SectionTimer('autotuning simulation'), ProgressBar(nb_iterations, task_name='Autotuning') as progress:
        for i in range(settings.autotuning_nb_iteration):
            for diagram in diagrams:
                procedure.reset_procedure()
                # Start the procedure
                start_coord = procedure.get_random_coordinates_in_diagram(diagram)
                logger.debug(f'Start tuning diagram {diagram.file_basename} '
                             f'(size: {len(diagram.x_axes)}x{len(diagram.y_axes)})')
                tuned_x, tuned_y = procedure.tune(diagram, start_coord)
                # Save final result
                charge_area = diagram.get_charge(tuned_x, tuned_y)
                results[diagram.file_basename][charge_area] += 1
                logger.debug(f'End tuning {diagram.file_basename} in {procedure.get_nb_steps()} steps. '
                             f'Final coordinates: ({tuned_x}, {tuned_y}) => {charge_area} e '
                             f'{"[Good]" if charge_area is ChargeRegime.ELECTRON_1 else "[Bad]"}')

                progress.incr()
                # Plot tuning steps for the first round
                if i == 0:
                    procedure.plot_step_history(diagram, (tuned_x, tuned_y))

    # Save results in yaml file
    save_results(final_regims={file: {str(charge): value for charge, value in counter.items()}
                               for file, counter in results.items()})
    # Log and plot results
    show_results(results)


def setup_procedure() -> AutotuningProcedure:
    """
    Set up the autotuning procedure based on the current settings.
    :return: The procedure.
    """
    patch_size = (settings.patch_size_x, settings.patch_size_y)
    label_offsets = (settings.label_offset_x, settings.label_offset_y)

    # Load model
    model: Optional[ClassifierNN] = None
    if not settings.autotuning_use_oracle:
        model = init_neural_network()
        if not load_network_(model, Path(settings.trained_network_cache_path)):
            # TODO allow to train network here
            raise RuntimeError(f'Trained parameters not found in: {settings.trained_network_cache_path}')

    # Load procedure
    procedure_name = settings.autotuning_procedure.lower()
    if procedure_name == 'random':
        return RandomBaseline((settings.patch_size_x, settings.patch_size_y))
    elif procedure_name == 'czischek':
        return Czischek2021(model, patch_size, label_offsets, settings.autotuning_use_oracle)
    else:
        raise ValueError(f'Unknown autotuning procedure name "{settings.autotuning_procedure}".')


def show_results(results: dict) -> None:
    """
    Show autotuning results in text output and plots.

    :param results: The result dictionary.
    """
    overall = Counter()
    headers = ['Diagram'] + list(map(str, ChargeRegime)) + ['Good', 'Bad', 'Success Rate']
    results_table = [headers]

    # Process counter of each diagram
    for diagram_name, diagram_counter in results.items():
        nb_good_regime = diagram_counter[ChargeRegime.ELECTRON_1]
        nb_total = sum(diagram_counter.values())
        nb_bad_regime = nb_total - nb_good_regime

        results_row = [diagram_counter[regime] for regime in ChargeRegime]
        results_row = [diagram_name] + results_row + [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
        results_table.append(results_row)
        overall += diagram_counter

    plot_autotuning_results(results_table)

    # Overall row
    if len(results_table) > 1:
        nb_good_regime = overall[ChargeRegime.ELECTRON_1]
        nb_total = sum(overall.values())
        nb_bad_regime = nb_total - nb_good_regime

        results_row = [overall[regime] for regime in ChargeRegime]
        results_row = ['Sum'] + results_row + [nb_good_regime, nb_bad_regime, (nb_good_regime / nb_total)]
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
