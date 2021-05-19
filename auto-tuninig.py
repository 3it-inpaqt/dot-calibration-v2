from pathlib import Path

from classes.diagram import Diagram
from datasets.qdsd import DATA_DIR
from networks.feed_forward import FeedForward
from utils.output import load_network_
from utils.settings import settings

TRAINED_NETWORK = 'out/base-ff/best_network.pt'


def run_auto_tuning():
    diagrams = Diagram.load_diagrams(Path(DATA_DIR, 'interpolated_csv.zip'), None, Path(DATA_DIR, 'charge_area.json'))

    model = FeedForward(input_shape=(settings.patch_size_x, settings.patch_size_y))
    if not load_network_(model, Path(TRAINED_NETWORK)):
        raise RuntimeError(f'Trained parameters not found in: {TRAINED_NETWORK}')


if __name__ == '__main__':
    run_auto_tuning()
