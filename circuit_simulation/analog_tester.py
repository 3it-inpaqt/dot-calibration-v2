import logging
import multiprocessing
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from classes.classifier_nn import ClassifierNN
from circuit_simulation.xyce_simulation import run_circuit_simulation


def test_analog(model: ClassifierNN, test_dataset: Dataset):
    """
    Convert the model an analog circuit, and run the simulation for each input of the test set.

    Args:
        model: The neural network model to convert.
        test_dataset: the test set.
    """
    test_loader = DataLoader(test_dataset)
    outputs_table = []

    for i, (inputs, label) in enumerate(test_loader):
            outputs_table.append(inference_job(model, inputs, label))


def inference_job(model: ClassifierNN, inputs: torch.tensor, label: torch.tensor):
    """
    Run an independent inference job.

    Args:
        model: The NN model to convert in a netlist.
        input_values: The input for this inference.
        label: The label for this input.
    """

    inputs = torch.flatten(inputs).tolist()
    label = label.tolist()

    sim_results, sim_output_before_thr, sim_output = run_circuit_simulation(model, inputs)