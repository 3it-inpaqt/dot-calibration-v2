import re
from typing import IO, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from classes.data_structures import ExperimentalMeasurement
from connectors.connector import Connector
from utils.logger import logger
from utils.output import get_new_measurement_out_file_path
from utils.settings import settings


class PyHegel(Connector):
    def _setup_connection(self) -> None:
        # TODO smarter parameters handling
        read_instrument_id = 'USB0::0x2A8D::0x0101::MY57515472::INSTR'
        axes_y_instrument_id = 'TCPIP::192.168.150.112::5025::SOCKET'
        axes_x_instrument_id = 'TCPIP::192.168.150.112::5025::SOCKET'

        commands = [
            # Reading instrument
            f"dmm = instruments.agilent_multi_34410A('{read_instrument_id}')",
            # Y-axes instrument
            f"bilt1 = instruments.iTest_be2102('{axes_y_instrument_id}', 1)",
            f"D1 = instruments.RampDevice(bilt1, 0.1)",
            # X-axes instrument
            f"bilt3 = instruments.iTest_be2102('{axes_x_instrument_id}', 3)",
            f"B2 = instruments.RampDevice(bilt3, 0.1)"
        ]

        for command in commands:
            self._send_command(command)

    def _measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float, start_volt_y: float,
                     end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:

        nb_measurements_x = round((end_volt_x - start_volt_x) / step_volt_x)
        nb_measurements_y = round((end_volt_y - start_volt_y) / step_volt_y)

        out_file = get_new_measurement_out_file_path(f'py_hegel_{start_volt_x:.4f}V_{start_volt_y:.4f}V')

        # Send the command to pyHegel, block the process until it's done.
        self._send_command(
            f"sweep_multi([B2,D1], "
            f"[{start_volt_x:.4f}, {start_volt_y:.4f}], "
            f"[{end_volt_x:.4f}, {end_volt_y:.4f}], "
            f"[{nb_measurements_x}, {nb_measurements_y}], "
            f"out=dmm.readval, "
            f"filename=r'{out_file.resolve()}', "
            f"graph=None, "
            f"updown=[False,'alternate'])"
        )

        # Parse the output file
        with open(out_file) as f:
            x, y, values = PyHegel._load_raw_points(f)

        return ExperimentalMeasurement(x, y, values)

    def _send_command(self, command: str) -> None:
        """
        Send a command to pyHegel process.
        :param command: The command to send.
        """

        logger.debug(f'Sending command to pyHegel: "{command}"')
        mode = settings.interaction_mode.lower().strip()

        if mode == 'manual':
            # Wait for the user to run the command himself.
            input(f'[MANUAL]: {command}')
        else:
            raise NotImplementedError  # TODO implement sending commands to pyHegel

    @staticmethod
    def _load_raw_points(file: IO) -> Tuple[Sequence[float], Sequence[float], Tensor]:
        """
        Load the raw files with all columns.

        :param file: The diagram file to load.
        :return: The columns x, y, z according to the selected ones.
        """

        amplification = None
        for line in file:
            line = line.decode("utf-8")
            match = re.match(r'.*:= Ampli(?:fication)?[=:](.+)$', line)
            if match:
                amplification = int(float(match[1]))  # Parse exponential notation to integer
                break
            if line[0] != '#':
                raise RuntimeError('Amplification value not found in file comments')

        data = np.loadtxt(file)
        x = np.unique(data[:, 0])
        y = np.unique(data[:, 2])
        values = torch.tensor(data[:, 4] / amplification, dtype=torch.float)

        logger.debug(f'Raw measurement data parsed, {len(x)} lines with amplification: {amplification}')

        return x, y, values
