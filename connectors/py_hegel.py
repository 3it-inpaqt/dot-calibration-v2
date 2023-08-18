import logging
import os
import re
import subprocess as sp
import threading
import time
from queue import Empty, SimpleQueue
from subprocess import Popen
from typing import IO, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from classes.data_structures import ExperimentalMeasurement
from connectors.connector import Connector
from utils.logger import logger
from utils.output import get_new_measurement_out_file_path
from utils.settings import settings


class PyHegel(Connector):

    def __init__(self, amplification: int = None):
        """
        Initialize the PyHegel connector.

        :param amplification: The amplification factor used to convert the measured values to real values in Ampere.
        """
        self._process: Optional[Popen] = None
        self._stdout_queue = SimpleQueue()
        self._stdout_consumer = None

        if amplification is None or amplification <= 0:
            self._amplification = 1e8
            logger.warning(f'No amplification defined, assuming {self._amplification} for all measurements.')
        else:
            self._amplification = amplification

    def _setup_connection(self) -> None:
        """
        Start a PyHegel process and open a PIPE for communicate with it.
        A consumer thread is created to consume the output of the process.
        """

        # Fix a potential crash with the CTRL+C handler in PyHegel sup-process. This is related to a Fortran library.
        os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'TRUE'

        # Start the process
        self._process = Popen([
            'C:\Anaconda3\python.exe',
            'C:\Anaconda3\cwp.py',
            'C:\Anaconda3\envs\py2',
            'C:\Anaconda3\envs\py2\Scripts\pyHegel.exe'
        ],
            stdin=sp.PIPE, stdout=sp.PIPE, text=False, bufsize=0,
            startupinfo=sp.STARTUPINFO(dwFlags=sp.CREATE_NEW_CONSOLE))
        # Create a thread to consume the output of the process
        self._create_stdout_consumer_thread()
        # Wait the end of the initialization
        self._wait_end_of_command(10)

        read_instrument_id = 'USB0::0x0957::0x0607::MY47015885::0'
        axes_instrument_id = 'TCPIP::192.168.150.112::5025::SOCKET'

        commands = [
            # Reading instrument
            f"dmm = instruments.agilent_multi_34410A('{read_instrument_id}')",
            # Gate instruments (same slot for both)
            f"bilt = instruments.iTest_be214x('{axes_instrument_id}', 9)",
            # X-axes instrument (channel 2)
            "set(bilt.slope, 0.1, ch=2)",
            "G1 = (bilt.ramp, dict(ch=2))",
            # Y-axes instrument (channel 1)
            "set(bilt.slope, 0.1, ch=1)",
            "G2 = (bilt.ramp, dict(ch=1))",
        ]

        for command in commands:
            self._send_command(command)

        self._is_connected = True
        logger.info('PyHegel connector ready.')

    def _close_connection(self) -> None:
        """
        Stop the PyHegel process and close the connection.
        Then consume the remaining output.
        After this, a new connection can be opened.
        """
        if self._process is not None:
            self._process.stdin.write(str.encode("exit\n"))

        if self._stdout_consumer is not None:
            # Wait 5 seconds for the thread to stop, but it could be not enough if a scan is in progress
            self._stdout_consumer.join(5)

            if self._stdout_consumer.is_alive():
                logger.error('PyHegel stdout consumer thread did not stop properly (an action is probably in progress)'
                             ', forcing stop.')
                # Force stop current action, the whole process may crash
                self._process.send_signal(sp.signal.CTRL_C_EVENT)

            self._read_process_out()

        self._process = None
        self._stdout_consumer = None
        self._stdout_queue = SimpleQueue()
        self._is_connected = False

        logger.info('PyHegel connector closed.')

    def _measurement(self, start_volt_x: float, end_volt_x: float, step_volt_x: float, start_volt_y: float,
                     end_volt_y: float, step_volt_y: float) -> ExperimentalMeasurement:

        nb_measurements_x = round((end_volt_x - start_volt_x) / step_volt_x)
        nb_measurements_y = round((end_volt_y - start_volt_y) / step_volt_y)

        out_file = get_new_measurement_out_file_path(f'{self._nb_measurement:03}_'
                                                     f'{start_volt_x:.4f}V_{start_volt_y:.4f}V')

        # Internally pyHegel use np.linespace which include the last point.
        # So we manually exclude the last point for compatibility with the other part of the code.
        end_volt_x -= step_volt_x
        end_volt_y -= step_volt_y

        # Send the command to pyHegel, block the process until it's done.
        self._send_command(
            f"sweep_multi([G1,G2], "
            f"[{start_volt_x:.4f}, {start_volt_y:.4f}], "
            f"[{end_volt_x:.4f}, {end_volt_y:.4f}], "
            f"[{nb_measurements_x}, {nb_measurements_y}], "
            f"out=dmm.readval, "
            f"filename=r'{out_file.resolve()}', "
            f"progress={logger.getEffectiveLevel() <= logging.DEBUG}, "
            f"graph=False, "
            f"updown=[False,'alternate'])",
            # Wait the answer for a maximum of 1 sec per point + some time for voltage ramp
            max_wait_time=nb_measurements_x * nb_measurements_y + 120
        )

        # Parse the output file
        with open(out_file) as f:
            x, y, values = PyHegel._load_raw_points(f, self._amplification)

        return ExperimentalMeasurement(x, y, values, None)

    def _send_command(self, command: str, max_wait_time: float = 10) -> None:
        """
        Send a command to pyHegel process.
        :param command: The command to send.
        :param max_wait_time: The maximum duration before to stop waiting the answer and fail.
        """

        if self._process is None:
            raise RuntimeError('No pyHegel process running. Setup connection should be called first.')

        mode = settings.interaction_mode.lower().strip()

        if mode == 'manual':
            # Wait for the user to run the command himself.
            input(f'[MANUAL]: {command}')
            return

        if mode == 'semi-auto':
            # Wait for the user to validate the command.
            validation = input(f'[SEMI-AUTO]: {command}\n[Y/n] ').strip().lower()
            if validation != 'y' and validation != '':
                raise RuntimeError(f'Command "{command}" rejected by user.')

        # Here if mode auto, or validated semi-auto
        if mode == 'auto' or mode == 'semi-auto':
            logger.debug(f'Sending command to pyHegel: "{command}"')
            self._process.stdin.write(str.encode(command + "\n"))
            self._wait_end_of_command(max_wait_time)
        else:
            # Here if mode is not recognized
            raise ValueError(f'Interaction mode "{mode}" not supported.')

    def _wait_end_of_command(self, max_wait_time: float, refresh_time: float = 0.1):

        start_time = time.time()
        while True:
            out = self._read_process_out()

            # If we find this pattern the command should be done. If we don't find it, keep waiting
            if re.search('^In \[\d+]: $', out, re.MULTILINE):
                return

            # Check for time out
            if time.time() - start_time > max_wait_time:
                raise RuntimeError(f'No answer from pyHegel process before the current command timeout: '
                                   f'{max_wait_time:.2f}s')

            # Wait a bit before to start again
            time.sleep(refresh_time)

    @staticmethod
    def _load_raw_points(file: IO, amplification: int = 1e8) -> Tuple[Sequence[float], Sequence[float], Tensor]:
        """
        Load the raw files with all columns.

        :param file: The diagram file to load.
        :param amplification: The amplification factor used to convert the measured values to real values in Ampere.
        :return: The columns x, y, z according to the selected ones.
        """

        data = np.loadtxt(file, usecols=(0, 1, 2))

        # Sort by y, then by x because the y sweeping can alternate between top and down.
        data = data[np.lexsort((data[:, 0], data[:, 1]))]

        x = np.unique(data[:, 0])
        y = np.unique(data[:, 1])
        values = torch.tensor(data[:, 2] / amplification, dtype=torch.float)

        # Reshape the values to match the x and y axes
        values = values.reshape(len(y), len(x))

        logger.debug(f'Raw measurement data parsed, {len(x)}×{len(y)} points with amplification: {amplification}')

        return x, y, values

    def _create_stdout_consumer_thread(self):

        def read_th():
            while True:
                # Block here until the process write in the pip
                out = self._process.stdout.read(1024)
                # Add the output to the queue
                self._stdout_queue.put_nowait(out)
                if out == b"":
                    break

        self._stdout_consumer = threading.Thread(target=read_th, daemon=True)
        self._stdout_consumer.start()

    def _read_process_out(self) -> str:

        out_strs = []
        # Read the queue
        while True:
            try:
                out_strs.append(self._stdout_queue.get_nowait().decode('utf8'))
            except Empty:
                break

        process_out = ''.join(out_strs)

        if process_out != '':
            logger.debug('[PY_HEGEL OUT]\n' + process_out.strip())

        return process_out
