from subprocess import call
import os
from tempfile import mkstemp
from shutil import move
import numpy as np
import shutil
import circuit_simulation.ltspice_local.xyceTranslate as xyceTranslate
from utils.logger import logger as logging
from utils.settings import settings as config
import time
import psutil

# ----------- Simulation controls ----------- #

def run_simulations(parameter_set=None, numerical_name_start=0):
    # Set appropriate variables according to the argument of parameter_set
    if parameter_set is not None:
        parameter = parameter_set[0]
        parameter_value_list = parameter_set[1]
        use_default_parameters = False
    else:
        use_default_parameters = True

    # Specify file paths
    file_path = config.LTSpice_asc_filename[:-4] # Use .asc file specified in config, but remove file ending
    file_path_generated = file_path + '_generated'
    spice_exe_path = config.ltspice_executable_path

    # Create a list of the generated files
    output_filenames = []

    if not use_default_parameters:
        # Run a simulation for each parameter value in the parameter set
        for i, parameter_value in enumerate(parameter_value_list):
            # Set specified parameters
            if config.LTspice_output_data_naming_convention == 'number':
                file_num = str(i + numerical_name_start)
                output_name = '0'*(3-len(file_num)) + file_num
                output_path = config.LTspice_output_directory + output_name + '.txt'
            else:
                output_path = config.LTspice_output_directory + parameter + '=' + str(parameter_value) + '.txt'
            output_filenames.append(output_path)
            set_parameters(file_path + '.' + config.LTSpice_asc_filetype , parameter, parameter_value)
            logging.debug('Starting simulation with the specified parameter: ' + parameter + '=' + str(parameter_value))
            # Run simulation
            simulate(spice_exe_path, file_path_generated)
            # Set header and cleanup the file
            output_header = 'SPICE simulation result. Parameters: ' + ', '.join(get_parameters(file_path_generated + '.' + config.LTSpice_asc_filetype)) + '\n' # Maybe not add the time variables
            if config.use_ltspice:
                clean_raw_file(spice_exe_path, file_path_generated, output_path, output_header)
    else:
        # Run a simulation with the preset values of the file
        output_path = config.output_data_path + 'result.txt'
        logging.debug('Starting simulation.')
        simulate(spice_exe_path, file_path)
        # Set header and cleanup the file
        output_header = 'SPICE simulation result. Parameters: ' + ', '.join(get_parameters(file_path + '.' + config.LTSpice_asc_filetype)) + '\n' # Maybe not add the time variables
        if config.use_ltspice:
           clean_raw_file(spice_exe_path, file_path, output_path, output_header)

    # Return the list with names of the output filenames
    return output_filenames

def simulate(spice_exe_path, file_path):
    file_name = os.path.join(os.getcwd(), file_path)
    file_name1 = str(file_path.split('/')[-1])
    logging.debug('Simulation starting: ' + file_name + '.' + config.LTSpice_asc_filetype)
    if config.LTSpice_asc_filetype == 'asc':
        if psutil.WINDOWS:
           runcmd = '"' + spice_exe_path + '" -netlist "' + file_path + '.' + config.LTSpice_asc_filetype + '"'
        elif psutil.LINUX:
            runcmd = '' + spice_exe_path + ' -netlist ' + file_path + '.' + config.LTSpice_asc_filetype + ''
        call(runcmd)
        while os.path.exists(file_path + '.net') == False:
            print('Waiting for "{file_path}" + .net to be created')
            time.sleep(0.001)
        shutil.copyfile(file_path + '.net',config.LTspice_output_directory + file_name1 + '.net')
        xyceTranslate.translate_netlist2xyce(config.LTspice_output_directory + file_name1 + '.net')
        runcmd = '"' + spice_exe_path + '" -b -ascii "' + file_path + '.net"'
        if config.use_ltspice:
           call(runcmd)
    else:
        runcmd = '"' + spice_exe_path + '" -b -ascii "' + file_path + '.' + config.LTSpice_asc_filetype + '"'
        call(runcmd)
    if config.use_ltspice:
        size = os.path.getsize(file_path + '.raw')
        logging.debug('Simulation finished: ' + file_name + '.raw created (' + str(size/1000) + ' kB)')
    else: 
        logging.debug('Finished processing files without simulation ')
def clean_raw_file(spice_exe_path, file_path, output_path, output_header):
    # Try to open the requested file
    file_name = file_path
    file_name1 = str(file_path.split('/')[-1])
    try:
        shutil.copyfile(file_path + '.raw',config.LTspice_output_directory + file_name1 + '.raw')
        shutil.copyfile(file_path + '.op.raw',config.LTspice_output_directory + file_name1 + '.op.raw')
        f = open(file_path + '.raw', 'r')
    except IOError:
        # If the requested raw file is not found, simulations will be run,
        # assuming a that a corresponding LTspice schematic exists
        logging.error('File not found: ' + file_name + '.raw')
        simulate(spice_exe_path, file_path)
        f = open(file_path + '.raw', 'r')

    logging.debug('Cleaning up file: ' + file_name + '.raw')

    reading_header = True
    data = []
    data_line = []

    for line_num, line in enumerate(f):

        if reading_header:
            if line_num == 4:
                number_of_vars = int(line.split(' ')[-1])
            if line_num == 5:
                number_of_points = int(line.split(' ')[-1])
            if line[:7] == 'Values:':
                reading_header = False
                header_length = line_num + 1
                continue
        else:
            data_line_num = (line_num - header_length) % number_of_vars
            if data_line_num in config.LTspice_variable_numbering.values():
                data_line.append(line.split('\t')[-1].split('\n')[0])
            if data_line_num == number_of_vars - 1:
                data.append(data_line)
                data_line = []

    f.close()

    # Rearrange data
    variables = sorted(config.LTspice_variable_numbering, key=config.LTspice_variable_numbering.__getitem__)
    variables = np.array(variables)[config.LTspice_preffered_sorting].tolist()
    data = np.array(data)[:, config.LTspice_preffered_sorting]

    # Write data to file
    try:
        f = open(output_path, 'w+')
    except IOError:
        logging.error('\nThe path specified for saving output data, \'' + config.LTspice_output_directory + '\', doesn\'t appear to exist.\nPlease check if the filepath set in \'config.py\' is correct.')
        exit(0)
    #//////////////////////f.write(output_header)
    f.write('\t'.join(variables) + '\n')
    for line in data:
        f.write('\t'.join(line) + '\n')
    f.close()

    size = os.path.getsize(output_path)
    logging.debug('CSV file created: ' + output_path + ' (' + str(size/1000) + ' kB)')



# ----------- Parameter controls ----------- #

def parse_parameter_file(filename):

    cmd_list = []
    param_file = open(filename, 'r')

    for line in param_file:
        line = line.split()
        if len(line) == 0:
            continue
        try:
            cmd = line[0]
            if cmd[0] == '#':
                continue
            elif cmd.lower() == 'set':
                parameter = line[1]
                value = line[2]
                cmd_list.append(('s', parameter, value))
            elif cmd.lower() == 'run':
                parameter = line[1]
                values = line[2:]
                cmd_list.append(('r', parameter, values))
            else:
                return None # Syntax error
        except IndexError:
            return None # Syntax error

    return cmd_list

def set_parameters(file_path, param, param_val, overwrite=False):
    f, abs_path = mkstemp()
    with open(abs_path,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                line_list = line.split(' ')
                if line_list[0] == 'TEXT':
                    for element_num, element in enumerate(line_list):
                        if element.split('=')[0] == param:
                            if param == config.LTspice_simtime_paramname:
                                line_list[element_num] = param + '=' + str(config.LTspice_simtime)
                            else:
                                line_list[element_num] = param + '=' + str(param_val)
                    if line_list[-1][-1] != '\n':
                        line_list[-1] = line_list[-1] + '\n'
                    new_file.write(' '.join(line_list))
                else:
                    new_file.write(line)
    os.close(f)
    if overwrite:
        os.remove(file_path)
        move(abs_path, file_path)
    else:
        move(abs_path, file_path[:-4] + '_generated.' + config.LTSpice_asc_filetype)

def get_parameters(file_path):
    output_list = []
    if config.LTSpice_asc_filetype == 'asc':
        f = open(file_path, 'r')
        for line in f:
            line_list = line.split()
            if line_list[0] == 'TEXT' and '!.param' in line_list:
                output_list.extend(line_list[line_list.index('!.param') + 1:])
        f.close()
    return output_list
