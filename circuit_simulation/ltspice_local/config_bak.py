from dataclasses import asdict, dataclass
from typing import Sequence

##########################################
## CONFIGURATION OF LTspice SIMULATIONS ##
##########################################

########## Coded by HA ##########
#################################
# Settings found in utils/settings.py
#################################
# The pulse duration for the input encoding (s)
sim_pulse_width: float = 3e-7
# Simulation initial latency before to start the first pulse (s)
sim_init_latency: float = 1e-9
# Whether the inference of the ML model should be simulated on a circuit or not.
simulate_circuit: bool = False
#################################
LTSpice_executable_path = 'C:\Program Files\ADI\LTspice\LTspice.exe'
patch_size_x: int = 8
patch_size_y: int = 8
hidden_layers_size: Sequence  = [20,10]
######## PATHs #######
LTspice_working_directory: str = 'data/ltspice_tmp_files/'
LTspice_spiceout_directory: str = LTspice_working_directory + 'spice/'
LTspice_data_directory: str = LTspice_working_directory + 'data/'
# required components directory
LTspice_required_directory: str = 'ltspice/spice/components/requiredSubCells'
# Leave blank if output should be writtten to root folder. The folder specified must be created manually if it doesn't exist.
LTspice_output_directory = LTspice_working_directory + 'simulation_results/'
######################
# The name of the circuit to be used in simulations
# It is important to set this parameter
# The best practice is to keep its default value  
LTSpice_asc_filetype : str = 'asc'
LTspice_wd_prefix: str = 'activ_v'   # Specifies the name prefix for the WD
LTSpice_asc_filename: str = LTspice_spiceout_directory + 'complete_circuit.asc'   # Specifies the circuit file name
LTspice_avail_activation_fn = ['column_activ_cmos', 'column_activ_comparator', 'column_activ_relu', 'column_activ_tia_relu', 'column_activ_cmos_v2', 'column_activ_cmos_v3']
LTspice_activation_fn_select: int = 5
LTspice_final_activation_fn_select: int = 5 # # Specifies the AF for the final stage
# Layers parameters 
LTspice_num_of_layers: int = 3 # Should be len(hidden_layers_size) + 1, if hidden_layers_size has a depth of 2 -> LTspice_num_of_layers = 3
LTspice_layer_dims: Sequence = [[65,20], [21,10],[11,1]] # Sizes with biases, will be modified in the code
# Setting the minimum vertical spacing between blocks in the circuit
LTspice_block_vspacing: int = 480 # should be multiple of 16
# The measurements to be saved from the simulation. The appropriate numbers can be found in the .raw file generated at simulation.
LTspice_variable_numbering = {'time': 0, 'final_out': 45}
# The ordering of the measurements to be saved in the output csv files can be changed below, by changing how the numbers are ordered.
# E.g. switch place of 0 and 1 if you want V_c to be placed left of time in the output csv files.
LTspice_preffered_sorting = [0, 1]
# Naming convention for output files. Can be numerical names, increasing in value as output is created.
# Parameter names gives the files the name of the parameter that it is run with.
# Assumes number by default. Set to 'parameter' or 'number'.
LTspice_output_data_naming_convention = 'parameter'
LTspice_simtime: str = '1u'
LTspice_simtime_paramname: str = 'simtime'
LTspice_overwrite_files = True
#################################



@dataclass(init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    # File path of the LTspice program installed on the system
    LTSpice_executable_path = 'C:\Program Files\ADI\LTspice\LTspice.exe'
    # The name of the circuit to be used in simulations
    # LTSpice_asc_filename = 'spice/Test_RNN_Sigmoid3_HA.asc'
    LTSpice_asc_filetype = 'asc'


    working_directory = 'ltspice/'  # Specifies the current WD
    spice_directory = working_directory + 'spice/'
    data_directory = working_directory + 'data/'
    wd_prefix = 'activ_v'   # Specifies the name prefix for the WD
    LTSpice_asc_filename = spice_directory + 'complete_circuit.asc'   # Specifies the circuit file name
    # required components directory
    required_directory = 'ltspice/spice/components/requiredSubCells'
    # Different alternatives for the auxiliary circuit
    activation_fn = ['column_activ_cmos', 'column_activ_comparator', 'column_activ_relu', 'column_activ_tia_relu', 'column_activ_cmos_v2', 'column_activ_cmos_v3']
    activation_fn_select = 5
    final_activation_fn_select = 5 # # Specifies the AF for the final stage

    # Layers parameters 
    num_of_layers: int = 3
    layer_dims = [[65,20], [21,10],[11,1]] # Sizes with biases 
    # Setting the minimum vertical spacing between blocks in the circuit
    block_vspacing: int = 480 # should be multiple of 16


    # activation_function = 'relu' # options {relu, cmos, comparator}

    # The measurements to be saved from the simulation. The appropriate numbers can be found in the .raw file generated at simulation.
    variable_numbering = {'time': 0, 'final_out': 45}
    # The ordering of the measurements to be saved in the output csv files can be changed below, by changing how the numbers are ordered.
    # E.g. switch place of 0 and 1 if you want V_c to be placed left of time in the output csv files.
    preffered_sorting = [0, 1]

    # Leave blank if output should be writtten to root folder. The folder specified must be created manually if it doesn't exist.
    output_data_path = 'ltspice/simulation_results/'
    # Naming convention for output files. Can be numerical names, increasing in value as output is created.
    # Parameter names gives the files the name of the parameter that it is run with.
    # Assumes number by default. Set to 'parameter' or 'number'.
    output_data_naming_convention = 'parameter'

configuration=Settings()