##########################################
## CONFIGURATION OF LTspice SIMULATIONS ##
##########################################

# File path of the LTspice program installed on the system
#LTSpice_executable_path = 'C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe'
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
activation_fn = ['column_activ_cmos', 'column_activ_comparator', 'column_activ_relu', 'column_activ_tia_relu', 'column_activ_cmos_v2']
activation_fn_select = 4
final_activation_fn_select = 4 # # Specifies the AF for the final stage

# Layers parameters 
num_of_layers: int = 2
layer_dims = [[26,5], [6,1]] # Sizes with biases 
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

