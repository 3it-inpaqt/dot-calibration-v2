##########################################
## CONFIGURATION OF LTspice SIMULATIONS ##
##########################################

# File path of the LTspice program installed on the system
#LTSpice_executable_path = 'C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe'
LTSpice_executable_path = 'C:\Program Files\ADI\LTspice\LTspice.exe'
# The name of the circuit to be used in simulations
# LTSpice_asc_filename = 'spice/Test_RNN_Sigmoid3_HA.asc'
LTSpice_asc_filetype = 'cir'

LTSpice_asc_filename = 'test/netlist.cir'

# The measurements to be saved from the simulation. The appropriate numbers can be found in the .raw file generated at simulation.
variable_numbering = {'time': 0, 'sum_h_out_002_001': 51}
# The ordering of the measurements to be saved in the output csv files can be changed below, by changing how the numbers are ordered.
# E.g. switch place of 0 and 1 if you want V_c to be placed left of time in the output csv files.
preffered_sorting = [0, 1]

# Leave blank if output should be writtten to root folder. The folder specified must be created manually if it doesn't exist.
output_data_path = 'data/'
# Naming convention for output files. Can be numerical names, increasing in value as output is created.
# Parameter names gives the files the name of the parameter that it is run with.
# Assumes number by default. Set to 'parameter' or 'number'.
output_data_naming_convention = 'parameter'
