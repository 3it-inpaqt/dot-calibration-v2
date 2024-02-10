import imp
from pickle import NONE
import sys, getopt
import nn_layer as nn_lyr
import simulation_tools
import config
import numpy as np
import generate_crossbar_asc as gasc
import generate_auxcircuit_asc as acir
import csv
import os
import shutil


diffx = 320
diffy = 160
#working_directory = config.working_directory
data_directory = config.LTspice_data_directory
file_name = 'complete_circuit'
num_of_rows_lyr1=26
num_of_cols_lyr1=10
num_of_rows_lyr2=6
num_of_cols_lyr2=2

def build_circuit():
    create_directory(config.LTspice_overwrite_files)
    dim = config.LTspice_layer_dims[0]
    input_voltages = ['']*(dim[0]+config.LTspice_num_of_layers-1)
    with open(data_directory + 'input_voltages.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
        itr = 0
        for row in spamreader:
            #if itr ==66:
            #    print('Breakpoint')
            if row == "":
                continue
            input_voltages[itr] = str(row).replace('[','').replace(']','').replace('\'','')
            itr += 1

    # Building Layers    
    lyrList = []
    i = int(0)
    current_vlocation = 16
    while i < config.LTspice_num_of_layers:
        # Getting Layer dimensions
        dim = config.LTspice_layer_dims[i]
        # Getting the resistances
        resistances = np.zeros([dim[0],dim[1]*2])
        with open(data_directory + 'resistances_lyr{lyr_num}.csv'.format(lyr_num = i+1), newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                itr = 0
                for row in spamreader:
                        resistances[itr,] = row
                        itr += 1
        if i == 0 and config.LTspice_num_of_layers == 1:
                lyr = nn_lyr.nnlayer(i,dim[0],dim[1], resistances, 'relu', input_voltages)
                lyr.isFirst = True
                lyr.hasNext = False
                lyr.hasPrveious = False
        elif i == 0 and config.LTspice_num_of_layers > 1:
                lyr = nn_lyr.nnlayer(i,dim[0],dim[1], resistances, 'relu', input_voltages)
                lyr.isFirst = True
                lyr.hasNext = True
                lyr.hasPrveious = False
        elif i == config.LTspice_num_of_layers - 1 and config.LTspice_num_of_layers > 1:
                lyr = nn_lyr.nnlayer(i,dim[0],dim[1], resistances, 'relu', None)
                lyr.hasNext = False
                lyr.isFirst = False
                lyr.hasPrveious = True
        else:
            lyr = nn_lyr.nnlayer(i,dim[0],dim[1], resistances, 'relu', None)
            lyr.hasNext = True
            lyr.isFirst = False
            lyr.hasPrveious = True
                
        lyr.build_layer()
        
        lyr.cb1.cirStartPosX = 0
        lyr.cb1.cirStartPosY = current_vlocation
        
        current_vlocation += config.LTspice_block_vspacing
        
        # Adding Auxiliary circuit aux1
        lyr.aux1.cirStartPosX = 0
        lyr.aux1.cirStartPosY = current_vlocation
        current_vlocation += config.LTspice_block_vspacing
        
        lyrList.append(lyr)

        i += 1
        

    # Generating the complete circuit
    stment = 'Version 4 \n'
    stment += 'SHEET 1 2360 680\n'
    
    for lyr in lyrList:
        # Adding the wires
        stment += lyr.print_wires()
    
    for lyr in lyrList:
        # Adding the pins
        if lyr.hasNext:
            NxtLyrVinNamePrefix = lyrList[lyr.idx_in_list + 1].cb1.VinNamePrefix + '[0:{end}]'.format(end=lyrList[lyr.idx_in_list + 1].cb1.num_of_rows - 2)
            stment += lyr.print_pins(NxtLyrVinNamePrefix)
        else:
            NxtLyrVinNamePrefix = lyr.aux1.VoutName
            stment += lyr.print_pins(NxtLyrVinNamePrefix) 
        
    for lyr in lyrList:
        # Adding the symbols
        stment += lyr.print_symbols()
        
    # Add input voltages 
    startX = -1500
    startY = 0
    vinSpacing = 192
    lyr = lyrList[0]
    for i in list(range(0,lyr.cb1.num_of_rows)):
        stment += 'SYMBOL voltage ' + str(startX) + ' ' + str(startY + (vinSpacing*i)) + ' R0\n'
        stment += 'SYMATTR InstName V' + str(i) + '\n'
        stment += 'SYMATTR Value ' + str(lyr.input_voltages[i]) + '\n'
        stment += 'FLAG ' + str(startX) + ' ' + str(startY + 96 + (vinSpacing*i)) + ' 0\n'
        stment += 'WIRE ' + str(startX) + ' ' + str(startY + (vinSpacing*i) + 16) + ' ' + str(startX) + ' ' + str(startY + (vinSpacing*i) - 48) + '\n'
        stment += 'WIRE ' + str(startX) + ' ' + str(startY + (vinSpacing*i) - 48) + ' ' + str(startX + 48) + ' ' + str(startY + (vinSpacing*i) - 48) + '\n'
        stment += 'FLAG ' + str(startX + 48) + ' ' + str(startY + (vinSpacing*i) - 48) + ' ' + lyr.cb1.VinNamePrefix + '[' + str(i) + ']\n'
    i += 1
    while i < len(lyrList[0].input_voltages):
        lyr_idx = i - lyrList[0].cb1.num_of_rows + 1
        lyr = lyrList[lyr_idx]
        # Adding the bias voltage for every layer
        stment += 'SYMBOL voltage ' + str(startX) + ' ' + str(startY + (vinSpacing*i)) + ' R0\n'
        stment += 'SYMATTR InstName V' + str(i) + '\n'
        stment += 'SYMATTR Value ' + str(input_voltages[i]) + '\n'
        stment += 'FLAG ' + str(startX) + ' ' + str(startY + 96 + (vinSpacing*i)) + ' 0\n'
        stment += 'WIRE ' + str(startX) + ' ' + str(startY + (vinSpacing*i) + 16) + ' ' + str(startX) + ' ' + str(startY + (vinSpacing*i) - 48) + '\n'
        stment += 'WIRE ' + str(startX) + ' ' + str(startY + (vinSpacing*i) - 48) + ' ' + str(startX + 48) + ' ' + str(startY + (vinSpacing*i) - 48) + '\n'
        stment += 'FLAG ' + str(startX + 48) + ' ' + str(startY + (vinSpacing*i) - 48) + ' ' + lyr.cb1.VinNamePrefix + '[' + str(lyr.cb1.num_of_rows-1) + ']\n'
        i += 1
    # Adding transient simulation configuration
    stment += 'TEXT -1008 -240 Left 2 !.param simtime \n'
    stment += 'TEXT -1008 -208 Left 2 !.tran {simtime} \n'
    #print(os.path.join(os. getcwd(),'ltspice/spice/components/180nm_bulk.txt'))
    stment += 'TEXT -1008 -272 Left 2 !.include "' + os.path.join(os. getcwd(),'ltspice/spice/components/180nm_bulk.txt') + '"\n'
    stment += 'TEXT -1008 -272 Left 2 !.include "' + os.path.join(os. getcwd(),'ltspice/spice/components/OTA1.sub') + '"\n'

    write_to_ascfile(stment)
    return config.LTspice_spiceout_directory + file_name + '.asc'

def write_to_ascfile(stment):
    circuit_file_name = config.LTspice_spiceout_directory + file_name + '.asc'
    with open(circuit_file_name,'w') as nf :
        nf.write(stment)
    return

def create_directory(overwrite: bool = False):
     if overwrite:
         wd = '{parent}{gwd}/'.format(parent = config.LTspice_spiceout_directory, gwd = config.LTspice_wd_prefix)
         if os.path.exists(wd):
            shutil.rmtree(wd)
     else:
        subfolders= [f.path for f in os.scandir(config.LTspice_spiceout_directory) if f.is_dir() and f.name.startswith(config.LTspice_wd_prefix) ]
        gwd = '{wdp}{idx}'.format(wdp = config.LTspice_wd_prefix, idx = len(subfolders))
        wd = '{parent}{gwd}/'.format(parent = config.LTspice_spiceout_directory, gwd = gwd)
     #os.mkdir(wd)
     config.LTspice_spiceout_directory = wd
     config.LTSpice_asc_filename = config.LTspice_spiceout_directory + 'complete_circuit.asc'
     # copying sub cells
     shutil.copytree(config.LTspice_required_directory, wd, dirs_exist_ok=True)
     shutil.copyfile('ltspice/spice/components/simulation_parameters.txt', config.LTspice_data_directory + '/simulation_parameters.txt')
     #cwd = os.listdir(config.working_directory)
     #print(config.working_directory)
     return 
