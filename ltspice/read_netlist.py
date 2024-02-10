from tkinter import simpledialog
from typing import Sequence
import numpy as np
import csv
import config as config
import shutil

def generate_CSVs_from_netlist(netlist_path: str = 'circuit_simulation/netlist.CIR', CSV_out_directory: str = 'ltspice_tmp_files/data/'):
    input_lyrs = []
    num_of_bias = len(config.LTspice_layer_dims)
    for i in range(num_of_bias):
        tmp = config.LTspice_layer_dims[i]
        input_lyrs.append([tmp[0] - 1, tmp[1]*2])
    
    lyr_1st = input_lyrs[0]
    line_counter = read_PWL_voltages_from_netlist(lyr_1st[0],
                                                  num_of_bias,
                                                  0,
                                                  netlist_path=netlist_path,
                                                  CSV_out_directory=CSV_out_directory)    
    for i in range(num_of_bias):
        line_counter = read_layer_resistances_from_netlist(i,
                                                        line_counter,
                                                        layer_size = input_lyrs[i],
                                                        netlist_path=netlist_path,
                                                        CSV_out_directory=CSV_out_directory)
    return 

def read_PWL_voltages_from_netlist(input_size:int, 
                                   num_of_layers: int, 
                                   line_counter: int,
                                   netlist_path: str = 'circuit_simulation/netlist.CIR', 
                                   CSV_out_directory: str = 'ltspice_tmp_files/data/') -> int:
    voltages = ['']*input_size
    f = open(netlist_path, "r")
    itr = 0
    num_of_res = input_size + num_of_layers
    for x in f:
        #print(x)
        line_counter += 1
        # read PWL voltages 
               
        if x.startswith("Vi") and itr <= num_of_res:
            row = x.split('    ')
            address = row[0].split('_')
            addr_row = int(address[1])
            voltages[addr_row-1] = row[3]
            if addr_row == 1:
                config.LTspice_simtime = row[3].split(' ')[-2]
            itr += 1
        if x.startswith("Vb_") and itr <= num_of_res:
            row = x.split('    ')
            address = row[0].split('_')
            addr_row = int(address[1])
            voltages.append(row[3])
            itr += 1

        if itr == num_of_res:
            with open(CSV_out_directory + 'input_voltages.csv', 'w', newline='') as csvfile:
                #for voltage in voltages:
                csvfile.writelines(voltages)
            return line_counter
    return line_counter

def read_layer_resistances_from_netlist(
                                   layer_idx: int, 
                                   line_counter: int,
                                   layer_size: Sequence = [0,0], 
                                   netlist_path: str = 'circuit_simulation/netlist.CIR', 
                                   CSV_out_directory: str = 'ltspice_tmp_files/data/') -> int:
    resistances = np.zeros([layer_size[0] + 1, layer_size[1]])
    f = open(netlist_path, "r")
    itr = 0
    num_of_res = (1 + layer_size[0]) * layer_size[1]
    this_line_counter = 0 
    for x in f:
        this_line_counter += 1
        if this_line_counter <= line_counter:
            continue
        str_check = 'Rh00'+str(layer_idx+1)
        if x.startswith(str_check) and itr <= num_of_res:
            row = x.split('    ')
            address = row[0].split('_')
            addr_row = int(address[1])
            address = row[2].split('_')
            addr_col = address[4]
            sign = addr_col[len(addr_col)-1]
            addr_col = addr_col[0:len(addr_col) - 1]
            addr_col = int(addr_col)
            if sign == '+' :
                resistances[(addr_row-1),2*(addr_col-1)] = float(row[3])
            else:
                resistances[(addr_row-1),2*(addr_col-1)+1] = float(row[3])
            itr += 1 
        str_check = 'Rb_h00'+str(layer_idx+1)
        if x.startswith(str_check) and itr <= num_of_res:
            row = x.split('    ')
            addr_row = layer_size[0] + 1
            address = row[2].split('_')
            addr_col = address[4]
            sign = addr_col[len(addr_col)-1]
            addr_col = addr_col[0:len(addr_col) - 1]
            addr_col = int(addr_col)
            if sign == '+' :
                resistances[(addr_row-1),2*(addr_col-1)] = float(row[3])
            else:
                resistances[(addr_row-1),2*(addr_col-1)+1] = float(row[3])
            itr += 1
            if itr == num_of_res:
                with open(CSV_out_directory + 'resistances_lyr' + str(layer_idx + 1) + '.csv', 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    #for res in resistances_lyr1:
                    spamwriter.writerows(resistances)
                #print(this_line_counter)
                return this_line_counter
    return this_line_counter

def copy_params_file(in_file: str, out_file: str):
    return

