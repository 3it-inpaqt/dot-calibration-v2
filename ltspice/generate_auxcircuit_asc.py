from pickle import NONE
import sys, getopt
import simulation_tools
import config
import numpy as np
import math


diffx = 0
diffy = 96
border_hedge = 208
border_vedge = 40
file_name = 'X_aux'


def build_circuit (num_of_rows=16,num_of_cols=10, hasNext = False):
    stment = 'Version 4 \nSHEET 1 880 680\n'
    posX = 0 
    posY = 0
    stment += connect_wires(num_of_cols)
    stment += place_io_pins(num_of_cols)
    if hasNext:
        activationFn = config.LTspice_activation_fn_select
    else:
        activationFn = config.LTspice_final_activation_fn_select

    for row in list(range(0, num_of_cols)) :    
        stment += 'SYMBOL {activ} 240 {posY} R0\n'.format(activ = config.LTspice_avail_activation_fn[activationFn], posY = int((row*128) + 96))
        stment += 'SYMATTR InstName X{row}\n'.format(row = row)
        
    asc_file_name = file_name + '_' + str(num_of_rows) + 'by' + str(num_of_cols)
    with open(config.LTspice_spiceout_directory + asc_file_name + '.asc','w') as nf :
        nf.write(stment)

    create_asy(num_of_rows,num_of_cols)
    return asc_file_name


def connect_wires(num_of_cols=10):
    # Connect the rows 
    stment = ''
    startY = 80
    for row in list(range(0, num_of_cols)) : 
        yy = (row*128) + 80
        stment += 'WIRE 48 {posY} -16 {posY}\n'.format(posY = str(yy))
        stment += 'WIRE 48 {posY} -16 {posY}\n'.format(posY = str(32 + yy))
        stment += 'WIRE 512 {posY} 448 {posY}\n'.format(posY = str(16 + yy))
    return stment

def place_io_pins(num_of_cols=10):
    stment = ''
    startY = 80
    for row in list(range(0, num_of_cols)) : 
        yy = (row*128) + 80
        stment += 'FLAG -16 {posY} TIA_H_IN_+[{row}]\n'.format(posY = str(yy),row = row)
        stment += 'IOPIN -16 {posY} IN\n'.format(posY = str(yy))
        stment += 'FLAG -16 {posY} TIA_H_IN_-[{row}]\n'.format(posY = str(32+yy),row = row)
        stment += 'IOPIN -16 {posY} IN\n'.format(posY = str(32 + yy))
        stment += 'FLAG 512 {posY} H_ACT_OUT[{row}]\n'.format(posY = str(16+yy),row = row)
        stment += 'IOPIN 512 {posY} OUT\n'.format(posY = str(16 + yy))
    return stment

def create_asy(num_of_rows=16,num_of_cols=10):
    stment = 'Version 4\n'
    stment += 'SymbolType BLOCK\n'
    stment += 'RECTANGLE Normal 224 64 -224 -64\n'
    stment += 'WINDOW 0 16 -72 Bottom 2\n'
    stment += 'PIN -224 -16 LEFT 8\n'
    stment += 'PINATTR PinName tia_h_in_+[0:{end}]\n'.format(end = num_of_cols - 1)
    stment += 'PINATTR SpiceOrder 1\n'
    stment += 'PIN -224 16 LEFT 8\n'
    stment += 'PINATTR PinName tia_h_in_-[0:{end}]\n'.format(end = num_of_cols - 1)
    stment += 'PINATTR SpiceOrder 2\n'
    stment += 'PIN 224 0 RIGHT 8\n'
    stment += 'PINATTR PinName h_act_out[0:{end}]\n'.format(end = num_of_cols - 1)
    stment += 'PINATTR SpiceOrder 3\n'
    asy_file_name = file_name + '_' + str(num_of_rows) + 'by' + str(num_of_cols)
    with open(config.LTspice_spiceout_directory + asy_file_name + '.asy','w') as nf :
        nf.write(stment)
    return asy_file_name


class aux_circuit():
    working_directory = config.LTspice_spiceout_directory
    block_file_name = 'X_aux'
    num_of_rows=16
    num_of_cols=10
    cirStartPosX = 0
    cirStartPosY = 0
    VoutName = 'h_act_out'
    VinNameP = 'tia_h_in_+'
    VinNameN = 'tia_h_in_-'
    border_hedge = 224
    border_vedge = 64
    def __init__(mysillyobject, num_of_rows, num_of_cols):
        mysillyobject.num_of_rows = num_of_rows
        mysillyobject.num_of_cols = num_of_cols
        mysillyobject.VoutName = 'h_act_out[0:{end}]'.format(end = num_of_cols - 1)
        mysillyobject.VinNameP = 'tia_h_in_+[0:{end}]'.format(end = num_of_cols - 1)
        mysillyobject.VinNameN = 'tia_h_in_-[0:{end}]'.format(end = num_of_cols - 1)
