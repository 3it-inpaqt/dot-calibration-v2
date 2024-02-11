from pickle import NONE
import sys, getopt
import circuit_simulation.ltspice_local.simulation_tools as simulation_tools
from utils.settings import settings as config
import numpy as np
import math


diffx = 320
diffy = 160
border_hedge = 300
border_vedge = 40
file_name = 'X_cbar'

def build_crossbar (num_of_rows=16,num_of_cols=10, resistances = None):
    stment = 'Version 4 \nSHEET 1 2360 680\n'
    if resistances is not None:
        a, b = resistances.shape
        if a != num_of_rows or b != num_of_cols:
            return None;
    else:
        return None
    posX = 0 
    posY = 0
    stment += connect_wires(num_of_rows,num_of_cols)
    stment += place_io_pins(num_of_rows,num_of_cols)
    for row in list(range(0, num_of_rows)) :    
        for col in list(range(0, num_of_cols)) : 
            stment += create_resistor(str(resistances[row,col]), 'r_'+str(row)+'_'+str(col),'R0',posX,posY)
            posX += diffx
            if (col + 1) % 2 == 0 :
                posX += diffx
        posY  += diffy
        posX = 0
        
    asc_file_name = file_name + '_' + str(num_of_rows) + 'by' + str(num_of_cols)
    with open(config.LTspice_spiceout_directory + asc_file_name + '.asc','w') as nf :
        nf.write(stment)

    create_asy(num_of_rows,num_of_cols)
    return asc_file_name

def create_resistor(restistance='R',name = 'R', model = 'R0', x = 0,y = 0) :
    stment = 'SYMBOL res ' + str(x) + ' ' + str(y) + ' ' + model + '\n' + 'SYMATTR InstName ' + name + '\n'\
        'SYMATTR Value ' + restistance + '\n'
    return stment

def connect_wires(num_of_rows=16,num_of_cols=10):
    # Connect the rows 
    stment = ''
    for row in list(range(0, num_of_rows)) : 
        startX = -1*diffy
        startY = 16 + (row * diffy)
        if num_of_cols%2 == 0:
          endX = 16+((num_of_cols-1 +int(num_of_cols/2) - 1)*diffx)
        else:
            endX = 16+((num_of_cols-1 +int(num_of_cols/2))*diffx)
        endY = startY
        stment += 'WIRE ' + str(startX) + ' ' +  str(startY) + ' ' + str(endX) + ' ' + str(endY) + '\n'
    for row in list(range(0, num_of_rows)) :
        for col in list(range(0, num_of_cols)) :
            startX = 16+((col +int(col/2))*diffx)
            startY = 16 + 80 + row*diffy
            endX = int(diffx/2) + startX
            endY = startY
            stment += 'WIRE ' + str(startX) + ' ' + str(startY) + ' ' + str(endX) + ' ' + str(endY) +'\n'
            startX = endX
            endX = startX
            endY = 16 + 80 + (1 + row)*diffy
            stment += 'WIRE ' + str(startX) + ' ' + str(startY) + ' ' + str(endX) + ' ' + str(endY) +'\n'
    return stment

def place_io_pins(num_of_rows=16,num_of_cols=10):
    stment = ''
    for row in list(range(0, num_of_rows)) : 
        #print(str(row) + ' \n')
        posX = -1*diffy
        posY = 16 + (row * diffy)
        pinName = 'Vin' + '[' + str(row) +']'
        pinType = 'In'
        stment += 'FLAG ' + str(posX) + ' ' +  str(posY) + ' ' + pinName + '\n'
        stment += 'IOPIN ' + str(posX) + ' ' +  str(posY) + ' ' + pinType + '\n'
    for col in list(range(0, num_of_cols)) : 
        #print(str(row) + ' \n')
        posX = 16 + ((col +int(col/2))*diffx) + int(diffx/2)
        posY = 16 + 80 + (num_of_rows)*diffy
        if col % 2 == 0:
            pinName = 'Vout_' + '+[' + str(int(col/2)) +']'
        else:
            pinName = 'Vout_' + '-[' + str(int((col-1)/2)) +']'
        pinType = 'Out'
        stment += 'FLAG ' + str(posX) + ' ' +  str(posY) + ' ' + pinName + '\n'
        stment += 'IOPIN ' + str(posX) + ' ' +  str(posY) + ' ' + pinType + '\n'
    return stment

def create_asy(num_of_rows=16,num_of_cols=10):
    stment = 'Version 4\n'
    stment += 'SymbolType BLOCK\n'
    stment += 'RECTANGLE Normal -' + str(border_hedge) + ' ' + '-' + str(border_vedge) + ' ' + str(border_hedge) + ' ' + str(border_vedge) + '\n'
    stment += 'WINDOW 0 8 -' + str(border_vedge) + ' Bottom 2\n'
    #stment += 'SYMATTR Prefix CB\n'
    stment += 'PIN -' + str(border_hedge) + ' 0 LEFT 8\n'
    stment += 'PINATTR PinName Vin' + '[0:' + str(num_of_rows-1) + ']\n'
    stment += 'PINATTR SpiceOrder 1\n'
    stment += 'PIN ' + str(border_hedge) + ' 18 RIGHT 8\n'
    if num_of_cols%2 == 0 :
        output_len = int(num_of_cols/2)
    else: 
        output_len = num_of_cols 
    stment += 'PINATTR PinName Vout_' + '+[0:' + str(output_len-1) + ']\n'
    stment += 'PINATTR SpiceOrder 2\n'
    stment += 'PIN ' + str(border_hedge) + ' -18 RIGHT 8\n'
    stment += 'PINATTR PinName Vout_' + '-[0:' + str(output_len-1) + ']\n'
    stment += 'PINATTR SpiceOrder 3\n'
    asy_file_name = file_name + '_' + str(num_of_rows) + 'by' + str(num_of_cols)
    with open(config.LTspice_spiceout_directory + asy_file_name + '.asy','w') as nf :
        nf.write(stment)
    return asy_file_name


class cross_bar():
    diffx = 320
    diffy = 160
    border_hedge = 300
    border_vedge = 40
    working_directory = config.LTspice_spiceout_directory
    block_file_name = 'X_cb'
    num_of_rows=16
    num_of_cols=10
    cirStartPosX = 0
    cirStartPosY = 0
    VinName = 'Vin'
    VoutName = 'Vout'
    VoutNameP = 'Vout'
    VoutNameN = 'Vout'
    VinNamePrefix = 'Vin'
    VoutNamePrefix = 'Vout'
    input_voltages = None
    resistances = np.zeros([num_of_rows,num_of_cols])
    def __init__(mysillyobject, num_of_rows, num_of_cols, resistances):
        mysillyobject.num_of_rows = num_of_rows
        mysillyobject.num_of_cols = num_of_cols
        mysillyobject.resistances = resistances
        if num_of_cols%2 == 0 :
            output_len = int(num_of_cols/2)
        else: 
            output_len = num_of_cols 
        mysillyobject.VinNamePrefix = 'Vin_cb' + str(num_of_rows) + 'by' + str(num_of_cols)
        mysillyobject.VoutNamePrefix = 'Vout_cb' + str(num_of_rows) + 'by' + str(num_of_cols)
        mysillyobject.VinName = 'Vin_cb' + str(num_of_rows) + 'by' + str(num_of_cols) + '[0:' + str(num_of_rows-1) +']'
        mysillyobject.VoutName = 'Vout_cb' + str(num_of_rows) + 'by' + str(num_of_cols) + '[0:' + str(num_of_cols-1) +']'
        mysillyobject.VoutNameP = 'Vout_cb' + str(num_of_rows) + 'by' + str(num_of_cols) + '_+[0:' + str(output_len-1) +']'
        mysillyobject.VoutNameN = 'Vout_cb' + str(num_of_rows) + 'by' + str(num_of_cols) + '_-[0:' + str(output_len-1) +']'
