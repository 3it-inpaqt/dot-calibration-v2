import circuit_simulation.ltspice_local.generate_crossbar_asc as gasc
import circuit_simulation.ltspice_local.generate_auxcircuit_asc as acir

class nnlayer(object):
    num_of_rows: int = 16
    num_of_cols: int = 10
    input_voltages = None
    resistances = None
    activation_fn: str = 'relu'
    working_dir = ''
    cb1  =  None
    aux1 = None
    isFirst = False
    hasNext = False
    hasPrveious = False
    idx_in_list = 0
    def __init__(mysillyobject, idx_in_list, num_of_rows, num_of_cols, resistances, activation_fn, input_voltages):
        mysillyobject.idx_in_list = idx_in_list
        mysillyobject.num_of_rows = num_of_rows
        mysillyobject.num_of_cols = num_of_cols
        mysillyobject.resistances = resistances
        mysillyobject.activation_fn = activation_fn
        mysillyobject.input_voltages = input_voltages
        mysillyobject.cb1 = gasc.cross_bar(num_of_rows,num_of_cols * 2,resistances)
        mysillyobject.aux1 = acir.aux_circuit(mysillyobject.cb1.num_of_cols, int(mysillyobject.cb1.num_of_cols/2))

    def build_layer(mysillyobject):
        mysillyobject.cb1.block_file_name = gasc.build_crossbar(mysillyobject.cb1.num_of_rows, mysillyobject.cb1.num_of_cols, mysillyobject.cb1.resistances)        
        mysillyobject.aux1.block_file_name = acir.build_circuit(mysillyobject.cb1.num_of_cols,int(mysillyobject.cb1.num_of_cols/2),mysillyobject.hasNext)
        return

    def print_wires(mysillyobject):
        # Adding cb1 wires
        stment = ''
        stment += 'WIRE ' + str(mysillyobject.cb1.cirStartPosX - mysillyobject.cb1.border_hedge - 16) + ' ' + str(mysillyobject.cb1.cirStartPosY) +\
            ' ' + str(mysillyobject.cb1.cirStartPosX - mysillyobject.cb1.border_hedge) + ' ' + str(mysillyobject.cb1.cirStartPosY) +  '\n'
        stment += 'WIRE ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge + 16) + ' ' + str(mysillyobject.cb1.cirStartPosY + 18) +\
            ' ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge) + ' ' + str(mysillyobject.cb1.cirStartPosY + 18) +  '\n'
        stment += 'WIRE ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge + 16) + ' ' + str(mysillyobject.cb1.cirStartPosY - 18) +\
            ' ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge) + ' ' + str(mysillyobject.cb1.cirStartPosY - 18) +  '\n'

        # Adding aux1 wires
        stment += 'WIRE ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge - 16) + ' ' + str(mysillyobject.aux1.cirStartPosY-16) +\
            ' ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge) + ' ' + str(mysillyobject.aux1.cirStartPosY-16) +  '\n'
        stment += 'WIRE ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge - 16) + ' ' + str(mysillyobject.aux1.cirStartPosY+16) +\
            ' ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge) + ' ' + str(mysillyobject.aux1.cirStartPosY+16) +  '\n'
        stment += 'WIRE ' + str(mysillyobject.aux1.cirStartPosX + mysillyobject.aux1.border_hedge + 16) + ' ' + str(mysillyobject.aux1.cirStartPosY) +\
            ' ' + str(mysillyobject.aux1.cirStartPosX + mysillyobject.aux1.border_hedge) + ' ' + str(mysillyobject.aux1.cirStartPosY) +  '\n'
        return stment
    
    def print_pins(mysillyobject,NxtLyrVinNamePrefix):
        # Adding cb1 pins
        stment = ''
        stment += 'FLAG ' + str(mysillyobject.cb1.cirStartPosX - mysillyobject.cb1.border_hedge - 16) + ' ' + str(mysillyobject.cb1.cirStartPosY) +\
            ' ' + mysillyobject.cb1.VinName + '\n'
        stment += 'FLAG ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge + 16) + ' ' + str(mysillyobject.cb1.cirStartPosY + 18) +\
            ' ' + mysillyobject.cb1.VoutNameP + '\n'
        stment += 'FLAG ' + str(mysillyobject.cb1.cirStartPosX + mysillyobject.cb1.border_hedge + 16) + ' ' + str(mysillyobject.cb1.cirStartPosY - 18) +\
            ' ' + mysillyobject.cb1.VoutNameN + '\n'
        # Adding aux1 pins
        stment += 'FLAG ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge - 16) + ' ' + str(mysillyobject.aux1.cirStartPosY-16) +\
            ' ' + mysillyobject.cb1.VoutNameP + '\n'
        stment += 'FLAG ' + str(mysillyobject.aux1.cirStartPosX - mysillyobject.aux1.border_hedge - 16) + ' ' + str(mysillyobject.aux1.cirStartPosY+16) +\
            ' ' + mysillyobject.cb1.VoutNameN + '\n'
        stment += 'FLAG ' + str(mysillyobject.aux1.cirStartPosX + mysillyobject.aux1.border_hedge + 16) + ' ' + str(mysillyobject.aux1.cirStartPosY) +\
            ' ' + '{end}\n'.format(end=NxtLyrVinNamePrefix)
        return stment
    
    def print_symbols(mysillyobject):
        # Adding cb1 symbol
        stment = ''        
        stment += 'SYMBOL ' + mysillyobject.cb1.block_file_name + ' ' + str(mysillyobject.cb1.cirStartPosX) + ' ' + str(mysillyobject.cb1.cirStartPosY) + ' R0\n'
        stment += 'SYMATTR InstName ' + mysillyobject.cb1.block_file_name + '\n'
        # Adding aux1 symbol
        stment += 'SYMBOL ' + mysillyobject.aux1.block_file_name + ' ' + str(mysillyobject.aux1.cirStartPosX) + ' ' + str(mysillyobject.aux1.cirStartPosY) + ' R0\n'
        stment += 'SYMATTR InstName ' + mysillyobject.aux1.block_file_name + '\n'
        return stment
    
    




