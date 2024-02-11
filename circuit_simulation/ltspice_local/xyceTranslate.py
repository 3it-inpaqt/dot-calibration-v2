#import resource
import os
from utils.settings import settings as config
from utils.logger import logger as logging


def translate_netlist2xyce(input_file_name: str = "data/ltspice_tmp_files/simulation_results/complete_circuit_generated.net"):
    output_file_name = input_file_name[0:len(input_file_name)-4] + "_xyce.net"
    logging.debug(f'Translate: File Size is {os.stat(input_file_name).st_size / (1024 * 1024)} MB')

    input_txt_file = open(input_file_name,'r')
    output_txt_file = open(output_file_name,'w')

    count = 0
    count_out = 0
    start_idx = 0
    end_idx = 0
    in_cbar = False
    in_aux = False
    cbar_dimensions = [0,1]
    num_cbars = 0
    cbar_idx = -1
    simtime = 0
    write_line = True

    for line in input_txt_file:
        # we can process file line by line here, for simplicity I am taking count of lines
        count += 1
        if line.startswith("*"):
            write_line = False
        elif line.startswith("XX_cbar"):
            num_cbars += 1
            start_idx = line.index("_") + len("_cbar_")
            end_idx = line.index("by") 
            lineSubStr = line[start_idx:end_idx]
            cbar_dimensions.append(int(lineSubStr))
            start_idx = end_idx + len("by")
            end_idx = start_idx + line[int(start_idx):len(line)].index(" ")
            lineSubStr = line[start_idx:end_idx]
            cbar_dimensions.append(int(lineSubStr))
            write_line = True
        elif line.startswith("XX_aux_"):
            write_line = True
        elif line.startswith(".subckt x_cbar"):
            cbar_idx += 1
            write_line = True
           # in_cbar = True
        elif line.startswith("r_"):
            if in_cbar:
                line_copy = line
                line_copy = line_copy.replace("r_","Rcbar" +str(cbar_dimensions[0 + 2*cbar_idx])+"by"+str(cbar_dimensions[1 + 2*cbar_idx]) + "_")
                line_copy = line_copy.replace("Vin[","Vin_cb" +str(cbar_dimensions[0 + 2*cbar_idx])+"by"+str(cbar_dimensions[1 + 2*cbar_idx]) + "[")
                line_copy = line_copy.replace("Vout_","Vout_cb" +str(cbar_dimensions[0 + 2*cbar_idx])+"by"+str(cbar_dimensions[1 + 2*cbar_idx]) + "_")
                line = line_copy
            write_line = True
        elif line.startswith(".ends x_cbar"):
            cbar_name = "x_cbar_" + str(cbar_dimensions[0 + 2*cbar_idx]) + "by" + str(cbar_dimensions[1 + 2*cbar_idx]) 
            write_line = True
            in_cbar = False
        elif line.startswith(".subckt x_aux"):
            #in_aux = True
            write_line = True
        elif line.startswith(".ends x_aux"):
            #in_aux = False
            write_line = True
        elif line.startswith("XX"):
            if in_aux:
                line_copy = line
                line_copy = line_copy.replace("XX","XXcbar" +str(cbar_dimensions[0 + 2*cbar_idx])+"by"+str(cbar_dimensions[1 + 2*cbar_idx]) + "_")
                line_copy = line_copy.replace("TIA_H_IN_","Vout_cb" +str(cbar_dimensions[0 + 2*cbar_idx])+"by"+str(cbar_dimensions[1 + 2*cbar_idx]) + "_")
            
                if cbar_idx < (len(cbar_dimensions)/2 - 1):
                    line_copy = line_copy.replace("H_ACT_OUT[","Vin_cb" +str(cbar_dimensions[0 + 2*(cbar_idx+1)])+"by"+str(cbar_dimensions[1 + 2*(cbar_idx+1)]) + "[")
                else:
                    line_copy = line_copy.replace("H_ACT_OUT[","Vfinal[")
                line = line_copy
            write_line = True
        elif line.find(".model NMOS NMOS") != -1 :
            write_line = False
        elif line.find(".model PMOS PMOS") != -1 :
            write_line = False
        elif line.find("standard.mos") != -1 :
            write_line = False
        elif line.find(".param") != -1 :
            line_copy = line.split(" ")
            for s in line_copy:
                if s.find("simtime") != -1:
                    s = s.split("=")
                    simtime = s[1].strip()
            write_line = True
        elif line.find(".tran") != -1 :
            line = line.replace("{simtime}","0 "+str(simtime))
            file_name1 = str(output_file_name.split('/')[-1])
            if config.xyce_output_filetype.startswith('csv'):
               line += ".print tran format=csv file=" + file_name1[:-4] + ".csv  v(*) i(*)\n"
            elif config.xyce_output_filetype.startswith('prn'):
               line += ".print tran v(*) i(*)\n"
            write_line = True
        elif line.find("UniversalOpAmp1.lib") != -1 :
            write_line = False
        elif line.find("backanno") != -1 :
            write_line = False
        else:
            write_line =True
        if write_line:
            output_txt_file.write(line)
            count_out += 1

    input_txt_file.close()
    output_txt_file.close()

    logging.debug(f'Translate: Number of Lines in the input file is {count}')
    logging.debug(f'Translate: Number of Lines copied to XYCE is {count_out}')

