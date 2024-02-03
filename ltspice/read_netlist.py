import numpy as np
import csv

f = open("ltspice/test/netlist_v2.cir", "r")
input_lyr1 = 25 # with bias
num_of_bias = 2 
output_lyr1 = 10 # plus and minus
resistances_lyr1 = np.zeros([input_lyr1+1,output_lyr1])

input_lyr2 = 5 # with bias
output_lyr2 = 2 # plus and minus
resistances_lyr2 = np.zeros([input_lyr2 + 1,output_lyr2])

voltages = ['']*input_lyr1


itr = 0
for x in f:
    print(x)
    # read PWL voltages 
    num_of_res = input_lyr1 + num_of_bias
    #itr = 0
    if x.startswith("Vi") and itr <= num_of_res:
        row = x.split('    ')
        address = row[0].split('_')
        addr_row = int(address[1])
        voltages[addr_row-1] = row[3]
        itr += 1
    if x.startswith("Vb_001") and itr <= num_of_res:
        row = x.split('    ')
        address = row[0].split('_')
        addr_row = int(address[1])
        voltages.append(row[3])
        itr += 1
    if x.startswith("Vb_002") and itr <= num_of_res:
        row = x.split('    ')
        address = row[0].split('_')
        addr_row = int(address[1])
        voltages.append(row[3])
        itr += 1
    # read resistances of layer 1  .
    num_of_res += (1 + input_lyr1) * output_lyr1
    # itr = 0
    if x.startswith('Rh001') and itr <= num_of_res:
        row = x.split('    ')
        address = row[0].split('_')
        addr_row = int(address[1])
        address = row[2].split('_')
        addr_col = address[4]
        sign = addr_col[len(addr_col)-1]
        addr_col = addr_col[0:len(addr_col) - 1]
        addr_col = int(addr_col)
        if sign == '+' :
            resistances_lyr1[(addr_row-1),2*(addr_col-1)] = int(row[3])
        else:
            resistances_lyr1[(addr_row-1),2*(addr_col-1)+1] = int(row[3])
        itr += 1 
    if x.startswith('Rb_h001') and itr <= num_of_res:
        row = x.split('    ')
        addr_row = input_lyr1 + 1
        address = row[2].split('_')
        addr_col = address[4]
        sign = addr_col[len(addr_col)-1]
        addr_col = addr_col[0:len(addr_col) - 1]
        addr_col = int(addr_col)
        if sign == '+' :
            resistances_lyr1[(addr_row-1),2*(addr_col-1)] = int(row[3])
        else:
            resistances_lyr1[(addr_row-1),2*(addr_col-1)+1] = int(row[3])
        itr += 1
    # read resistances of layer 2.
    num_of_res += (1 + input_lyr2) * output_lyr2
    # itr = 0
    if x.startswith('Rh002') and itr < num_of_res:
        row = x.split('    ')
        address = row[0].split('_')
        addr_row = int(address[1])
        address = row[2].split('_')
        addr_col = address[4]
        sign = addr_col[len(addr_col)-1]
        addr_col = addr_col[0:len(addr_col) - 1]
        addr_col = int(addr_col)
        if sign == '+' :
            resistances_lyr2[(addr_row-1),2*(addr_col-1)] = int(row[3])
        else:
            resistances_lyr2[(addr_row-1),2*(addr_col-1)+1] = int(row[3])
        itr += 1 
    if x.startswith('Rb_h002') and itr <= num_of_res:
        row = x.split('    ')
        addr_row = input_lyr2 + 1
        address = row[2].split('_')
        addr_col = address[4]
        sign = addr_col[len(addr_col)-1]
        addr_col = addr_col[0:len(addr_col) - 1]
        addr_col = int(addr_col)
        if sign == '+' :
            resistances_lyr2[(addr_row-1),2*(addr_col-1)] = int(row[3])
        else:
            resistances_lyr2[(addr_row-1),2*(addr_col-1)+1] = int(row[3])
        itr += 1

# Write CSVs 
path = './data/'
with open(path + 'input_voltages.csv', 'w', newline='') as csvfile:
    #for voltage in voltages:
    csvfile.writelines(voltages)

with open(path + 'resistances_lyr1.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #for res in resistances_lyr1:
    spamwriter.writerows(resistances_lyr1)

with open(path + 'resistances_lyr2.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #for res in resistances_lyr1:
    spamwriter.writerows(resistances_lyr2)
print("Finish")