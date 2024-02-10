import matplotlib.pyplot as plt 
import csv
import numpy as np

from numpy import double 

col_names=['TIME','V(VIN_CB65BY40[22])','V(XX_AUX_20BY10:XX5:XX5:XU1:4)']
x = [] 
y = [] 
z = []

with open('ltspice/spice/activ_v0/complete_circuit_generated_xyce.csv','r') as csvfile: 
	plots = csv.DictReader(csvfile, delimiter = ',') 
	i = 0
	for row in plots: 
		x.append(row[col_names[0]])
		# print(col_names[1].upper())
		y.append(double(row[col_names[1].upper()]))
		z.append(double(row[col_names[2].upper()]))

x = double(x)
x = x*1e6
plt.plot(x, y, color = 'g', label = col_names[1]) 
plt.plot(x, z, color = 'r', label = col_names[2])
plt.xlabel('Time') 
plt.xticks(rotation=45, ha='right')
plt.xticks(np.arange(min(x), max(x), (max(x) - min(x))/10))
plt.ylabel('Voltage') 
plt.title('Plot') 
plt.legend() 
plt.show() 
