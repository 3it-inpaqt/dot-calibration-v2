# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
import os
import numpy as np
import matplotlib.pyplot as plt
import ltspice
import subprocess
import torch

sample_data_folder = os.getcwd()

sample_data_raw_file = os.path.join(
    sample_data_folder,"plot", "Test_RNN_Sigmoid3_HA_xyce.raw"
)

print('# Loading continuous data')
l = ltspice.Ltspice(sample_data_raw_file)
l.parse()
time = l.getVariableNames()
print(time)
time = l.get_time()
V_source = l.get_data('V(IN)')
V_en = l.get_data('V(OUT)')
#V_en2 = l.get_data('V(VB)')
#V_cap = l.get_data('V(OUT)')
startidx=1
fig, axs = plt.subplots(1)
#fig.suptitle('Vertically stacked subplots')
axs.plot(V_source[startidx:], V_en[startidx:],label="SPICE")
axs.label_outer()
axs.set(ylabel='V(VP)')
#startidx=15
#axs[1].plot(time[startidx:], V_en[startidx:])
#axs[1].label_outer()
#axs[1].set(ylabel='V(out)')
#axs[2].plot(time, V_en2)
#axs[2].label_outer()
#axs[2].set(ylabel='V(EN2)')
#axs[3].plot(time, V_cap)
#axs[3].set(ylabel='VCAP',xlabel='Time (us)')
#plt.show()



# Create input tensor
x = torch.linspace(-10, 10, 100)

# Apply sigmoid function
y = torch.sigmoid(x)

# Convert tensors to NumPy arrays
x = x.numpy()
vx = x * 0.2
y = y.numpy()
vy = y*0.2

# Plot the sigmoid function
plt.plot(vx, vy, label="Python")
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend(loc="upper left")
plt.show()

print("end")