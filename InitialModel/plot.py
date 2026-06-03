import os
import shutil
# from simulator import So_history, Sw_history, x, N
import matplotlib.pyplot as plt
import numpy as np

folder = 'plots'

if os.path.exists(folder):
     shutil.rmtree(folder)

os.makedirs(folder)

d = np.loadtxt("output.txt",skiprows=1)
x = d[:,1]
Sw = d[:,2]
So = d[:,3]
Cp = d[:,4]


Sw_0 = d[:,5]
So_0 = d[:,6]
Cp_0 = d[:,7]

Sw_1 = d[:,8]
So_1 = d[:,9]
Cp_1 = d[:,10]

Sw_2 = d[:,11]
So_2 = d[:,12]
Cp_2 = d[:,13]


Sw_3 = d[:,14]
So_3 = d[:,15]
Cp_3 = d[:,16]

plt.figure(figsize=(12,8))
plt.plot(x,Sw_0, label = 'Sw')
plt.plot(x,So_0, label = 'So')
plt.plot(x,Cp_0, label = 'Cp')
plt.grid(True)
plt.legend()
plt.savefig(f'{'plots'}/graphic_t0.png', bbox_inches='tight')

plt.figure(figsize=(12,8))
plt.plot(x,Sw_1, label = 'Sw')
plt.plot(x,So_1, label = 'So')
plt.plot(x,Cp_1, label = 'Cp')
plt.grid(True)
plt.legend()
plt.savefig(f'{'plots'}/graphic_t1.png', bbox_inches='tight')


plt.figure(figsize=(12,8))
plt.plot(x,Sw_2, label = 'Sw')
plt.plot(x,So_2, label = 'So')
plt.plot(x,Cp_2, label = 'Cp')
plt.grid(True)
plt.legend()
plt.savefig(f'{'plots'}/graphic_t2.png', bbox_inches='tight')


plt.figure(figsize=(12,8))
plt.plot(x,Sw_3, label = 'Sw')
plt.plot(x,So_3, label = 'So')
plt.plot(x,Cp_3, label = 'Cp')
plt.grid(True)
plt.legend()
plt.savefig(f'{'plots'}/graphic_t3.png', bbox_inches='tight')


plt.figure(figsize=(12,8))
plt.plot(x,Sw, label = 'Sw')
plt.plot(x,So, label = 'So')
plt.plot(x,Cp, label = 'Cp')
plt.grid(True)
plt.legend()
plt.savefig(f'{'plots'}/graphic_tf.png', bbox_inches='tight')











# # n = len(Sw_history)
# # plot_values = [0.0,int(N/4),int(N/2),int(N*3/4),N-1]

# # for i in range(n):
# #     plt.figure(figsize=(12,8))
# #     plt.plot(x,Sw_history[i], label = 'Water Saturation')
# #     plt.plot(x,So_history[i], label = 'Oil Saturation')
# #     plt.title('t = {}'.format(plot_values[i]))
# #     plt.grid(True)
# #     plt.legend()   
# #     plt.savefig(f'{'plots'}/graphic_t_{i}.png', bbox_inches='tight')






