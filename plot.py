import os
import shutil
import matplotlib.pyplot as plt
from simulator import So_history, Sw_history, x, plot_values


folder = 'plots'

if os.path.exists(folder):
    shutil.rmtree(folder)

os.makedirs(folder)

n = len(Sw_history)

for i in range(n):
    plt.figure(figsize=(12,8))
    plt.plot(x,Sw_history[i], label = 'Water Saturation')
    plt.plot(x,So_history[i], label = 'Oil Saturation')
    plt.title('t = {}'.format(plot_values[i]))
    plt.grid(True)
    plt.legend()   
    plt.savefig(f'{'plots'}/graphic_t_{i}.png', bbox_inches='tight')

