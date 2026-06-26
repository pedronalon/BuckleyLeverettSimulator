import json
import numpy as np 
import matplotlib.pyplot as plt


with open('input_impes.json' , 'r' ) as p:
    parameters = json.load(p)

t_inj = parameters.get("t_inj")
t_init = parameters.get("t_init")
tf = parameters.get("tf")


plt.figure(figsize=(12,8))

for i in range(1,5):
    d = np.loadtxt("output_prod_{}.txt".format(i),skiprows=1)
    t = d[:,1]
    prod = d[:,2]

    plt.plot(t,prod,label = "Oil production for {}  days ".format(t_inj[i-1] - t_init))


plt.title("Oil accumulated Production with polymer injection | tf = {} " .format(tf[2]))
plt.grid(True)
plt.legend()
plt.show()
