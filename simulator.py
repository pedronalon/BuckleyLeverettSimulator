
import json 
import numpy as np
import sympy as sp 
import pandas as pd 
import time 


start = time.perf_counter() 

with open('input.json' , 'r' ) as p:
    parameters = json.load(p)


Swc = parameters.get("Swc") 
Sor = parameters.get("Sor") 
krw0 = parameters.get("krw0") 
kro0 = parameters.get("kro0") 
nw = parameters.get("nw")  
no = parameters.get("no") 
uw = parameters.get("uw") 
uo = parameters.get("uo") 
phi = parameters.get("phi") 
Sw0 = parameters.get("Sw0") 
v = parameters.get("v")
vp = parameters.get("vp")

ti = parameters.get("ti") 
tf = parameters.get("tf") 

Li = parameters.get("Li")
Lf = parameters.get("Lf") 

dx = parameters.get("dx")
dt = parameters.get("dt")


t = np.arange(ti,tf+dt,dt)
x = np.arange(Li,Lf+dx,dx)

h = (v*dt)/(dx*phi)

M = len(x)
N = len(t)



def Cp_step(x,t,v_p):
    x_front = v_p*t
    return np.where(x <= x_front, 1.0, 0.0)

def rrf(Cp):
    maxrrf = 1.0
    return 1.0 + (maxrrf-1.0)*Cp

def max_fw(cp_val):
    
    Sw = sp.symbols('Sw')

    lmbd_w = (krw0 * ((Sw - Swc) /(1 - Swc - Sor))**nw) / (uw *rrf(cp_val))
    lmbd_o = (kro0 * (1 - ((Sw - Swc) / (1 - Swc - Sor)))**no) / uo

    fw = lmbd_w / (lmbd_w + lmbd_o)
    
    derivate_1 = sp.diff(fw,Sw)
    derivate_2 = sp.diff(derivate_1,Sw)


    maximum_point = sp.nsolve(derivate_2,Sw, [(Swc+1e-5),(1-Sor-1e-5)] , solver = 'bisect', verify = False)
    maximum_value = derivate_1.subs(Sw,maximum_point)

    return maximum_value

def Krw(Sw): 
    return krw0 * ((Sw - Swc) /(1 - Swc - Sor))**nw

def Kro(Sw):
    return kro0 * (1 - ((Sw - Swc)/(1 - Swc - Sor)))**no 

def fw(Sw,cp): 
    lmbd_w = Krw(Sw)/(uw*rrf(cp))
    lmbd_o = Kro(Sw)/uo 

    return lmbd_w/(lmbd_w + lmbd_o)

def upwind(Sw,h,x,t,plot_values,fw,Cp_step):
    Sw_history = []
    So_history = []

    Sw_history.append(Sw.copy())
    So_history.append(1-Sw.copy())
    
    for i in range(N):
    
        cp = Cp_step(x,t[i],vp)

        Sw[1:] = Sw[1:] - (h*(fw(Sw[1:],cp[1:])-fw(Sw[:-1],cp[:-1])))
        Sw[0] = 1 - Sor

        if i == 0.0:
            continue
        elif i in plot_values:
            Sw_history.append(Sw.copy())
            So_history.append(1-Sw.copy())
      
    return Sw, Sw_history, So_history


fw_max_polymer = max_fw(1.0)
fw_max_w = max_fw(0.0)
fw_max = max(fw_max_polymer,fw_max_w)

cfl = (dx*phi)/(v*fw_max*dt)

Sw = np.zeros(M)
Sw[:] = Sw0

plot_values = [0.0,int(N/4),int(N/2),int(N*3/4),N-1]
Sw_history = []
So_history = []


Sw, Sw_history,So_history = upwind(Sw,h,x,t,plot_values,fw,Cp_step)
So = 1-Sw

data = {
    'x' : x , 
    'Sw' : Sw,
    'So' : So,
}

df = pd.DataFrame(data)
end = time.perf_counter() 
execution_time = end - start

with open('output.txt', 'w') as archive:
    archive.write('CFL: {}'.format(cfl))
    archive.write('\n')
    archive.write('Execution time : {}'.format(execution_time))
    archive.write(2*'\n')
    archive.write(df.to_string())


