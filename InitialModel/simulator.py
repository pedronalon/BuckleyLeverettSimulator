
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
maxrrf = parameters.get("maxrrf")
fw_ghost = parameters.get("fw_ghost")

# N = parameters.get("N")
M = parameters.get("M")

ti = parameters.get("ti") 
tf = parameters.get("tf") 

Li = parameters.get("Li")
Lf = parameters.get("Lf") 

dx = parameters.get("dx")
cfl = parameters.get("cfl")




# t = np.arange(ti,tf+dt,dt)
# x = np.arange(Li,Lf+dx,dx)

# N = len(t)
# M = len(x) 







def Cp_step(x,t,v_p):
    x_front = v_p*t
    return np.where(x <= x_front, 1.0, 0.0)

def rrf(Cp):
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

def upwind(Sw,h,x,t,N,plot_values,fw,Cp_step,fw_ghost,Cp_history):

    Sw_history.append(Sw.copy())
    So_history.append(1-Sw.copy())
    
    
    for i in range(N):
    
        cp = Cp_step(x,t[i],vp)
        

        Sw[1:] = Sw[1:] - (h*(fw(Sw[1:],cp[1:])-fw(Sw[:-1],cp[:-1])))
        Sw[0] = Sw[0] - h*(fw(Sw[0],cp[0]) - fw_ghost) # Neumann BC 
        
        if i in plot_values:
            Sw_history.append(Sw.copy())
            So_history.append(1-Sw.copy())
            Cp_history.append(cp)
      
    return Sw, Sw_history, So_history, Cp_history


fw_max_polymer = max_fw(1.0)
fw_max_w = max_fw(0.0)
fw_max = max(fw_max_polymer,fw_max_w)


dt = cfl*(dx*phi)/(v*fw_max)
print("dt:", dt)
N = int(tf/dt)

t = np.linspace(ti,tf,N)
x = np.linspace(Li,Lf,M)
h = (v*dt)/(dx*phi)

Sw = np.zeros(M)
Sw[:] = Sw0

plot_values = [int(N/4),int(N/2),int(N*3/4),N-1]
Sw_history = []
So_history = []
Cp_history = []


Sw, Sw_history,So_history,Cp_history = upwind(Sw,h,x,t,N,plot_values,fw,Cp_step,fw_ghost,Cp_history)
So = 1-Sw

data = {
    'x' : x , 
    'Sw' : Sw,
    'So' : So,
    'Cp' : Cp_step(x,t[N-1],vp),
    
    'Sw_0' : Sw_history[0],
    'So_0' : So_history[0],
    'Cp_0' : Cp_history[0],
    
    'Sw_1' : Sw_history[1],
    'So_1' : So_history[1],
    'Cp_1' : Cp_history[1],
    
    'Sw_2' : Sw_history[2],
    'So_2' : So_history[2],
    'Cp_2' : Cp_history[2],
    
    'Sw_3' : Sw_history[3],
    'So_3' : So_history[3],
    'Cp_3' : Cp_history[3],

    

}   

df = pd.DataFrame(data)

with open('output.txt', 'w') as archive:
    archive.write(df.to_string())


end = time.perf_counter() 
execution_time = end - start

print("CFL:", cfl)
print("Execution Time:", execution_time)