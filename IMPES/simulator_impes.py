import json 
import numpy as np
import sympy as sp 
import pandas as pd 
import time
from scipy.sparse import diags
import matplotlib.pyplot as plt


start = time.perf_counter() 


with open('input_impes.json' , 'r' ) as p:
    parameters = json.load(p)


Swc = parameters.get("Swc") 
Sor = parameters.get("Sor") 

krw0 = parameters.get("krw0") 
kro0 = parameters.get("kro0") 
k = parameters.get("k")

nw = parameters.get("nw")  
no = parameters.get("no") 

uw = parameters.get("uw") 
uo = parameters.get("uo") 

phi = parameters.get("phi") 
a = parameters.get("a") 
q = parameters.get("q") 

ti = parameters.get("ti") 
tf = parameters.get("tf") 

Li = parameters.get("Li")
Lf = parameters.get("Lf") 
# dx = parameters.get("dx")
M = parameters.get("M")

cfl = parameters.get("cfl")
fw_ghost = parameters.get("fw_ghost")
max_rrf = parameters.get("max_rrf")
Csf = parameters.get("Csf")


def p3(x): 

    match x:
        case 0:
            return 0.5
        case 100:
            return 2.5
        case 300:
            return 4.0
        case 500:
            return 7.8
        case _ :
            return 1.0   

def max_fw(x):
    
    Sw = sp.symbols('Sw')

    lmbd_w = (k*(krw0 * ((Sw - Swc) /(1 - Swc - Sor))**nw)) / (uw*p3(x))
    lmbd_o = (kro0 * (1 - ((Sw - Swc) / (1 - Swc - Sor)))**no) / uo

    fw = lmbd_w / (lmbd_w + lmbd_o)
    
    derivate_1 = sp.diff(fw,Sw)
    derivate_2 = sp.diff(derivate_1,Sw)


    maximum_point = sp.nsolve(derivate_2,Sw, [(Swc+1e-5),(1-Sor-1e-5)] , solver = 'bisect', verify = False)
    maximum_value = derivate_1.subs(Sw,maximum_point)

    return float(maximum_value)

def Krw(Sw):
    sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
    kr = krw0 * (sn)**nw
    return kr

def Kro(Sw):
    np.clip(Sw,Sor,1-Swc)
    sn = np.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
    kr = kro0 * (1 - sn)**no 
    return kr

def fw(Sw): #,x): 
    lmbd_w = (k*Krw(Sw))/(uw)#*p3(x))
    lmbd_o = (k*Kro(Sw))/uo 

    return lmbd_w/(lmbd_w + lmbd_o)

v = q/a 

concentration = [0.0 , 100.0 , 300.0 , 500.0]

fw_values = np.zeros(len(concentration))
    
for i in range(len(concentration)):
    fw_values[i] = max_fw(concentration[i])
    
        
fw_max_value = np.nanmax(fw_values)

dx = (Lf-Li)/(M-1)
dt = cfl*(dx*phi)/(v*fw_max_value)

print("dt:", dt)
N = int(tf/dt)
N = 500


t = np.linspace(ti,tf,N)
x = np.linspace(Li,Lf,M)
h = (v*dt)/(dx*phi)

Sw = np.zeros((N,M))
p = np.zeros((N,M))
C = np.zeros((N,M))
T = np.zeros((N,M))
lmbd = np.zeros((N,M))

T_matrix = np.zeros((M-1,M-1))

T_inter = np.zeros(M-1)

qt = np.zeros(M-1)
qw = np.zeros(M-1)

# qt[0] = -q
# qt[-1] = q
 
Sw[0,:] = Swc+0.001


p_right = 0.0


for i in range(N-1):
    lmbd_w = (k * Krw(Sw[i, :])) / (uw)#*p3(500))
    lmbd_o = (k* Kro(Sw[i, :]) )/ uo 

    lmbd[i, :] = lmbd_o + lmbd_w

    fw_up = fw(Sw[i,:])

    T[i,:] = (6.3283e-3 *lmbd[i, :]*a)/dx
    print(T[i,:])

    main_diag = np.zeros(M-1)
    lower_diag = np.zeros(M-2)
    upper_diag = np.zeros(M-2)

    # print(T[i,:])
    # print(lmbd_w)

    for j in range(0,M-1):
        
        T_left  = 0.0 if j == 0 else (2*T[i,j]*T[i,j-1]) / (T[i,j]+T[i,j-1])
        T_right = 0.0 if j == M-1 else (2*T[i,j]*T[i,j+1]) / (T[i,j]+T[i,j+1])

        T_left  = (2*T[i,j]*T[i,j-1]) / (T[i,j]+T[i,j-1])
        T_right = (2*T[i,j]*T[i,j+1]) / (T[i,j]+T[i,j+1])

        
        
        T_inter[j] = T_right 
        
        
        if j == 0: 
        
            main_diag[j] = T_right
            upper_diag[j] = -T_right 
            qt[j] = q
            continue 


        if j == M-2:                       
            main_diag[j] = T_left + T_right
            lower_diag[j-1] = -T_left
            qt[j] = T_right * p_right  
            continue

        
        
        main_diag[j] = T_left + T_right
        lower_diag[j-1] = -T_left
        upper_diag[j] = -T_right
        qt[j] =  0.0


    T_matrix = diags([lower_diag, main_diag , upper_diag], [-1, 0, 1]).toarray()
    print(T_matrix)
    print("\n")
    print(qt)
    print('\n')
    p[i+1,0:-1] = np.linalg.solve(T_matrix,qt[:]) 
    # p[i+1, 0]  = p_left
    p[i+1, -1] = p_right
    
    # print(p[1,:])

    vp = phi*a*dx
    alpha = (v*dt)/(phi*dx)



    # print("TW",Tw)


    for j in range(0, M-1):
        
        #topico 5 e 6 
        Q_t = -T_inter[j]*(p[i+1,j+1] - p[i+1,j])
        

        if (p[i+1,j] > p[i+1,j+1]):
            qw[j] = fw_up[j]*Q_t
        else:
            
            qw[j] = fw_up[j+1]*Q_t


        if j == 0:
            Sw[i+1, j] = Sw[i, j] + (dt / vp) * (q - qw[j])
        
        else: 
            Sw[i+1, j] = Sw[i, j] + (dt / vp) * (qw[j-1] - qw[j])
       
            
    


        
        C[i+1, j] = ((Sw[i, j] - alpha) * C[i, j] + alpha * C[i, j-1]) / Sw[i+1, j]
    
    Sw[i+1, M-1] = Sw[i, M-1]
    C[i+1, 0] = Csf
    # print("\n")
    # print(Sw[i+1,0])
    
    np.clip(Sw[i+1, :], Swc, 1.0 - Sor)

    if(np.isnan(Sw[i+1,:]).any()):
        break

    if(i%100==0):
        plt.plot(x,Sw[i+1],label=(i+1))
        # plt.plot(x,p[i+1,:])
    

plt.legend()
plt.show()





data = {
    'x' : x , 
    
    'Sw_0' : Sw[0,:],
    'Cp_0' : C[0,:],
    
    'Sw_1' : Sw[int(N/4),:],
    'Cp_1' : C[int(N/4),:],
    
    'Sw_2' : Sw[int(N/2),:],
    'Cp_2' : C[int(N/2),:],
    
    'Sw_3' : Sw[int(3*N/4),:],
    'Cp_3' : C[int(3*N/4),:],

    'Sw_f' : Sw[-1,:],
    'Cp_f' : C[-1,:],
    'pf' : p[-1,:]


    
}   


df = pd.DataFrame(data)

with open('output_impes.txt', 'w') as archive:
    archive.write(df.to_string())

end = time.perf_counter() 
execution_time = end - start

print("CFL:", cfl)
print("Execution Time:", execution_time)




