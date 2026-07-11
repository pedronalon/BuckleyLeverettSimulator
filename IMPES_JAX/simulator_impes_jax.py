import json
import jax.numpy as jnp 
import pandas as pd
import time
import jax.scipy.linalg as jsp_linalg
import jax



start = time.perf_counter() 

with open('input_impes_jax.json' , 'r' ) as p:
    parameters = json.load(p)


Swc = parameters.get("Swc") 
Sor = parameters.get("Sor") 

krw0 = parameters.get("krw0") 
kro0 = parameters.get("kro0") 
k = parameters.get("k")

nw = parameters.get("nw")  
no = parameters.get("no") 

mu_w = parameters.get("mu_w") 
mu_o = parameters.get("mu_o") 

phi = parameters.get("phi") 
a = parameters.get("a") 
q = parameters.get("q") 

ti = parameters.get("ti") 
tf = parameters.get("tf") 

Li = parameters.get("Li")
Lf = parameters.get("Lf") 
M = parameters.get("M")


cfl = parameters.get("cfl")
Csf = parameters.get("Csf")
max_rrf = parameters.get("max_rrf")

t_init = parameters.get("t_init")
t_inj = parameters.get("t_inj")




# fw_values = jnp.zeros(len(concentration))


# def max_fw():
#     Sw = sp.symbols('Sw')

#     lmbd_w = (k*(krw0 * ((Sw - Swc) /(1 - Swc - Sor))**nw)) / mu_w
#     lmbd_o = (kro0 * (1 - ((Sw - Swc) / (1 - Swc - Sor)))**no) / mu_o

#     fw = lmbd_w / (lmbd_w + lmbd_o)
    
#     derivate_1 = sp.diff(fw,Sw)
#     derivate_2 = sp.diff(derivate_1,Sw)

#     maximum_point = sp.nsolve(derivate_2,Sw, [(Swc+1e-5),(1-Sor-1e-5)] , solver = 'bisect', verify = False)
#     maximum_value = derivate_1.subs(Sw,maximum_point)

#     return float(maximum_value)

def Krw(Sw):
    sn = jnp.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
    kr = krw0 * (sn)**nw
    return kr

def Kro(Sw):
    jnp.clip(Sw,Sor,1-Swc)
    sn = jnp.clip( (Sw - Swc) /(1 - Swc - Sor) , 0.0, 1.0)
    kr = kro0 * (1 - sn)**no 
    return kr

def linear_interpolation(c,v,x):
    return jnp.interp(x,c,v)

# adsorção, ligar depois
# def r_factor(Cs):
#     R = 1 + (max_rrf-1)*(Cs/Csf)
#     return jnp.clip(R,1.0,max_rrf)

# def gamma(c):
#     l = 0.8
#     Cs = l*c
#     return Cs


# fw_values[:] = max_fw()
# fw_max_value = jnp.nanmax(fw_values)


dx = (Lf-Li)/(M-1)
v = q/a 
vp = phi*a*dx
# dt = cfl*(dx*phi)/(v) #*fw_max_value)
#forçando dt, pq calculando ficou muito grande
dt = 0.0800827936386817
N = int(tf/dt)

prod = jnp.zeros(N)
Sw = jnp.zeros(M)
p = jnp.zeros(M)
C = jnp.zeros(M)
T = jnp.zeros(M)
lmbd = jnp.zeros(M)
qt = jnp.zeros(M-1)
qw = jnp.zeros(M-1)
T_inter = jnp.zeros(M-1)
T_matrix = jnp.zeros((M-1,M-1))
# Cs = jnp.zeros(M)
# R = jnp.ones(M)

t = jnp.linspace(ti,tf,N-1)
x = jnp.linspace(Li,Lf,M)


Sw = Sw.at[:].set(Swc+0.001) 
p_right = 0.0
cont = 0 

def solve_pressure(pressure): 
    
    global T_inter, qt, T, cont

    concentration = jnp.array([0.0 , 100.0 , 300.0 , 500.0])
    viscosity = jnp.array([0.5 , 1.5 , 4.0 , 7.8])

    lmbd_w = (k*Krw(Sw))/(linear_interpolation(concentration, viscosity, C))
    lmbd_o = (k*Kro(Sw))/ mu_o 
    lmbd = lmbd_o + lmbd_w

    fw_up = lmbd_w/(lmbd_o+lmbd_w)

    T = (6.3283e-3 *lmbd[:]*a)/dx
            
    main_diag = jnp.zeros(M-1)
    lower_diag = jnp.zeros(M-1)
    upper_diag = jnp.zeros(M-1)

    T_right = jnp.zeros(M-1)
    T_left = jnp.zeros(M-1)
            
    T_right = (2 * T[:-1] * T[1:]) / (T[:-1] + T[1:])
    T_left_temp  = (2 * T[:-2] * T[1:-1]) / (T[:-2] + T[1:-1])
    T_left = T_left.at[1:].set(T_left_temp)

        
    T_inter = T_right

    # if j == 0
    main_diag = main_diag.at[0].set(T_right[0])
    upper_diag = upper_diag.at[0].set(-T_right[0])
    lower_diag = lower_diag.at[0].set(0.0)
    qt = qt.at[0].set(q)

    
    #if j == M-2 
    main_diag = main_diag.at[-1].set(T_left[-1] + T_right[-1])
    lower_diag = lower_diag.at[-1].set(-T_left[-1])
    upper_diag = upper_diag.at[-1].set(0.0)
    qt = qt.at[-1].set(T_right[-1] * p_right) 

    # internal points 
    main_diag = main_diag.at[1:-1].set(T_left[1:-1] + T_right[1:-1])
    lower_diag = lower_diag.at[1:-1].set(-T_left[1:-1])
    upper_diag  = upper_diag.at[1:-1].set(-T_right[1:-1])
    qt = qt.at[1:-1].set(0.0)

    
    #Tm = jnp.diag([lower_diag, main_diag , upper_diag], [-1, 0, 1]).toarray()
    
    # conseguimos resolver o sistema de uma matriz diagonal com o jax sem precisar montar a matriz (eu acho)
    # todos os arrays que passamos devem ter o mesmo tamanho, lower_diag[0] = 0.0 , upper_diag[m-1] = 0 
    # o vetor do lado direito precisa ser uma matriz
    qt_matrix = qt[:, None]
    
    solver = jax.lax.linalg.tridiagonal_solve(lower_diag,main_diag,upper_diag,qt_matrix)
    
    # pressure = pressure.at[:-1].set(jnp.linalg.solve(Tm,q_t[:]))
    # solver tem shape(M,1) , temos que tirar essa dimensão extra adicionada
    solver = solver.squeeze()
    pressure = pressure.at[:-1].set(solver)
    pressure = pressure.at[-1].set(p_right)

    Qt = -T_inter[:]*(pressure[1:] - pressure[:-1])

    Qw = fw_up[:-1]*Qt

    
    print("iteração: ", cont)
    cont+= 1
    return pressure, Qw

def impes_step(carry, t):# p, M, q_t, qw, p_right,c,v,T,Ti,vp,poly_inj): 

    global p, Sw, C
    Sw, C = carry
    p, qw = solve_pressure(p)

    Sw_old = Sw.copy()
    
    
    Sw_0 = Sw[0] + (dt / vp) * (q - qw[0])
    Sw = Sw.at[0].set(Sw_0)

    Sw_internal = Sw[1:-1] + (dt / vp) * (qw[:-1] - qw[1:])
    Sw = Sw.at[1:-1].set(Sw_internal)

    Sw = Sw.at[-1].set(Sw_old[-1])

    alpha = (qw*dt)/(vp)
    alpha_inj = (q*dt)/(vp)

    # encontrar forma de implementar o tempo de injeção
    poly_inj = jnp.where(jnp.logical_and(t >= t_init, t <= t_inj[1]), 500.00, 0.0)
    
    # poly_inj = 500.00

    C_f = ((Sw_old[-1] - alpha[-1]) * C[-1] + alpha[-1] * C[-2]) / Sw[-1]
    C = C.at[-1].set(C_f)
    
    C_inernal = ((Sw_old[1:-1] - alpha[1:]) * C[1:-1] + alpha[:-1] * C[:-2]) / Sw[1:-1]
    C = C.at[1:-1].set(C_inernal)

    C_0 = ((Sw_old[0] - alpha[0]) * C[0] + alpha_inj * poly_inj) / Sw[0]
    C = C.at[0].set(C_0)

    jnp.clip(Sw, Swc, 1.0 - Sor)


    return (Sw, C) , (Sw, C, p, qw)


impes_sol, history_impes = jax.lax.scan(impes_step,(Sw,C),t)

plot_values = [int(N/4)-1,int(N/2)-1,int(3*N/4)-1]

Sw_f = impes_sol[0]
C_f = impes_sol[1]

Sw_history = history_impes[0]
C_history = history_impes[1]
p_history = history_impes[2]
q_history = history_impes[3]

data =   {

    'Sw_1' : Sw_history[plot_values[0],:],
    'p_1' : p_history[plot_values[0],:],
    
    'Sw_2' : Sw_history[plot_values[1],:],
    'p_2' : p_history[plot_values[1],:],
    
    'Sw_3' : Sw_history[plot_values[2],:], 
    'p_3' : p_history[plot_values[2],:], 
    
    'Sw_4' : Sw_f ,
    'p_4' : p_history[-1,:] ,

    'C_1' : C_history[plot_values[0],:],
    'C_2' : C_history[plot_values[1],:],
    'C_3' : C_history[plot_values[2],:], 
    'C_4' : C_f ,
       
        
        }



df = pd.DataFrame(data)
path = "/home/pedro/Área de trabalho/Faculdade /BuckleyLeverettSimulator/Diff/output_impes_jax.txt"

with open(path, 'w') as archive:
    archive.write(df.to_string())