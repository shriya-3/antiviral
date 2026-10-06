import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import pandas as pd

COLOR = "grey"

configs = {

"purple": {
    "virus_file": "data/virus_purple.dat",
    "cells_file": "data/cells_purple.dat",
    "K": 4767840.060099503,
    "initial_guess":[2.15340282e-06, 3.18465532e+01, 9.67901147e+00, 1.31840638e+00],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_purple.png",
    "title":"ERDRP-0519 (3 dpi)"
},

"green": {
    "virus_file": "data/virus_green.dat",
    "cells_file": "data/cells_green2.dat",
    "K": 3123092.42921152,
    "initial_guess":[3.33848192e-05, 2.75132844e+02, 1.74369880e+01, 1.67546080e+00],
    "prior":[-8,-2,1,7,-2,4,-3,2],
    "fig":"final_fig_green.png",
    "title":"GHP-88309 (5 dpi)"
},

"red": {
    "virus_file": "data/virus_red.dat",
    "cells_file": "data/cells_red2.dat",
    "K": 4502249.959989419,
    "initial_guess":[1.71571339e-05, 3.01583306e+01, 2.61835030, 4.06336670],
    "prior":[-7,-2,-3,3,-3,3,-3,2],
    "fig":"final_fig_red.png",
    "title":"GHP-88309 (3 dpi)"
},

"yellow": {
    "virus_file": "data/virus_yellow.dat",
    "cells_file": "data/cells_yellow2.dat",
    "K": 5142701.720558477,
    "initial_guess":[2.45361792e-06, 2.22159842e+01, 4.59566061e+00, 5.37628973e-01],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_yellow.png",
    "title":"GHP-88309 (7 dpi)"
},

"grey": {
    "virus_file": "data/virus_grey.dat",
    "cells_file": "data/cells_grey2.dat",
    "K": 5972735.535321577,
    "initial_guess":[9.68398003e-06, 2.82016815e+02, 1.45086952e+01, 5.59776275e-01],
    "prior":[-6,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_grey.png",
    "title":"Vehicle"
}

}

cfg = configs[COLOR]

virus = np.loadtxt(cfg["virus_file"])
cells = np.loadtxt(cfg["cells_file"])

vdata = np.log10(virus[:,1])
tdata = np.log10(cells[:,1])
t = virus[:,0]


K_fixed = cfg["K"]

lamb = 0.085
k = 3.0

def model(y, t, params):
    T, E, I, V = y
    lamb = 0.085
    k = 3.0
    K = K_fixed
    beta, delta, p, c  = params
    dTdt = lamb * T * (1 - T / K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])

def ssr_basic(pred, true):
    return np.sum((np.log10(pred) - np.log10(true))**2)

def ssr(params, y0, t, V_data, T_data):
    params = 10**params
    result = odeint(model, y0, t, args=(params,))

    T_pred_ssr = result[:, 0]
    E_pred_ssr = result[:, 1]
    I_pred_ssr = result[:, 2]
    V_pred_ssr = result[:, 3]

    T_sum, V_sum = 0, 0
    for i in range(len(T_pred_ssr)):
        T_sum += (np.log10(T_data[i]) - (np.log10(T_pred_ssr[i]))) ** 2

    for i in range(len(V_pred_ssr)):
        V_sum += (np.log10(V_data[i]) - np.log10(V_pred_ssr[i])) ** 2

    return T_sum + V_sum



def main():
    virus = np.loadtxt(cfg["virus_file"])
    cells = np.loadtxt(cfg["cells_file"])

    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]



    y0=[(cfg["K"]), 0, 0, 1]
    bounds = [(None, 0), (-5, 2), (-2, 4), (-5, 2)]
    initial_guess = np.log10(cfg["initial_guess"])
    result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead", bounds=bounds)

    print(f"ssr: {ssr(result.x, y0, t, V_data, T_data)}")
    estimated_params = 10**result.x
    print(estimated_params)

   
    t_new = np.linspace(0, 22, 100)
    result = odeint(model, y0, t_new, args=(estimated_params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]
    plt.figure(figsize=(12, 8))
    

    plt.plot(t_new, T_pred, color='blue', linestyle='--', label='Predicted Lymphocytes', linewidth=2)
    plt.plot(t_new, V_pred, color='red', linestyle='--', label='Predicted Virus Titer', linewidth=2)
    
    plt.scatter(t, T_data, color='blue', label='Experimental Lymphocytes $ml^{-1}$ data', zorder=5)
    plt.scatter(t, V_data, color='red', label='Experimental Virus Titer data', zorder=5)

    plt.rcParams.update({'font.size': 23})
    plt.xlabel('Time post CDV infection (days)', fontsize=20)
    plt.ylabel('Concentration', fontsize=20)
    plt.yscale('log')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.ylim(1e-1, 1e8) #yellow
    plt.ylim(1e-1, 1e7) 
    plt.xlim(0, 12)
    plt.title(cfg["title"])

    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
