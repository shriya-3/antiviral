from __future__ import print_function, division

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from scipy.optimize import minimize

global par
global data


COLOR = "grey"


configs = {

"purple": {
    "virus_file": "data/virus_purple.dat",
    "cells_file": "data/cells_purple.dat",
    "K": 4767840.060099503,
    "initial_guess":[5.34848694e-06, 1.00000000e+02, 1.98769229e+01, 1.35770017e+00],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"samecons_fig_purple.png",
    "csv":"boot_purple.csv",
    "title":"ERDRP-0519 (3 dpi)"
},

"green": {
    "virus_file": "data/virus_green.dat",
    "cells_file": "data/cells_green2.dat",
    "K": 3123092.42921152,
    "initial_guess":[4.06356553e-04,1.00000000e+02,3.10613835e-01,9.93663321e-01],
    "prior":[-8,-2,1,7,-2,4,-3,2],
    "fig":"samecons_fig_green.png",
    "csv":"boot_green.csv",
    "title":"GHP-88309 (5 dpi)"
},

"red": {
    "virus_file": "data/virus_red.dat",
    "cells_file": "data/cells_red2.dat",
    "K": 4502249.959989419,
    "initial_guess":[5.00455586e-05,9.99882682e+01,1.76178104e+01,2.17512702e+01],
    "prior":[-7,-2,-3,3,-3,3,-3,2],
    "fig":"samecons_fig_red.png",
    "csv":"boot_red.csv",
    "title":"GHP-88309 (3 dpi)"
},

"yellow": {
    "virus_file": "data/virus_yellow.dat",
    "cells_file": "data/cells_yellow2.dat",
    "K": 5142701.720558477,
    "initial_guess":[1.45794847e-06,9.99997040e+01,5.25328929e+01,8.64861350e-01],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"samecons_fig_yellow.png",
    "csv":"boot_yellow.csv",
    "title":"GHP-88309 (7 dpi)"
},

"grey": {
    "virus_file": "data/virus_grey.dat",
    "cells_file": "data/cells_grey2.dat",
    "K": 5972735.535321577,
    "initial_guess":[5.34848694e-06,1.00000000e+02,1.98769229e+01,1.35770017e+00],
    "prior":[-6,-2,-3,3,-3,4,-3,2],
    "fig":"samecons_fig_grey.png",
    "csv":"boot_grey.csv",
    "title":"Vehicle"
}

}

cfg = configs[COLOR]


###########################################################
# LOAD DATA
###########################################################

virus = np.loadtxt(cfg["virus_file"])
cells = np.loadtxt(cfg["cells_file"])

vdata = np.log10(virus[:,1])
tdata = np.log10(cells[:,1])
t = virus[:,0]

print("data",vdata)

K_fixed = cfg["K"]

lamb = 0.085
k = 3.0


###########################################################
# ODE MODEL
###########################################################

def model(y,t):

    global par

    beta = par[0]
    delta = par[1]
    p = par[2]
    c = par[3]

    K = K_fixed

    lamb = 0.085
    k = 3.0

    T,E,I,V = y

    dTdt = lamb*T*(1-T/K) - beta*T*V
    dEdt = beta*T*V - k*E
    dIdt = k*E - delta*I
    dVdt = p*I - c*V

    return np.array([dTdt,dEdt,dIdt,dVdt])


###########################################################
# SOLVE MODEL
###########################################################

def solveeqs(beta,delta,p,c,t):

    global par

    fpar = np.power(10,[beta,delta,p,c])
    par = list(fpar)

    y=[K_fixed,0.0,0.0,1.0]

    ysol = odeint(model,y,t)

    Tm = ysol[:,0]
    Vm = ysol[:,3]

    return np.array([Tm,Vm])


###########################################################
# INITIAL GUESS
###########################################################

#result = np.log10(cfg["initial_guess"])
best_fit_log = np.log10(cfg["initial_guess"])

best_model = solveeqs(*best_fit_log, t)

T_fit_log = np.log10(best_model[0])
V_fit_log = np.log10(best_model[1])

T_resid = tdata - T_fit_log
V_resid = vdata - V_fit_log

residual_pool = np.concatenate([T_resid, V_resid])

print("Residual pool size =", len(residual_pool))

def objective(theta, t, tdata_boot, vdata_boot):

    model = solveeqs(*theta, t)
    model = np.where(model > 0, np.log10(model), -20)

    ssq = np.sum((tdata_boot - model[0])**2)
    ssq += np.sum((vdata_boot - model[1])**2)

    return ssq

nboot = 1000

bootstrap_params = np.zeros((nboot, 4))

bounds = [
    (-8, -2),   # beta
    (-1, 8),    # delta
    (-2, 10),   # p
    (-3, 3)     # c
]

for i in range(nboot):

    if i % 50 == 0:
        print("Bootstrap", i, "/", nboot)

    # Shuffle all residuals without replacement
    shuffled = np.random.permutation(residual_pool)

    # Split shuffled residuals back into T and V groups
    resid_T = shuffled[:len(tdata)]
    resid_V = shuffled[len(tdata):len(tdata)+len(vdata)]

    t_boot = T_fit_log + resid_T
    v_boot = V_fit_log + resid_V

    fit = minimize(
        objective,
        x0=best_fit_log,
        args=(t, t_boot, v_boot),
        bounds=bounds,
        method="L-BFGS-B"
    )

    bootstrap_params[i] = fit.x

###########################################################
# SAVE CSV
###########################################################
K_col = np.full((nboot,1),K_fixed)
lamb_col = np.full((nboot,1),lamb)
k_col = np.full((nboot,1),k)

output = np.hstack([
    bootstrap_params,
    lamb_col,
    k_col,
    K_col
])

np.savetxt(
    cfg["csv"],
    output,
    delimiter=',',
    header='beta,delta,p,c,lamb,k,K',
    comments=''
)

print("Saved", nboot, "bootstrap fits to", cfg["csv"])