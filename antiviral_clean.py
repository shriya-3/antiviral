import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, dual_annealing, basinhopping
import math
import matplotlib.pyplot as plt
import pandas as pd

COLOR = "red"

configs = {

"purple": {
    "virus_file": "data/virus_purple.dat",
    "cells_file": "data/cells_purple.dat",
    "K": 4767840.060099503,
    "initial_guess":[2.15340282e-06, 3.18465532e+01, 9.67901147e+00, 1.31840638e+00],
    #"initial_guess":[2.00853841e-06, 3.31976254e+01, 3.15216189e+01, 3.96836500e+00, 1.87299485e-11, 5.22880864e-01],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_purple.png",
    "csv":"boot_purple.csv",
    "title":"ERDRP-0519 (3 dpi)"
},

"green": {
    "virus_file": "data/virus_green.dat",
    "cells_file": "data/cells_green2.dat",
    "K": 3123092.42921152,
    "initial_guess":[3.33848192e-05, 2.75132844e+02, 1.74369880e+01, 1.67546080e+00],
    #"initial_guess":[3.09797504e-05, 1.00000000e+02, 9.65063794e+00, 2.13469946e+00, 5.55906075e-07, 3.07777025e-03],
    "prior":[-8,-2,1,7,-2,4,-3,2],
    "fig":"final_fig_green.png",
    "csv":"boot_green.csv",
    "title":"GHP-88309 (5 dpi)"
},

"red": {
    "virus_file": "data/virus_red.dat",
    "cells_file": "data/cells_red2.dat",
    "K": 4502249.959989419,
    #"initial_guess":[1.71571339e-05, 3.01583306e+01, 2.61835030, 4.06336670],
    "initial_guess":[8.45361792e-06, 3.01583306e+01, 2.61835030, 4.06336670],
    #"initial_guess":[9.80495773e-04, 8.84406862e+00, 1.00000000e-02, 2.19724926e+00],
    
    #"initial_guess":[3.7083578e-06, 3.01583306e+01, 2.61835030, 4.06336670],

    "prior":[-7,-2,-3,3,-3,3,-3,2],
    "fig":"final_fig_red.png",
    "csv":"boot_red.csv",
    "title":"GHP-88309 (3 dpi)"
},

"yellow": {
    "virus_file": "data/virus_yellow.dat",
    "cells_file": "data/cells_yellow2.dat",
    "K": 5142701.720558477,
    "initial_guess":[2.45361792e-06, 2.22159842e+01, 4.59566061e+00, 5.37628973e-01],
    #"initial_guess":[6.14626327e-08, 1.00000000e+02, 3.68365200e+03, 10, 7.22415476e-01, 5.62588194e-04],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_yellow.png",
    "csv":"new_format_yellow.csv",
    "csv_mcmc":"new_format_yellow.csv",
    "title":"GHP-88309 (7 dpi)"
},

"grey": {
    "virus_file": "data/virus_grey.dat",
    "cells_file": "data/cells_grey2.dat",
    "K": 5972735.535321577,
    "initial_guess":[9.68398003e-06, 2.82016815e+02, 1.45086952e+01, 5.59776275e-01],
    "prior":[-6,-2,-3,3,-3,4,-3,2],
    "fig":"final_fig_grey.png",
    "csv":"boot_grey.csv",
    "title":"Vehicle"
}

}

cfg = configs[COLOR]

#DATA LOADING
virus = np.loadtxt(cfg["virus_file"])
cells = np.loadtxt(cfg["cells_file"])

vdata = np.log10(virus[:,1])
tdata = np.log10(cells[:,1])
t = virus[:,0]

#print("data",vdata)

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
'''

def model(y, t, params):

    T, R, E, I, V = y

    lamb = 0.085
    k = 3.0
    K = K_fixed
    beta, delta, p, c, phi, rho  = params

    dTdt = lamb * T * (1 - (T + R + E + I) / K) - (beta * T * V) - (phi * T * I) + (rho + R)
    dRdt = lamb * R * (1 - (T + R + E + I) / K) + (phi * T * I) - (rho * R)
    dEdt = (beta * T * V) - (k * E)
    dIdt = (k * E) - (delta * I)
    dVdt = (p * I) - (c * V)

    return np.array([dTdt, dRdt, dEdt, dIdt, dVdt])'''


def ssr_basic(pred, true):
    """Compute the sum of squared residuals between predicted and true values."""
    return np.sum((np.log10(pred) - np.log10(true))**2)

def ssr(params, y0, t, V_data, T_data):
    params = 10**params
    result = odeint(model, y0, t, args=(params,))

    T_pred_ssr = result[:, 0]
    #R_pred_ssr = result[:, 1]
    E_pred_ssr = result[:, 1]
    I_pred_ssr = result[:, 2]
    V_pred_ssr = result[:, 3]

    
    T_sum, V_sum = 0, 0
    for i in range(len(T_pred_ssr)):
        T_sum += (np.log10(T_data[i]) - (np.log10((T_pred_ssr[i]) + (E_pred_ssr[i]) + (I_pred_ssr[i])))) ** 2
        #T_sum += (np.log10(T_data[i]) - (np.log10(T_pred_ssr[i]))) ** 2
        #T_sum += (np.log10(T_data[i]) - (np.log10(T_pred_ssr[i] + R_pred_ssr[i]))) ** 2
        #print(T_sum)

    for i in range(len(V_pred_ssr)):
        V_sum += (np.log10(V_data[i]) - np.log10(V_pred_ssr[i])) ** 2

    return T_sum + V_sum

def ssr_for_mcmc(params, y0, t, V_data, T_data):
    """Compute the SSR for a set of MCMC parameters."""
    result = odeint(model, y0, t, args=(params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]
    #ƒ
    T_ssr = ssr_basic(T_pred, T_data)
    V_ssr = ssr_basic(V_pred, V_data)
    
    return T_ssr + V_ssr


def main():
    virus = np.loadtxt(cfg["virus_file"])
    cells = np.loadtxt(cfg["cells_file"])

    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]



    y0=[(cfg["K"]), 0, 0, 1]
    #beta, delta, p, c
    bounds = [(-8, -2), (-3, 3), (-2, 10), (-3, 3)] #basin hopping bounds


    initial_guess = np.log10(cfg["initial_guess"])
    result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead", bounds=bounds)
    #result = dual_annealing(ssr, bounds, args=(y0, t, V_data, T_data), x0=initial_guess)
    #result = basinhopping(ssr, initial_guess, minimizer_kwargs={"args": (y0, t, V_data, T_data), "method": "L-BFGS-B", "bounds": bounds})
    
    #beta: none to 0
    #delta: -5 to 2
    #p: -2 to 4
    #c: -5 to 2
    print(f"ssr: {ssr(result.x, y0, t, V_data, T_data)}")
    estimated_params = 10**result.x
    #print(result.x)
    print(estimated_params)

   
    #t_new = np.linspace(0, 22, 100)
    t_new = np.linspace(0, 12, 100) #GREY
    result = odeint(model, y0, t_new, args=(estimated_params,))
    T_pred = result[:, 0]
    E_pred = result[:, 1]
    I_pred = result[:, 2]
    V_pred = result[:, 3]

    '''params_df = pd.read_csv((cfg["csv"]))


    # Arrays to store predictions from all samples
    T_preds_all = []
    V_preds_all = []
    ssr_values = []

    
    for _, row in params_df.iloc[:1000].iterrows():
        mcmc_params = row.values[:4]
        result = odeint(model, y0, t_new, args=(10**mcmc_params,))
        T_preds_all.append(result[:, 0])  
        V_preds_all.append(result[:, 3])
        
        # Compute SSR for each MCMC sample
        T_ssr = ssr_basic(result[:, 0], T_pred)
        V_ssr = ssr_basic(result[:, 3], V_pred)
        ssr_values.append(T_ssr + V_ssr)

    T_preds_all = np.array(T_preds_all)
    V_preds_all = np.array(V_preds_all)
    ssr_values = np.array(ssr_values)

    # Print SSR values to debug
    #print("SSR values: ", ssr_values)
    
    # Check if any values are below the threshold
    #print("Number of MCMC samples below threshold: ", np.sum(ssr_values < 10))

    # Define a fixed SSR threshold (adjust as necessary)
    ssr_threshold = 15  # Use a fixed threshold for testing

    # Plot the MCMC sample lines that are reasonably close to the original predictions
    plt.figure(figsize=(12, 8))
    
    sorted_indices = np.argsort(ssr_values)

# Keep only the best 90%
    num_to_keep = int(0.9 * len(sorted_indices))
    best_indices = sorted_indices[:num_to_keep]

# Plot only the best 90%
    for i in best_indices:
        plt.plot(t_new, T_preds_all[i], color='blue', alpha=0.1, linewidth=0.1)
        plt.plot(t_new, V_preds_all[i], color='red', alpha=0.1, linewidth=0.1)
    
    #print(f"Number of MCMC samples plotted: {sample_count}")'''

    plt.figure(figsize=(12, 8))
    
    # Overlay the original predicted line for Lymphocytes and Virus Titer
    plt.plot(t_new, (T_pred+E_pred+I_pred), color='blue', linestyle='--', label='Predicted Lymphocytes', linewidth=2)
    plt.plot(t_new, V_pred, color='red', linestyle='--', label='Predicted Virus Titer', linewidth=2)
    
    # Scatter plot for experimental data
    plt.scatter(t, T_data, color='blue', label='Experimental Lymphocytes $ml^{-1}$ data', zorder=5)
    plt.scatter(t, V_data, color='red', label='Experimental Virus Titer data', zorder=5)

    # Customize the plot
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
    #plt.legend()
    #plt.legend(prop={'size': 16})

    plt.grid(True)
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()
