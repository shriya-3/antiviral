import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import pandas as pd


def model(y, t, params):
    #K= 4767840.060099503 #PURPLE
    #K = 3123092.42921152 #GREEN
    #K = 4502249.959989419 #RED
    #K = 5142701.720558477 # YELLOW
    K = 5972735.535321577 #GREY

    T, E, I, V = y
    lamb, beta, k, delta, p, c  = params

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
        #T_sum += (np.log10(T_data[i]) - (np.log10((T_pred_ssr[i]) + (E_pred_ssr[i]) + (I_pred_ssr[i])))) ** 2
        T_sum += (np.log10(T_data[i]) - (np.log10(T_pred_ssr[i]))) ** 2
        #print(T_sum)

    for i in range(len(V_pred_ssr)):
        V_sum += (np.log10(V_data[i]) - np.log10(V_pred_ssr[i])) ** 2

    return T_sum + V_sum

def ssr_for_mcmc(params, y0, t, V_data, T_data):
    result = odeint(model, y0, t, args=(params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]
    
    T_ssr = ssr_basic(T_pred, T_data)
    V_ssr = ssr_basic(V_pred, V_data)
    
    return T_ssr + V_ssr


def main():
    #virus = np.loadtxt('data/virus_purple.dat')
    #cells = np.loadtxt('data/cells_purple.dat')

    #virus = np.loadtxt('data/virus_green.dat')
    #cells = np.loadtxt('data/cells_green2.dat')

    #virus = np.loadtxt('data/virus_red.dat')
    #cells = np.loadtxt('data/cells_red2.dat')

    #virus = np.loadtxt('data/virus_yellow.dat')
    #cells = np.loadtxt('data/cells_yellow2.dat')

    virus = np.loadtxt('data/virus_grey.dat')
    cells = np.loadtxt('data/cells_grey2.dat')

    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    #PURPLE DATASET
    #y0 = [4767840.060099503, 0, 0, 1]  # Initial conditions for purple dataset
    #initial_guess = np.log10([9.74976275e-02, 2.13536947e-06, 9.34478242e+01, 3.13379594e+01, 9.92056554e+00, 1.30706746e+00]) #MOST RECENT

    #GREEN DATASET
    #y0 = [3123092.42921152, 0, 0, 1]
    #initial_guess = np.log10([2.28996755e-01, 3.33833763e-05, 5.49056759e+00, 2.75241904e+02, 1.74436279e+01, 1.67540367e+00]) #MOST RECENT

    #RED DATASET
    #y0 = [4502249.959989419, 0, 0, 1]
    #initial_guess = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01])
    
    #initial_guess = np.log10([1.00879387e-01, 1.49584807e-05, 2.34061063e+03,	2.00029305e+01,	2.82975144e+00,	6.00281280e+00]) # MOST RECENT

    #YELLOW DATASET
    #y0 = [5142701.720558477, 0, 0, 1]
    #initial_guess = np.log10([3.47942871e-01, 2.87270668e-06, 1.03655311e+1, 2.54726019e+01, 4.24845402e+00, 5.34303464e-01]) #last used


    #GREY DATASET
    y0 = [5972735.535321577, 0, 0, 1]
    initial_guess = np.log10([5.65271052e-01, 9.70303786e-06, 9.72892766e+01, 2.82402642e+02, 1.45084101e+01, 5.60841578e-01]) # last used

    result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead")
    estimated_params = 10**result.x
    print(estimated_params)

    plot_profile(estimated_params, y0, t, V_data, T_data)

   
    #t_new = np.linspace(0, 22, 100)
    t_new = np.linspace(0, 12, 100) #GREY
    result = odeint(model, y0, t_new, args=(estimated_params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]


    
    # Load MCMC parameter samples
    #params_df = pd.read_csv('format_green_Kfixed.csv') #GREEN
    #params_df = pd.read_csv('format_purple_Kfixed1_new.csv') #PURPLE
    #params_df = pd.read_csv('format_red_Kfixed1_new.csv') #RED
    #params_df = pd.read_csv('format_yellow_Kfixed2_new.csv') #YELLOW
    params_df = pd.read_csv('format_grey_Kfixed1_new.csv') #GREY

    # Arrays to store predictions from all samples
    T_preds_all = []
    V_preds_all = []
    ssr_values = []

    
    for _, row in params_df.iloc[:1000].iterrows():
        mcmc_params = row.values[:-1]
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
    #ssr_threshold = 15  # Use a fixed threshold for testing

    # Plot the MCMC sample lines that are reasonably close to the original predictions
    plt.figure(figsize=(12, 8))
    
    sample_count = 0
    for i in range(T_preds_all.shape[0]):
        #if ssr_values[i] < ssr_threshold:
            plt.plot(t_new, T_preds_all[i], color='blue', alpha=0.1, linewidth=0.1)
            plt.plot(t_new, V_preds_all[i], color='red', alpha=0.1, linewidth=0.1)
            sample_count += 1
    #print(f"Number of MCMC samples plotted: {sample_count}")
    
    
    plt.plot(t_new, T_pred, color='blue', linestyle='--', label='Predicted Lymphocytes', linewidth=2)
    plt.plot(t_new, V_pred, color='red', linestyle='--', label='Predicted Virus Titer', linewidth=2)
    
    plt.scatter(t, T_data, color='blue', label='Experimental Lymphocytes $ml^{-1}$ data', zorder=5)
    plt.scatter(t, V_data, color='red', label='Experimental Virus Titer data', zorder=5)

    # Customize the plot
    '''plt.xlabel('Time post CDV infection (days)')
    plt.ylabel('Concentration')
    plt.yscale('log')
    plt.ylim(1e-5, 1e15)
    plt.title("MCMC Samples: Vehicle (n=3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()'''

def plot_profile(best_fit_params, y0, t, V_data, T_data):
    param_names = ["lambda", "beta", "k", "delta", "p", "c"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))  
    axes = axes.flatten()  

    for i, param in enumerate(best_fit_params):
        scan_vals = np.linspace(0.9 * param, 1.1 * param, 40)
        ssr_vals = []

        for val in scan_vals:
            trial_params = best_fit_params.copy()
            trial_params[i] = val
            ssr_val = ssr(np.log10(trial_params), y0, t, V_data, T_data)
            ssr_vals.append(ssr_val)

        axes[i].plot(scan_vals, ssr_vals, marker="o")
        axes[i].set_xlabel(f"{param_names[i]}")
        axes[i].set_ylabel("SSR")
        axes[i].set_title(f"Profile for {param_names[i]}")
        axes[i].grid(True)
        
    plt.suptitle("Vehicle")
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
