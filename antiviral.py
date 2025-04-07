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
    K = 5142701.720558477 # YELLOW
    #K = 5972735.535321577 #GREY

    T, E, I, V = y
    lamb, beta, k, delta, p, c  = params

    dTdt = lamb * T * (1 - T / K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])


def ssr_basic(pred, true):
    """Compute the sum of squared residuals between predicted and true values."""
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
    """Compute the SSR for a set of MCMC parameters."""
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

    virus = np.loadtxt('data/virus_yellow.dat')
    cells = np.loadtxt('data/cells_yellow2.dat')

    #virus = np.loadtxt('data/virus_grey.dat')
    #cells = np.loadtxt('data/cells_grey2.dat')

    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    #PURPLE DATASET
    #y0 = [4767840.060099503, 0, 0, 1]  # Initial conditions for purple dataset
    #initial_guess = np.log10([3.34131606e-03, 7.97858286e-07, 1.61285348e+02, 5.17873197e+00, 
                               #6.14732745e+01, 2.75111791e+01, 3.44613946e+05])   #Purple
    #initial_guess = np.log10([7.24985555e-01, 2.50715804e-08, 2.61706640e+01, 5.24540588e+01, 8.67834551e+03, 1.44740750e+01])
    #initial_guess = np.log10([5.76489543e-01, 1.00211170e-04, 1.59401898e+02, 2.44051237e+02, 1.70054323e+00, 1.70663815e+00])
    #initial_guess = np.log10([5.18722676e-01, 8.16929473e-05, 2.72999599e+12, 2.55144891e+02, 2.38201727e+00, 2.03015420e+00])
    #initial_guess = np.log10([5.08695517e-01, 7.75800234e-05, 5.21452736e+11, 2.90082169e+02, 2.92194289e+00, 2.12307842e+00])
    #initial_guess = np.log10([5.08695517e-01, 7.75800234e-07, 5.21452736e+11, 2.90082169e+02, 2.92194289e+00, 2.12307842e+00])
    #initial_guess = np.log10([7.17316553e-01, 3.36823860e-04, 7.83678424e-01, 2.00291307e+00, 7.61300436e-03, 8.72944403e-01])
    
    #initial_guess  = np.log10([9.74976275e-02, 2.13536947e-06, 9.34478242e+01, 3.13379594e+01, 9.38060100e+00, 1.29855834e+00])
    #initial_guess = np.log10([9.74976275e-02, 2.13536947e-06, 9.34478242e+01, 3.13379594e+01, 9.92056554e+00, 1.30706746e+00]) #MOST RECENT

    #GREEN DATASET
    #y0 = [3123092.42921152, 0, 0, 1]
    #initial_guess = np.log10([7.45030763e-05, 6.67527894e-78, 1.07337535e+00, 5.94586265e+04, 8.61326107e+77, 6.52842748e+01])
    #initial_guess = np.log10([3.70953349e-14, 1.45018186e-01, 1.36542113e+00, 1.81089115e+05, 2.22640356e-08, 1.42935715e-07])
    #initial_guess = np.log10([2.67887319e-13, 1.45855792e-01, 2.06997619e+00, 2.98439478e+06, 5.68800150e+00, 2.11636568e-01])
    #initial_guess = np.log10([2.28997907e-01, 3.33826044e-05, 5.49055206e+00, 2.75346933e+02, 1.74508957e+01, 1.67540179e+00])
    #initial_guess = np.log10([2.28996755e-01, 3.33833763e-05, 5.49056759e+00, 2.75241904e+02, 1.74436279e+01, 1.67540367e+00]) #MOST RECENT

    #RED DATASET
    #y0 = [4502249.959989419, 0, 0, 1]
    #initial_guess = np.log10([1.36703974e-01, 2.20334893e-05, 2.15432783e+00, 2.89684226e+18, 3.12119482e+18, 8.89698760e+01])
    #initial_guess = np.log10([1.22760402e-01, 9.85543908e-06, 3.20460729e+00, 2.14665969e+17, 4.29331728e+17, 4.33646054e+01])
    #initial_guess = np.log10([1.15198344e-01, 1.18099001e-05, 3.34894348e+00, 2.75255359e+17, 3.53503799e+17, 3.48296812e+01])
    #initial_guess = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01])
    
    #initial_guess = np.log10([1.00879387e-01, 1.49584807e-05, 2.34061063e+03,	2.00029305e+01,	2.82975144e+00,	6.00281280e+00]) # MOST RECENT

    #YELLOW DATASET
    y0 = [5142701.720558477, 0, 0, 1]
    #initial_guess = np.log10([1.33811643e-3, 1.51975522e-06, 2.38853749e+02, 1.65858268e+02, 4.75308073e+01, 2.77989318e-01])
    #initial_guess = np.log10([2.15757047e-10, 4.17204494e-10, 9.27997087e-01, 4.29260067e+02, 8.63388019e+07, 4.57617113e+01])
    #initial_guess = np.log10([1.00879387e-01, 1.49584807e-05, 2.34061063e+03,	2.00029305e+01,	2.82975144e+00,	6.00281280e+00])
    #initial_guess = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01])
    #initial_guess = np.log10([1.53601571e-02, 2.41945152e-06, 8.33933957e+01, 3.71390854e+00, 2.05764514e+00, 5.71621797e-01]) #ok
    initial_guess = np.log10([3.47942871e-01, 2.87270668e-06, 1.03655311e+1, 2.54726019e+01, 4.24845402e+00, 5.34303464e-01]) #last used
    #initial_guess = np.log10([3.47942871e-01, 2.87270668e-06, 1.03655311e+11, 2.54726019e+01, 4.24845402e+00, 5.34303464e-01])
    #initial_guess = np.log10([1.48251874e-01, 1.84039128e-06, 3.42223960e+07, 1.26325461e+08, 3.18247380e+07, 5.77409907e-01])
    #initial_guess = np.log10([1.28412369e-03, 1.38021791e-06, 2.71869278e+02, 1.57844300e+02, 4.43770414e+01, 2.72085443e-01])

    #GREY DATASET
    #y0 = [5972735.535321577, 0, 0, 1]
    #initial_guess = np.log10([9.74976275e-02, 2.13536947e-06, 9.34478242e+01, 3.13379594e+01, 9.92056554e+00, 1.30706746e+00])
    #initial_guess = np.log10([6.41622613e-02, 2.14293620e-06, 1.10951244e+02, 3.18296531e+01, 1.04259314e+01, 1.26635932e+00])
    #initial_guess = np.log10([6.52013669e-02, 2.13584266e-06, 1.28425088e+02, 3.41105979e+01, 1.10587889e+01, 1.25955490e+00])
    #initial_guess = np.log10([2.28996755e-01, 3.33833763e-05, 5.49056759e+00, 2.75241904e+02, 1.74436279e+01, 1.67540367e+00])
    
    #initial_guess = np.log10([2.36379119e-01, 5.37849597e-06, 2.90929667e+00, 2.99263225e+02, 6.13611138e+01, 1.42072123e+00])
    #initial_guess = np.log10([3.47942871e-01, 2.87270668e-06, 1.03655311e+1, 2.54726019e+01, 4.24845402e+00, 5.34303464e-01])
    #initial_guess = np.log10([5.65271052e-01, 9.70303786e-06, 9.72892766e+01, 2.82402642e+02, 1.45084101e+01, 5.60841578e-01]) # last used

    result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead")
    estimated_params = 10**result.x
    print(estimated_params)
   
    t_new = np.linspace(0, 22, 100)
    #t_new = np.linspace(0, 12, 100) #GREY
    result = odeint(model, y0, t_new, args=(estimated_params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]

    
    # Load MCMC parameter samples
    #params_df = pd.read_csv('format_green_Kfixed.csv') #GREEN
    #params_df = pd.read_csv('format_purple_Kfixed1_new.csv') #PURPLE
    #params_df = pd.read_csv('format_red_Kfixed1_new.csv') #RED
    params_df = pd.read_csv('format_yellow_Kfixed2_new.csv') #YELLOW
    #params_df = pd.read_csv('format_grey_Kfixed1_new.csv') #GREY

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
    
    
    # Overlay the original predicted line for Lymphocytes and Virus Titer
    plt.plot(t_new, T_pred, color='blue', linestyle='--', label='Predicted Lymphocytes', linewidth=2)
    plt.plot(t_new, V_pred, color='red', linestyle='--', label='Predicted Virus Titer', linewidth=2)
    
    # Scatter plot for experimental data
    plt.scatter(t, T_data, color='blue', label='Experimental Lymphocytes $ml^{-1}$ data', zorder=5)
    plt.scatter(t, V_data, color='red', label='Experimental Virus Titer data', zorder=5)

    # Customize the plot
    plt.xlabel('Time post CDV infection (days)')
    plt.ylabel('Concentration')
    plt.yscale('log')
    plt.ylim(1e-5, 1e15)
    plt.title("MCMC Samples: GHP-88309 (7 dpi) (n=3)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()



if __name__ == "__main__":
    main()
