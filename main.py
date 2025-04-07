import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns




def model(y, t, params):
    T, E, I, V = y
    lamb, beta, k, delta, p, c, K = params

    dTdt = lamb * T * (1-T/K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])


def ssr(params, y0, t, V_data, T_data):
    params = 10**params
    result = odeint(model, y0, t, args=(params,))

    T_pred_ssr = result[:, 0]
    V_pred_ssr = result[:, 3]
    
    T_sum, V_sum = 0, 0
    for i in range(len(T_pred_ssr)):
        T_sum += (np.log10(T_data[i]) - np.log10(T_pred_ssr[i])) ** 2
        #t_shuffled
    
    for i in range(len(V_pred_ssr)):
        V_sum += (np.log10(V_data[i]) - np.log10(V_pred_ssr[i])) ** 2

    #print(V_pred)

    return T_sum + V_sum


def main():
    virus = np.loadtxt('data/virus_purple.dat')
    cells = np.loadtxt('data/cells_purple.dat')
    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    #y0 = [10000000, 0, 0, 1] 
    #y0 = [3123092.42921152, 0, 0, 1] #GREEN
    y0 = [4693206.05, 0, 0, 1] #PURPLE
    #y0 = [4502249.95998941, 0, 0, 1] #RED
    #y0 = [5142701.72055847, 0, 0, 1] #YELLOW

    #FINAL: After re-running initial guesses 
    initial_guess = np.log10([3.34131606e-03, 7.97858286e-07, 1.61285348e+02, 5.17873197e+00, 6.14732745e+01, 2.75111791e+01, 3.44613946e+05]) #PURPLE
    #initial_guess = np.log10([2.15090516e-01, 3.45657502e-05, 1.52896846e+00, 1.69064399e+07, 6.14777220e+06, 9.91326364e+00, 2.99780262e+06]) #GREEN
    #initial_guess = np.log10([2.15090516e-01, 3.45657502e-05, 1.52896846e+00, 1.69064399e+00, 6.14777220e+06, 9.91326364e+00, 2.99780262e+06]) #GREEN fixed
    #initial_guess = np.log10([1.30662809e-01, 1.60994300e-05, 1.99801204e+01, 3.95619435e+00, 1.63277801e+00, 2.67318059e+01, 5.40522939e+06]) #RED
    #initial_guess = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01, 4.96716462e+06]) #YELLOW
    #lamb, beta, k, delta, p, c, K = params


    #lamb, beta, k, delta, p, c, K = params

    result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead")
    #result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Powell")
    estimated_params = 10**result.x
    #plug in params
    print("Estimated parameters: ", estimated_params)
    print(result.fun)

    #model with estimated parameters
    t_new = np.linspace(0, 22, 100); 
    result = odeint(model, y0, t_new, args=(estimated_params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]

    

    #plt.figure(figsize=(0.5, 100))
    
    plt.figure(figsize=(12, 8))
    '''
    plt.scatter(t, np.log10(T_data), color='blue', label='Experimental T data')
    plt.scatter(t, np.log10(V_data), color='red', label='Experimental V data')

    plt.plot(t, np.log10(T_pred), linestyle='-', color='blue', label='Predicted T')
    plt.plot(t, np.log10(V_pred), linestyle='-', color='red', label='Predicted V')
    '''
    params_df = pd.read_csv('purple_samples_test.csv')

    # Select the first 10 rows
    '''
    params_df = params_df.head(100)

    def is_valid_param_set(param_set):
        lamb, beta, k, delta, p, c, K = param_set
        #K = 9.26530430e+05
        print(f"lamb: {lamb}, beta: {beta}, k: {k}, delta: {delta}, p: {p}, c: {c}, K: {K}")
        return (-4 < lamb < 1) and (-10 < beta < -3)  and (-3 < k < 3) and (-3 < delta < 2) and (-3 < p < 4) and (-3 < c < 2) and (3 < K < 20)  #PURPLE



    for _, row in params_df.iterrows():
        mcmc_params = row.values  
        result_new = odeint(model, y0, t, args=(mcmc_params,))
        T_pred = result_new[:, 0]
        V_pred = result_new[:, 3]
        plt.plot(t, T_pred, linestyle='-', color='blue', alpha=0.3)  
        plt.plot(t, V_pred, linestyle='-', color='red', alpha=0.3)

        #if is_valid_param_set(mcmc_params):'''
            

        
    
    plt.scatter(t, T_data, color='blue', label='Experimental Lymphocytes $ml^-1$ data')
    plt.scatter(t, V_data, color='red', label='Experimental Virus Titer data')

    plt.plot(t_new, T_pred, linestyle='-', color='blue', label='Predicted Lymphocytes $ml^-1$')
    plt.plot(t_new, V_pred, linestyle='-', color='red', label='Predicted Virus Titer')

    #plt.title("ERDRP-0519 (3 dpi) (n=3)") #PURPLE
    #plt.title("GHP-88309 (3 dpi) (n=3)") #RED
    #plt.title("GHP-88309 (7 dpi) (n=3)") #YELLOW
    plt.title("GHP-88309 (5 dpi) (n=3)") #GREEN

    plt.xlabel('time post CDV infection (d)')
    plt.yscale('log')
    plt.ylim(10e0, 10e7)
    #plt.ylim([min(T_data.min(), T_pred.min()) * 0.1, max(T_data.max(), T_pred.max()) * 10])  # Adjust dynamically

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig('green_params_one.png')
    plt.show()
    plt.close()
    
    
    print(T_pred)
    print(V_pred)
    
    #print(V_data)
    print("T_pred: ", T_pred)
    print("V_pred: ", V_pred)
    #boot(V_data, T_data, V_pred, T_pred, initial_guess, y0, t)

if __name__ == "__main__":
    main()