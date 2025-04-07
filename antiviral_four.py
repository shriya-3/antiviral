import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import math
import matplotlib.pyplot as plt
import random



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
    #params = params
    result = odeint(model, y0, t, args=(params,))

    T_pred_ssr = result[:, 0]
    V_pred_ssr = result[:, 3]
    
    T_sum, V_sum = 0, 0
    for i in range(len(T_pred_ssr)):
        T_sum += ((T_data[i]) - np.log10(T_pred_ssr[i])) ** 2
        #T_sum += ((T_data[i]) - (T_pred_ssr[i])) ** 2
    
    for i in range(len(V_pred_ssr)):
        V_sum += ((V_data[i]) - np.log10(V_pred_ssr[i])) ** 2
        #V_sum += ((V_data[i]) - (V_pred_ssr[i])) ** 2

    return T_sum + V_sum


def conf_interval(params, index):
    s = sorted(params, key=lambda tup: tup[index])
    return s[1][index], s[-2][index]

def calc_residuals(V_data, T_data, V_pred, T_pred):
    residuals_one = np.zeros(len(V_data))
    residuals_two = np.zeros(len(T_data))
    
    for i in range(len(V_data)):
        residuals_one[i] = np.log10(V_pred[i]) - np.log10(V_data[i])
        #residuals_one[i] = (V_pred[i]) - (V_data[i])
    
    for j in range (len(T_data)):
        residuals_two[j] += np.log10(T_pred[j]) - np.log10(T_data[j])
        #residuals_two[j] += (T_pred[j]) - (T_data[j])
    
    return np.concatenate((residuals_one, residuals_two))

def boot(V_data, T_data, V_pred, T_pred, initial_guess, y0, t):
    n = len(V_data) + len(T_data)
    residuals = calc_residuals(V_data, T_data, V_pred, T_pred)
    
    params = []
    ssr_values = []

    i = 0
    while (i < 1000):

        V_sample = np.zeros(len(V_data))
        T_sample = np.zeros(len(T_data))
        random.shuffle(residuals)
        
        c = 0
        for j, k in zip(V_pred, residuals[: len(V_pred)]):
            V_sample[c] = np.log10(j) + k
            #V_sample[c] = (j) + k
            c += 1
        print(f"Bootstrap {i + 1} V_sample:", V_sample)
        d = 0
        for j, k in zip(T_pred, residuals[len(V_pred) :]):
            T_sample[d] = np.log10(j) + k
            #T_sample[d] = (j) + k
            d += 1
        print(f"Bootstrap {i + 1} T_sample:", T_sample)

        result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="L-BFGS-B")
        
        '''if (result.success):
            #result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="Powell")
            #result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="TNC", bounds=[(0, None)] * len(initial_guess))

            #powell, lbgt, newton, sci py minimze 
            
            estimated_params = 10**result.x
            #estimated_params = result.x
            params.append(estimated_params)
            #print params

            ssr_value = ssr(result.x, y0, t, V_sample, T_sample)
            ssr_values.append(ssr_value)

            print(f"Bootstrap {i+1} Estimated Params: ", estimated_params)
            print(f"Bootstrap {i+1} SSR: ", ssr_value)
            i += 1'''
        ssr_value = ssr(result.x, y0, t, V_sample, T_sample)
        #Purple: ssr_value > 0 and ssr_value < 5 
        if (result.success):
            #result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="Powell")
            #result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="TNC", bounds=[(0, None)] * len(initial_guess))

            #powell, lbgt, newton, sci py minimze 
            
            estimated_params = 10**result.x
            #estimated_params = result.x
            params.append(estimated_params)
            #print params

            #ssr_value = ssr(result.x, y0, t, V_sample, T_sample)
            ssr_values.append(ssr_value)

            print(f"Bootstrap {i+1} Estimated Params: ", estimated_params)
            print(f"Bootstrap {i+1} SSR: ", ssr_value)
            i += 1


    #K big range
    #k, delta, c -bigger range
    print(f"lamb: {conf_interval(params, 0)}")
    print(f"beta: {conf_interval(params, 1)}")
    print(f"k: {conf_interval(params, 2)}")
    print(f"delta: {conf_interval(params, 3)}")
    print(f"p: {conf_interval(params, 4)}")
    print(f"c: {conf_interval(params, 5)}")
    print(f"K: {conf_interval(params, 6)}")

    return params, ssr_values

def save_to_csv(params, ssr_values):
    params_array = np.array(params)
    data_to_save = np.column_stack((ssr_values, params_array))
    np.savetxt('bootstrap_results_red_trial_1.csv', data_to_save, delimiter=',',
               header='SSR, lamb, beta, k, delta, p, c, K', comments='')


def main():
    virus = np.loadtxt('data/virus_red.dat')
    cells = np.loadtxt('data/cells_red2.dat')
    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    y0 = [10000000, 0, 0, 1] 
    
    #initial_guess = np.log10([0.6, 0.00001, 1, 1, 0.1, 1, 10000000])
    
    #initial_guess = np.log10([7.17316553e-01, 3.36823860e-04, 7.83678424e-01, 2.00291307e+00, 7.61300436e-03, 8.72944403e-01, 4.52239206e+06]) #purple
    #estimated_params = [1.02605046e-16, 1.20989507e-06, 8.49060327e+10, 2.99309184e+01, 1.07597258e+01, 2.60597208e+00, 2.09812514e+01] #purple
    
    #initial_guess = [7.17316553e-01, 3.36823860e-04, 7.83678424e-01, 2.00291307e+00, 7.61300436e-03, 8.72944403e-01, 4.52239206e+06] #

    
    #initial_guess = np.log10([6.07587120e-08, 3.17711949e-08, 1.02084761e+01, 9.72035665e+00, 8.68862663e+02, 1.31153509e+01, 4.88886045e+00])

    #yellow
    #initial_guess = np.log10([1.33811643e-25, 1.51975522e-06, 2.38853749e+02, 1.65858268e+02, 4.75308073e+01, 2.77989318e-01, 2.87466633e-18]) #BEST

    #red
    initial_guess = np.log10([1.36703974e-01, 2.20334893e-05, 2.15432783e+00, 2.89684226e+18, 3.12119482e+18, 8.89698760e+01, 8.02762543e+06]) #BEST


    
    #estimated_params = [5.89226887e-08, 3.13602300e-08, 1.01182394e+01, 1.00424615e+01, 8.81551632e+02, 1.27346964e+01, 4.75467965e+00]

    #params
    
    #yellow
    #estimated_params = [1.74745889e-25, 1.61921944e-06, 2.41197783e+02, 1.70484991e+02, 4.87953217e+01, 2.61049431e-01, 3.37173075e-18]

    #red
    estimated_params = [1.38046171e-01, 2.01599421e-05, 2.15005723e+00, 2.53791597e+18, 2.98546588e+18, 8.91172159e+01, 8.38024108e+06]

    #model with estimated parameters
    result = odeint(model, y0, t, args=(estimated_params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]

    #plt.figure(figsize=(0.5, 100))
    '''
    plt.figure(figsize=(12, 8))

    plt.scatter(t, np.log10(T_data), color='blue', label='Experimental T data')
    plt.scatter(t, np.log10(V_data), color='red', label='Experimental V data')

    plt.plot(t, np.log10(T_pred), linestyle='-', color='blue', label='Predicted T')
    plt.plot(t, np.log10(V_pred), linestyle='-', color='red', label='Predicted V')

    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('purple_one.png')
    plt.show()
    plt.close()
    
    print(T_pred)
    print(V_pred)
    '''
    print("T_pred: ", T_pred)
    print("V_pred: ", V_pred)
    params, ssr_values = boot(V_data, T_data, V_pred, T_pred, initial_guess, y0, t)
    save_to_csv(params, ssr_values)
    

if __name__ == "__main__":
    main()



    