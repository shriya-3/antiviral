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
    result = odeint(model, y0, t, args=(params,))

    T_pred = result[:, 0]
    V_pred = result[:, 3]
    
    T_sum, V_sum = 0, 0
    #epsilon = 1e-10
    for i in range(len(T_pred)):
        T_sum += ((T_data[i]) - np.log10(T_pred[i])) ** 2
        #t_shuffled
    
    for i in range(len(V_pred)):
        V_sum += ((V_data[i]) - np.log10(V_pred[i])) ** 2

    #print(V_pred)

    return T_sum + V_sum

'''def calc_residuals(T_data, T_pred):
    n = len(T_data)
    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = T_pred[i] - T_data[i]
    return residuals'''

def conf_interval(params, index):
    s = sorted(params, key=lambda tup: tup[index])
    return s[1][index], s[-2][index]

def calc_residuals(V_data, T_data, V_pred, T_pred):
    n = len(V_data) + len(T_data)
    residuals_one = np.zeros(len(V_data))
    residuals_two = np.zeros(len(T_data))
    
    for i in range(len(V_data)):
        residuals_one[i] = np.log10(V_pred[i]) - np.log10(V_data[i])
    
    for j in range (len(T_data)):
        residuals_two[j] += np.log10(T_pred[j]) - np.log10(T_data[j])
    
    return np.concatenate((residuals_one, residuals_two))

def boot(V_data, T_data, V_pred, T_pred, initial_guess, y0, t):
    n = len(V_data) + len(T_data)
    residuals = calc_residuals(V_data, T_data, V_pred, T_pred)
    
    params = []
    #1000
    for i in range(100):
        V_sample = np.zeros(len(V_data))
        T_sample = np.zeros(len(T_data))
        random.shuffle(residuals)
        
        c = 0
        for j, k in zip(V_pred, residuals[: len(V_pred)]):
            V_sample[c] = np.log10(j) + k
            c += 1
        print(f"Bootstrap {i + 1} V_sample:", V_sample)
        d = 0
        for j, k in zip(T_pred, residuals[len(V_pred) :]):
            T_sample[d] = np.log10(j) + k
            d += 1
        print(f"Bootstrap {i + 1} T_sample:", T_sample)
        result = minimize(ssr, initial_guess, args=(y0, t, V_sample, T_sample), method="L-BFGS-B")
        #powell, lbgt, newton, sci py minimze 
        estimated_params = 10**result.x
        params.append(estimated_params)

    print(f"lamb: {conf_interval(params, 0)}")
    print(f"beta: {conf_interval(params, 1)}")
    print(f"k: {conf_interval(params, 2)}")
    print(f"delta: {conf_interval(params, 3)}")
    print(f"p: {conf_interval(params, 4)}")
    print(f"c: {conf_interval(params, 5)}")
    print(f"K: {conf_interval(params, 6)}")




def main():
    virus = np.loadtxt('data/virus_purple.dat')
    cells = np.loadtxt('data/cells_purple.dat')
    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    y0 = [10000000, 0, 0, 1] 
    #initial_guess = np.log10([0.6, 0.00001, 1, 1, 0.1, 1, 10000000])

    #YELLOW GUESS
    #initial_guess = np.log10([1.24150227e-25, 1.37693985e-06, 4.08614420e+01, 1.39166123e+02, 4.88445968e+01, 4.19605737e-02, 1.68056424e-18])
    #initial_guess = np.log10([1.33811643e-25, 1.51975522e-06, 2.38853749e+02, 1.65858268e+02, 4.75308073e+01, 2.77989318e-01, 2.87466633e-18]) #BEST
    #initial_guess = np.log10([1.74745889e-25, 1.61921944e-06, 2.41197783e+02, 1.70484991e+02, 4.87953217e+01, 2.61049431e-01, 3.37173075e-18])

    #GREEN GUESS
    #initial_guess = np.log10([2.43911482e-03, 5.65968210e-81, 8.49817074e-01, 1.11561814e+01, 5.75256208e+75, 7.72454120e-01, 4.96999578e+03])
    #initial_guess = np.log10([7.45030763e-05, 6.67527894e-78, 1.07337535e+00, 5.94586265e+04, 8.61326107e+77, 6.52842748e+01, 2.67172008e+02]) #BEST??
    #initial_guess = np.log10([4.59928148e-003, 1.53196095e-313, 1.31504358e+000, 6.88946981e+000, 1.79768995e+308, 9.96115810e-001, 7.69522115e+003])
    #initial_guess = np.log10([3.10640874e-036, 3.52172901e-303, 4.65716190e+006, 1.89411927e+011, 7.42074476e+307, 9.42816370e-001, 9.69921490e-030])

    #RED GUESS
    #initial_guess = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01, 4.96716462e+06])
    #initial_guess = np.log10([1.36703974e-01, 2.20334893e-05, 2.15432783e+00, 2.89684226e+18, 3.12119482e+18, 8.89698760e+01, 8.02762543e+06]) #BEST

    #RED GUESS NEW
    #initial_guess = np.log10([5.17168000e-01, 1.28618844e-02, 1.04391007e+00, 1.03360140e+00, 7.48091293e-05, 9.20837923e-01, 4.61340647e+06])
    #initial_guess = np.log10([1.30662809e-01, 1.60994300e-05, 1.99801204e+01, 3.95619435e+00, 1.63277801e+00, 2.67318059e+01, 5.40522939e+06])

    #GREY GUESS
    #initial_guess = np.log10([4.50923909e-87, 6.25940011e-06, 2.29761332e+01, 6.94252961e+01, 7.76005238e+00, 1.23334214e-04, 3.46906333e-80])
    #initial_guess = np.log10([4.60953276e-87, 5.72620522e-06, 4.46627314e+01, 7.14508556e+01, 7.71044398e+00, 1.82283903e-01, 5.30837209e-80]) #BEST

    #GREY GUESS (NEW)
    #initial_guess = np.log10([6.47942054e-100, 6.10519210e-006, 7.96510066e+001, 1.55192834e-005, 1.38910662e+001, 8.05444176e+001, 3.31034972e-093])
    #initial_guess = np.log10([6.18374636e-100, 6.31615255e-006, 8.00664875e+001, 1.53163810e-005, 1.40496251e+001, 8.09656321e+001, 3.44884042e-093])
    
    # PURPLE GUESS
    initial_guess = np.log10([7.17316553e-01, 3.36823860e-04, 7.83678424e-01, 2.00291307e+00, 7.61300436e-03, 8.72944403e-01, 4.52239206e+06]) #
    #initial_guess = np.log10([1.02605046e-16, 1.20989507e-06, 8.49060327e+10, 2.99309184e+01, 1.07597258e+01, 2.60597208e+00, 2.09812514e+01])
    #initial_guess  = np.log10([3.66105728e-08, 1.53192161e-06, 7.26572130e+10, 3.08258276e+01, 1.12312573e+01, 2.95106325e+00, 6.73958104e+00])
    #initial_guess = np.log10([5.41443456e-09, 3.01249721e-07, 4.27332874e+09, 4.41801056e+01, 1.30759750e+02, 6.02545079e+00, 1.32409159e+00])
    #initial_guess = np.log10([9.67202013e-09, 1.60039525e-07, 1.43015997e+01, 1.04657423e+01, 1.10891046e+02, 7.82884993e+00, 8.08335542e-01]) #BEST
    #initial_guess = [6.00002452e-01, 1.01219057e-04, 1.00000326e+00, 1.00000282e+00, 9.99996157e-02, 9.99993756e-01, 1.00030762e+07]
    #initial_guess = [-4.39625096e-02, 2.46320189e-06, 3.57787192e+01, 3.01255438e+01, 5.67911660e+00, 2.16094191e+00, -1.53964319e+09]
    #initial_guess = [-5.09052792e-02, 1.30277576e-06, 3.32721815e+01, 3.55089210e+01, 1.45941414e+01, 2.61455664e+00, -4.73368976e+09]
    #initial_guess = np.log10([-5.87414646e-02, 7.76668651e-07, 1.21251373e+02,  3.09354163e+01, 2.19928930e+01, 2.72203500e+00, -1.05647347e+11])
    #lamb, beta, k, delta, p, c, K = params

    #result = minimize(ssr, initial_guess, args=(y0, t, V_data, T_data), method="Nelder-Mead")
    estimated_params = [1.02605046e-16, 1.20989507e-06, 8.49060327e+10, 2.99309184e+01, 1.07597258e+01, 2.60597208e+00, 2.09812514e+01]
    #plug in params
    #print("Estimated parameters: ", estimated_params)
    #print(result.fun)

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
    #print(V_data)
    boot(V_data, T_data, V_pred, T_pred, initial_guess, y0, t)
    

if __name__ == "__main__":
    main()



    