import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy import stats


def model(y, t, params):
    T, I, S, V = y  
    gamma, beta, rd, rp, delta, p, c = params  
    
    dTdt = -beta * T * V - ((gamma * T) * (I + S))
    dIdt = (beta * T * V) - ((gamma * I) * (T + (2*I) + S)) - (delta * I)
    dSdt = (gamma * T) * (2*I + S) + ((gamma * I) * (2*I + S)) - (rd * delta * S)
    dVdt = (p * I) + (rp * p * S) - (c * V)
    
    return [dTdt, dIdt, dSdt, dVdt]

def generate_data(params, noise_level, t_points, virus_data):
    noise = stats.norm.rvs(loc=0, scale=noise_level * virus_data, size=virus_data.shape)
    #noise = np.random.normal(0, noise_level * np.max(virus_data), size=virus_data.shape)
    #noise level * actual data point
    #try small amount of noise
    #normal distribution from stats package
    noisy_data = virus_data + noise
    #virua_data + stats.
    #may need to iterate over each point
    #proportiional to actual measuremment
    return noisy_data

def ssr(params, t_points, data):
    y0 = [1, 0, 0, 1]
    sol = odeint(model, y0, t_points, args=(params,))
    virus_model = sol[:,3]
    return np.sum((virus_model - data) ** 2)

def fit_params(initial, t_points, data):
    result = minimize(ssr, initial, args=(t_points, data), method="L-BFGS-B")
    return result.x

def are(true_params, noise_level, t_points, n=100):
    estimates = []
    y0 = [1, 0, 0, 1]
    sol = odeint(model, y0, t_points, args=(true_params,))

    virus_data = sol[:,3]
    for i in range (n):
        data = generate_data(true_params, noise_level, t_points, virus_data)
        est_params = fit_params(true_params, t_points, data)
        print(est_params)
        # exit()
        estimates.append(est_params)

    estimates = np.array(estimates)
    ARE = np.mean(np.abs((true_params - estimates) / true_params), axis=0)
    #ARE = np.mean(np.abs(np.log(estimates) - np.log(true_params)), axis=0)
    #ARE = np.mean((true_params - estimates) / true_params, axis=0)


    return ARE


if __name__ == "__main__":
    true_params = np.array([0.1, 2.04e-8, 2, 0.5, 0.0735, 4.66e6, 0.0763])
    t_points = np.linspace(0, 240, 10)
    noise_levels = np.linspace(0.05, 0.2, 16)
    #noise_levels = [2]
    #240 hours
    #meausre once a day: 24, 48, 72
    # noise_levels = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
    #in betwen by integers

    AREs = []
    for noise in noise_levels:
        AREs.append(are(true_params, noise, t_points))
    AREs = np.array(AREs)

    param_names = ["gamma", "beta", "rd", "rp", "delta", "p", "c"]
    plt.figure(figsize=(8, 6))
    for i, param in enumerate(param_names):
        plt.plot([n * 100 for n in noise_levels], AREs[:, i], marker="o", label=param)


    plt.xlabel("Noise level (100 stats)")
    plt.ylabel("ARE")
    plt.legend()
    plt.grid(True)
    #plt.ylim(0.1, 0.1005)
    #plt.yscale("log")

    plt.show()

#plot of best fit_(virus over time)
#plot of 1 curve with noise (same shape as no noise)
#do 10 (plot 10 lines on a curve)

#implemented below
#initial guess = true_params

    '''noise_level = 0.1
    data = generate_data(true_params, noise_level, t_points)
    est_params = fit_params(true_params, t_points, data)

    y0 = [1, 0, 0, 1]
    sol_fit = odeint(model, y0, t_points, args=(est_params,))
    sol_true = odeint(model, y0, t_points, args=(true_params,))

    plt.figure(figsize=(8, 6))
    plt.scatter(t_points, data, color="red", label="Noisy data", zorder=3)
    plt.plot(t_points, sol_fit[:, 3], "b-", label="Best fit", linewidth=2)
    plt.plot(t_points, sol_true[:, 3], "k--", label="True (noiseless)")
    plt.xlabel("Time")
    plt.ylabel("Virus")
    plt.legend()
    plt.grid(True)
    plt.title("Best-fit curve vs noisy data")
    plt.show()

    # Plot 3: Noisy vs noiseless trajectory
    sol_true = odeint(model, y0, t_points, args=(true_params,))
    data_noisy = generate_data(true_params, noise_level=0.1, t_points=t_points)

    plt.figure(figsize=(8, 6))
    plt.plot(t_points, sol_true[:, 3], "k-", label="Noiseless (true)")
    plt.scatter(t_points, data_noisy, color="red", label="Noisy data")
    plt.xlabel("Time")
    plt.ylabel("Virus")
    plt.legend()
    plt.grid(True)
    plt.title("Noisy vs noiseless trajectory")
    plt.show()

    # Plot 4: 10 fits with noise
    plt.figure(figsize=(8, 6))
    for i in range(10):
        noisy_data = generate_data(true_params, noise_level=0.1, t_points=t_points)
        est_params = fit_params(true_params * 0.9, t_points, noisy_data)
        sol_fit = odeint(model, y0, t_points, args=(est_params,))
        plt.plot(t_points, sol_fit[:, 3], alpha=0.6)

    sol_true = odeint(model, y0, t_points, args=(true_params,))
    plt.plot(t_points, sol_true[:, 3], "k--", linewidth=2, label="True (noiseless)")
    plt.xlabel("Time")
    plt.ylabel("Virus")
    plt.title("10 fits with noise (showing variability)")
    plt.grid(True)
    plt.legend()
    plt.show()'''




