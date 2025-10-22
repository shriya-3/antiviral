import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 
from scipy.optimize import minimize


def model(y, t, params):
    K = 5.97e6  # GREY dataset (fixed carrying capacity)

    T, E, I, V = y
    lamb, beta, k, delta, p, c = params

    dTdt = lamb * T * (1 - T / K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])


def ssr(params, y0, t, V_data, T_data):
    """Compute SSR between model predictions and experimental data."""
    result = odeint(model, y0, t, args=(params,))
    T_pred = result[:, 0]
    V_pred = result[:, 3]

    # guard against invalid predictions
    if np.any(T_pred <= 0) or np.any(V_pred <= 0) or np.any(np.isnan(T_pred)) or np.any(np.isnan(V_pred)):
        return 1e12

    T_ssr = np.sum((np.log10(T_data) - np.log10(T_pred)) ** 2)
    V_ssr = np.sum((np.log10(V_data) - np.log10(V_pred)) ** 2)

    return T_ssr + V_ssr


def profile_ssr(fixed_index, fixed_value, best_fit_params, y0, t, V_data, T_data):
    def objective(free_params):
        trial_params = best_fit_params.copy()
        trial_params[fixed_index] = fixed_value
        trial_params[np.arange(len(best_fit_params)) != fixed_index] = free_params
        return ssr(trial_params, y0, t, V_data, T_data)

    init_guess = np.delete(best_fit_params, fixed_index)
    result = minimize(objective, init_guess, method="Nelder-Mead")
    return result.fun

def plot_profile(best_fit_params, y0, t, V_data, T_data):
    param_names = ["lambda", "beta", "k", "delta", "p", "c"]

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
    axes = axes.flatten()

    for i, param in enumerate(best_fit_params):
        scan_vals = np.linspace(0.9 * param, 1.1 * param, 40)  # ±10% scan
        ssr_vals = []

        for val in scan_vals:
            #trial_params = best_fit_params.copy()
            #trial_params[i] = val
            #ssr_val = ssr(np.log10(trial_params), y0, t, V_data, T_data)
            ssr_val = profile_ssr(i, val, best_fit_params, y0, t, V_data, T_data)
            ssr_vals.append(ssr_val)

        axes[i].plot(scan_vals, ssr_vals, marker="o")
        axes[i].set_xlabel(f"{param_names[i]}")
        axes[i].set_ylabel("SSR")
        axes[i].set_title(f"Profile for {param_names[i]}")
        axes[i].grid(True)

    plt.suptitle("Parameter Profile Likelihoods")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Hard-coded dataset (replace with your real experimental data)


    virus = np.loadtxt('data/virus_grey.dat')
    cells = np.loadtxt('data/cells_grey2.dat')

    V_data = virus[:, 1]
    T_data = cells[:, 1]
    t = virus[:, 0]

    # Initial condition
    y0 = [5.97e6, 0, 0, 1]

    # Hard-coded best-fit parameters (replace with your fitted values)
    best_fit_params = np.array([
        0.559,
        9.68e-6,
        97.2,
        282,
        14.5,
        0.560
    ])

    plot_profile(best_fit_params, y0, t, V_data, T_data)
