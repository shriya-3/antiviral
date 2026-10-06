from __future__ import print_function, division

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

configs = {

    "purple": {
        "virus_file": "data/virus_purple.dat",
        "cells_file": "data/cells_purple.dat",
        "K": 4767840.060099503,
        "dpi": 3,
        "initial_guess": [2.15340282e-06, 3.18465532e+01, 9.67901147e+00, 1.31840638e+00],
        "fig": "final_fig_purple.png",
        "title": "ERDRP-0519 (3 dpi)"
    },

    "green": {
        "virus_file": "data/virus_green.dat",
        "cells_file": "data/cells_green2.dat",
        "K": 3123092.42921152,
        "dpi": 5,
        "initial_guess": [3.33848192e-05, 2.75132844e+02, 1.74369880e+01, 1.67546080e+00],
        "fig": "final_fig_green.png",
        "title": "GHP-88309 (5 dpi)"
    },

    "red": {
        "virus_file": "data/virus_red.dat",
        "cells_file": "data/cells_red2.dat",
        "K": 4502249.959989419,
        "dpi": 3,
        "initial_guess": [1.71571339e-05, 3.01583306e+01, 2.61835030, 4.06336670],
        "fig": "final_fig_red.png",
        "title": "GHP-88309 (3 dpi)"
    },

    "yellow": {
        "virus_file": "data/virus_yellow.dat",
        "cells_file": "data/cells_yellow2.dat",
        "K": 5142701.720558477,
        "dpi": 7,
        "initial_guess": [2.45361792e-06, 2.22159842e+01, 4.59566061e+00, 5.37628973e-01],
        "fig": "final_fig_yellow.png",
        "title": "GHP-88309 (7 dpi)"
    },

    "grey": {
        "virus_file": "data/virus_grey.dat",
        "cells_file": "data/cells_grey2.dat",
        "K": 5972735.535321577,
        "dpi": None,
        "initial_guess": [9.68398003e-06, 2.82016815e+02, 1.45086952e+01, 5.59776275e-01],
        "fig": "final_fig_grey.png",
        "title": "Vehicle"
    }
}



# LOAD DATA


datasets = {}

for group, cfg in configs.items():

    virus = np.loadtxt(cfg["virus_file"])
    cells = np.loadtxt(cfg["cells_file"])

    datasets[group] = {

        "t": virus[:, 0],

        # keep raw values for plotting
        "V": virus[:, 1],
        "T": cells[:, 1],

        "K": cfg["K"],

        "dpi": cfg["dpi"],

        "title": cfg["title"],

        "fig": cfg["fig"]
    }

lamb = 0.085
k = 3.0

def model(y, t, params, K):

    T, E, I, V = y

    beta, delta, p, c = params

    dTdt = lamb * T * (1 - T / K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])


# PARAMETER UNPACKING
def unpack_parameters(log_params):
    params = 10 ** log_params

    shared = params[0:4]
    purple = params[4:8]
    green = params[8:12]
    red = params[12:16]
    yellow = params[16:20]

    return {
        "shared": shared,
        "purple": purple,
        "green": green,
        "red": red,
        "yellow": yellow
    }


# VEHICLE SIMULATION
def simulate_vehicle(t, K, shared_params):
    y0 = [K, 0, 0, 1]
    result = odeint(model, y0, t, args=(shared_params, K))
    return result


# TWO-STAGE TREATMENT SIMULATION
def simulate_treatment(t, K, dpi, shared_params, treatment_params):

    """
    Two-stage simulation.

    If dpi is not an experimental time point,
    it is inserted into the time vector.

    Example:

    Original:
    [1,3,7,10,14,21]

    dpi=5

    Becomes:
    [1,3,5,7,10,14,21]
    """

    y0 = [K,0,0,1]

    # Insert treatment day if missing
    if dpi not in t:
        t_full = np.sort(np.append(t, dpi))
    else:
        t_full = t.copy()

    # Split around treatment day
    t_pre = t_full[t_full <= dpi]
    t_post = t_full[t_full >= dpi]


    # Before treatment
    sol_pre = odeint(model, y0, t_pre, args=(shared_params,K))



    # After treatment
    y0_post = sol_pre[-1]
    sol_post = odeint(model, y0_post, t_post, args=(treatment_params,K))


    solution_full = np.vstack((sol_pre, sol_post[1:]))


    # Remove artificial dpi point
    #
    # Return only original experimental points

    mask = np.isin(t_full, t)
    solution_final = solution_full[mask]


    return solution_final



# SSR CALCULATION


def calculate_ssr(solution, T_data, V_data):

    """
    Calculates SSR on log10 scale.
    """

    T_pred = solution[:,0]
    V_pred = solution[:,3]


    T_error = (np.log10(T_data) - np.log10(T_pred)) ** 2
    V_error = (np.log10(V_data) - np.log10(V_pred)) ** 2


    return np.sum(T_error) + np.sum(V_error)

# GLOBAL SSR FUNCTION
def global_ssr(log_params):

    """
    Global objective function.

    Parameter structure:

    0-3:
        shared pre-treatment parameters

    4-7:
        purple post-treatment parameters

    8-11:
        green post-treatment parameters

    12-15:
        red post-treatment parameters

    16-19:
        yellow post-treatment parameters


    Returns:
        total SSR across all groups
    """


    # Unpack parameters
    parameter_sets = unpack_parameters(log_params)
    shared = parameter_sets["shared"]
    purple_params = parameter_sets["purple"]
    green_params = parameter_sets["green"]
    red_params = parameter_sets["red"]
    yellow_params = parameter_sets["yellow"]
    total_ssr = 0

    # VEHICLE
    vehicle = datasets["grey"]
    vehicle_solution = simulate_vehicle(vehicle["t"], vehicle["K"], shared)
    total_ssr += calculate_ssr(vehicle_solution, vehicle["T"], vehicle["V"])

    # PURPLE
    purple = datasets["purple"]
    purple_solution = simulate_treatment(purple["t"], purple["K"], purple["dpi"], shared, purple_params)
    total_ssr += calculate_ssr(purple_solution, purple["T"], purple["V"])

    # GREEN
    green = datasets["green"]
    green_solution = simulate_treatment(green["t"], green["K"], green["dpi"], shared, green_params)
    total_ssr += calculate_ssr(green_solution, green["T"], green["V"])

    # RED
    red = datasets["red"]
    red_solution = simulate_treatment(red["t"], red["K"], red["dpi"], shared, red_params)
    total_ssr += calculate_ssr(red_solution, red["T"], red["V"])

    # YELLOW
    yellow = datasets["yellow"]
    yellow_solution = simulate_treatment(yellow["t"], yellow["K"], yellow["dpi"], shared, yellow_params)
    total_ssr += calculate_ssr(yellow_solution, yellow["T"], yellow["V"])


    return total_ssr


# GLOBAL SSR FUNCTION
def global_ssr(log_params):

    """
    Global objective function.

    Parameter structure:

    0-3:
        shared pre-treatment parameters

    4-7:
        purple post-treatment parameters

    8-11:
        green post-treatment parameters

    12-15:
        red post-treatment parameters

    16-19:
        yellow post-treatment parameters


    Returns:
        total SSR across all groups
    """


    # Unpack parameters

    parameter_sets = unpack_parameters(log_params)
    shared = parameter_sets["shared"]
    purple_params = parameter_sets["purple"]
    green_params = parameter_sets["green"]
    red_params = parameter_sets["red"]
    yellow_params = parameter_sets["yellow"]
    total_ssr = 0



    # VEHICLE
    vehicle = datasets["grey"]
    vehicle_solution = simulate_vehicle(
        vehicle["t"],
        vehicle["K"],
        shared
    )


    total_ssr += calculate_ssr(
        vehicle_solution,
        vehicle["T"],
        vehicle["V"]
    )



    # PURPLE

    purple = datasets["purple"]

    purple_solution = simulate_treatment(
        purple["t"],
        purple["K"],
        purple["dpi"],
        shared,
        purple_params
    )


    total_ssr += calculate_ssr(
        purple_solution,
        purple["T"],
        purple["V"]
    )



    # GREEN

    green = datasets["green"]

    green_solution = simulate_treatment(
        green["t"],
        green["K"],
        green["dpi"],
        shared,
        green_params
    )


    total_ssr += calculate_ssr(
        green_solution,
        green["T"],
        green["V"]
    )



    # RED

    red = datasets["red"]

    red_solution = simulate_treatment(
        red["t"],
        red["K"],
        red["dpi"],
        shared,
        red_params
    )


    total_ssr += calculate_ssr(
        red_solution,
        red["T"],
        red["V"]
    )



    # YELLOW

    yellow = datasets["yellow"]

    yellow_solution = simulate_treatment(
        yellow["t"],
        yellow["K"],
        yellow["dpi"],
        shared,
        yellow_params
    )


    total_ssr += calculate_ssr(
        yellow_solution,
        yellow["T"],
        yellow["V"]
    )


    return total_ssr


# OPTIMIZATION


def run_optimization():

    # Initial guess

    """
    Use the previous individual fits as starting guesses
    for the post-treatment parameters.

    For shared parameters, use the vehicle initial guess.
    """


    shared_guess = configs["grey"]["initial_guess"]


    purple_guess = configs["purple"]["initial_guess"]

    green_guess = configs["green"]["initial_guess"]

    red_guess = configs["red"]["initial_guess"]

    yellow_guess = configs["yellow"]["initial_guess"]



    initial_guess = np.concatenate(
        [
            shared_guess,
            purple_guess,
            green_guess,
            red_guess,
            yellow_guess
        ]
    )


    # optimizer works in log10 space

    initial_guess = np.log10(initial_guess)



    # Bounds

    """
    Same bounds applied to every parameter.

    beta:
        10^-5 to 10^2

    delta:
        10^-2 to 10^4

    p:
        10^-2 to 10^4

    c:
        10^-5 to 10^2

    Since parameters are optimized in log10 space,
    the bounds are already logarithmic.
    """


    single_bounds = [
        (-5, 2),   # beta, phi
        (-2, 4),   # delta, rho
        (-2, 4),   # p
        (-5, 2)    # c

    ]


    bounds = (
        single_bounds
        +
        single_bounds
        +
        single_bounds
        +
        single_bounds
        +
        single_bounds
    )



    # Run optimizer

    result = minimize(
        global_ssr,
        initial_guess,
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxiter": 100000,
            "disp": True
        }
    )



    # Results

    print("\nOptimization complete")

    print("----------------------------")

    print("Final SSR:")
    print(global_ssr(result.x))


    fitted = unpack_parameters(result.x)



    print("\nShared parameters:")
    print(fitted["shared"])


    print("\nPurple parameters:")
    print(fitted["purple"])


    print("\nGreen parameters:")
    print(fitted["green"])


    print("\nRed parameters:")
    print(fitted["red"])


    print("\nYellow parameters:")
    print(fitted["yellow"])



    return result, fitted


# FINAL FIT SIMULATIONS


def generate_final_fits(fitted_params):


    final_fits = {}

    shared = fitted_params["shared"]



    # Vehicle

    vehicle = datasets["grey"]

    final_fits["grey"] = simulate_vehicle(
        vehicle["t"],
        vehicle["K"],
        shared
    )



    # Purple

    purple = datasets["purple"]

    final_fits["purple"] = simulate_treatment(
        purple["t"],
        purple["K"],
        purple["dpi"],
        shared,
        fitted_params["purple"]
    )



    # Green

    green = datasets["green"]

    final_fits["green"] = simulate_treatment(
        green["t"],
        green["K"],
        green["dpi"],
        shared,
        fitted_params["green"]
    )



    # Red

    red = datasets["red"]

    final_fits["red"] = simulate_treatment(
        red["t"],
        red["K"],
        red["dpi"],
        shared,
        fitted_params["red"]
    )



    # Yellow

    yellow = datasets["yellow"]

    final_fits["yellow"] = simulate_treatment(
        yellow["t"],
        yellow["K"],
        yellow["dpi"],
        shared,
        fitted_params["yellow"]
    )


    return final_fits





# PLOTTING


def plot_fit(group, solution):


    data = datasets[group]


    t = data["t"]

    T_data = data["T"]

    V_data = data["V"]



    T_pred = solution[:,0]

    V_pred = solution[:,3]



    plt.figure(figsize=(12,8))


    plt.plot(
        t,
        T_pred,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Predicted Lymphocytes"
    )


    plt.plot(
        t,
        V_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Predicted Virus Titer"
    )



    plt.scatter(
        t,
        T_data,
        color="blue",
        s=70,
        label="Experimental Lymphocytes",
        zorder=5
    )


    plt.scatter(
        t,
        V_data,
        color="red",
        s=70,
        label="Experimental Virus",
        zorder=5
    )



    plt.rcParams.update(
        {"font.size":23}
    )


    plt.xlabel(
        "Time post CDV infection (days)",
        fontsize=20
    )


    plt.ylabel(
        "Concentration",
        fontsize=20
    )


    plt.yscale("log")


    plt.xticks(fontsize=15)

    plt.yticks(fontsize=15)


    plt.ylim(
        1e-1,
        1e7
    )


    plt.xlim(
        0,
        22
    )


    plt.title(
        data["title"]
    )


    plt.grid(True)

    plt.tight_layout()



    plt.savefig(
        data["fig"],
        dpi=300
    )


    plt.show()






# MAIN


if __name__ == "__main__":


    # optimize all groups simultaneously

    result, fitted_params = run_optimization()



    # generate final trajectories

    final_fits = generate_final_fits(
        fitted_params
    )



    # plot all five groups

    for group in [
        "grey",
        "purple",
        "green",
        "red",
        "yellow"
    ]:

        plot_fit(
            group,
            final_fits[group]
        )