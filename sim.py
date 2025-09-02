import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def model(y, t, params, K):
    T, E, I, V = y
    lamb, beta, k, delta, p, c = params

    dTdt = lamb * T * (1 - T / K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return [dTdt, dEdt, dIdt, dVdt]

def sim(y0, t, base_params, affected_indices, K, mode):
    epsilons = np.linspace(0, 1, 50)
    peak_vals = []
    peak_times = []

    for epsilon in epsilons:
        params_with_drug = base_params.copy()
        for i in affected_indices:
            if mode == 'purple':
                params_with_drug[i] *= (1 - epsilon)
            elif mode == 'red':
                params_with_drug[i] /= (1 - epsilon) 

        sol = odeint(model, y0, t, args=(params_with_drug, K))
        V = sol[:, 3]

        peak_vals.append(np.max(V))
        peak_times.append(t[np.argmax(V)])

    return epsilons, np.array(peak_vals), np.array(peak_times)


def main():
    t = np.linspace(0, 50, 200)

    # PURPLE: affects lamb (0) and beta (1)
    #K_purple = 4767840.060099503 #PURPLE
    K_purple = 5972735.535321577
    #purple_params = [9.58646156e-02, 2.15340282e-06, 9.65755873e+01, 3.18465532e+01, 9.67901147e+00, 1.31840638e+00]
    purple_params = [
    5.59242217e-01,
    9.68398003e-06,
    9.72039172e+01,
    2.82016815e+02,
    1.45086952e+01,
    5.59776275e-01
]

    y0_purple = [K_purple, 0, 0, 1]
    affected_purple = [0, 1]
    #eps_p, peaks_p, times_p = sim(y0_purple, t, purple_params, affected_purple, K_purple)
    eps_p, peaks_p, times_p = sim(y0_purple, t, purple_params, affected_purple, K_purple, mode='purple')

# Red: restoration


    # RED: affects k (2) and c (5)
    #K_red = 4502249.959989419 #RED
    K_red = 5972735.535321577
    #red_params = [0.527414353, 1.37599351e-02, 1.00271020, 1.24340295, 6.43668547e-05, 0.750468323]
    red_params = [
    5.59242217e-01,
    9.68398003e-06,
    9.72039172e+01,
    2.82016815e+02,
    1.45086952e+01,
    5.59776275e-01
]
    y0_red = [K_red, 0, 0, 1]
    affected_red = [2, 5]
    #eps_r, peaks_r, times_r = sim(y0_red, t, red_params, affected_red, K_red)
    eps_r, peaks_r, times_r = sim(y0_red, t, red_params, affected_red, K_red, mode='red')



    #Peak Viral Load
    plt.rcParams.update({'font.size': 23})
    plt.figure(figsize=(8,5))
    plt.plot(eps_p, peaks_p, label="ERDRP-0519 (3 dpi)", color="blue")
    plt.plot(eps_r, peaks_r, label="GHP-88309 (3 dpi)", color="red")
    plt.yscale('log')
    plt.xlabel("Drug Efficacy (ε)", fontsize=20)
    plt.ylabel("Peak Viral Load", fontsize=20)
    plt.title("Peak Viral Load vs. Drug Efficacy")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 16}, loc='best')
    plt.tight_layout()
    plt.show()

    #Plot: Time of Viral Peak
    plt.rcParams.update({'font.size': 23})
    plt.figure(figsize=(8,5))
    plt.plot(eps_p, times_p, label="ERDRP-0519 (3 dpi)", color="blue")
    plt.plot(eps_r, times_r, label="GHP-88309 (3 dpi)", color="red")
    plt.yscale('linear')
    plt.xlabel("Drug Efficacy (ε)", fontsize=20)
    plt.ylabel("Time of Viral Peak", fontsize=20)
    plt.title("Time of Viral Peak vs. Drug Efficacy")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend(prop={'size': 16}, loc='best')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
