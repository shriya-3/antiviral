import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def main():
    ctrl_df = pd.read_csv('format_grey_Kfixed1_new.csv')

    antiviral_files = ['format_green_Kfixed.csv', 'format_purple_Kfixed1_new.csv',
                       'format_red_Kfixed1_new.csv', 'format_yellow_Kfixed2_new.csv']
    names = ["green", "purple", "red", "yellow"]
    params = ["lamb", "beta", "k", "delta", "p", "c"]

    for i in range(len(antiviral_files)):
        antiviral_df = pd.read_csv(antiviral_files[i])
        print(f"Comparing control vs {names[i]}:")

        for j in range(len(params)):
            U_array = []
            p_array = []

            ctrl_param = ctrl_df[params[j]].iloc[:1000].to_numpy()
            antiviral_param = antiviral_df[params[j]].iloc[:1000].to_numpy()

            for _ in range(10):
                random_ctrl = np.random.choice(ctrl_param, size=10, replace=False)
                random_antiviral = np.random.choice(antiviral_param, size=10, replace=False)
                U, p = mannwhitneyu(random_antiviral, random_ctrl)
                U_array.append(U)
                p_array.append(p)

            avg_U = np.mean(U_array)
            avg_p = np.mean(p_array)

            print(f"  Parameter: {params[j]} | Avg U: {avg_U:.2f} | Avg p: {avg_p:.4f}")
if __name__ == "__main__":
    main()