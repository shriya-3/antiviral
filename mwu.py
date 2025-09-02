import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def main():
    #antiviral_param = antiviral_df[params[j]].iloc[:1000].to_numpy()
    #ctrl_param = ctrl_df[params[j]].iloc[:1000].to_numpy()
    ctrl_df = pd.read_csv('format_grey_Kfixed1_new.csv')

    antiviral = np.array(['format_green_Kfixed.csv', 'format_purple_Kfixed1_new.csv', 'format_red_Kfixed1_new.csv', 'format_yellow_Kfixed2_new.csv'])
    names = np.array(["green", "purple", "red", "yellow"])
    params = np.array(["lamb", "beta", "k", "delta", "p", "c"])
    for i in range(len(antiviral)):
        antiviral_df = pd.read_csv(antiviral[i])
        for j in range(6):

            U_array = []
            p_array = []
            for k in range(10):
                antiviral_param = antiviral_df.iloc[:1000, j].to_numpy()
                ctrl_param = ctrl_df.iloc[:1000, j].to_numpy()
                random_antiviral = np.random.choice(antiviral_param, size=10, replace=False)
                random_control = np.random.choice(ctrl_param, size=10, replace=False)
                U, p = mannwhitneyu(random_antiviral, random_control)
                U_array.append(U)
                p_array.append(p)
            U_avg = sum(U_array) / len(U_array)
            p_avg = sum(p_array) / len(p_array)
            print(f'{names[i]}, {params[j]}, U-value: {U_avg}')
            print(f'{names[i]}, {params[j]}, p-value: {p_avg}')



if __name__ == "__main__":
    main()
