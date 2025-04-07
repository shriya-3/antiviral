import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def main():
    ctrl_df = pd.read_csv('format_grey_Kfixed1_new.csv')

    antiviral = np.array(['format_green_Kfixed.csv', 'format_purple_Kfixed1_new.csv', 'format_red_Kfixed1_new.csv', 'format_yellow_Kfixed2_new.csv'])
    names = np.array(["green", "purple", "red", "yellow"])
    params = np.array(["lamb", "beta", "k", "delta", "p", "c"])
    for i in range(len(antiviral)):
        antiviral_df = pd.read_csv(antiviral[i])
        for j in range(6):
            #antiviral_param = antiviral_df[params[j]].iloc[:1000].to_numpy()
            #ctrl_param = ctrl_df[params[j]].iloc[:1000].to_numpy()
            antiviral_param = antiviral_df.iloc[:1000, j].to_numpy()
            ctrl_param = ctrl_df.iloc[:1000, j].to_numpy()
            U, p = mannwhitneyu(ctrl_param, antiviral_param)
            print(f'{names[i]}, {params[j]}, U-value: {U}')
            print(f'{names[i]}, {params[j]}, p-value: {p}')



if __name__ == "__main__":
    main()
