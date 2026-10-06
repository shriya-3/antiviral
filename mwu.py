import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

def main():

    #ctrl_df = pd.read_csv('new_format_grey.csv')

    #ctrl_df = pd.read_csv('samecons_format_grey.csv')

    #ctrl_df = pd.read_csv('boot_grey.csv')
    ctrl_df = pd.read_csv('samecons_format_grey.csv')
    '''antiviral_files = [
        'new_format_green.csv',
        'new_format_purple.csv',
        'new_format_red.csv',
        'new_format_yellow.csv'
    ]'''

    '''antiviral_files = [
        'samecons_format_green.csv',
        'samecons_format_purple.csv',
        'samecons_format_red.csv',
        'samecons_format_yellow.csv'
    ]'''

    '''
    antiviral_files = [
        'boot_green.csv',
        'boot_purple.csv',
        'boot_red.csv',
        'boot_yellow.csv'
    ]'''

    antiviral_files = [
        'ei_format_green.csv',
        'ei_format_purple.csv',
        'ei_format_red.csv',
        'samecons_format_yellow.csv'
    ]


    names = ["green", "purple", "red", "yellow"]
    params = ["beta", "delta", "p", "c"]

    for i, file in enumerate(antiviral_files):

        antiviral_df = pd.read_csv(file)

        for j in range(4):  # each parameter

            antiviral_param = antiviral_df.iloc[:1000, j].to_numpy()
            ctrl_param = ctrl_df.iloc[:1000, j].to_numpy()

            p_values = []
            U_values = []

            for run in range(100):  # 100 bootstrap comparisons

                sample_treat = np.random.choice(antiviral_param, size=10, replace=False)
                sample_ctrl = np.random.choice(ctrl_param, size=10, replace=False)

                U, p = mannwhitneyu(sample_treat, sample_ctrl, alternative='two-sided')

                U_values.append(U)
                p_values.append(p)

            # summarize distribution of bootstrap tests
            print(f"{names[i]}, {params[j]}")

            print(f"  U mean:    {np.mean(U_values)}")

            print(f"  p mean:    {np.mean(p_values)}")
            print(f"  p < 0.05 fraction: {np.mean(np.array(p_values) < 0.05):.2f}")
            print()

if __name__ == "__main__":
    main()