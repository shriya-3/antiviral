import pandas as pd
import numpy as np

df = pd.read_csv('format_grey_Kfixed1_new.csv')  

ci_95 = df.quantile([0.025, 0.975])

for param in df.columns:
    lower_log = ci_95.loc[0.025, param]
    upper_log = ci_95.loc[0.975, param]
    
    lower_lin = 10 ** lower_log
    upper_lin = 10 ** upper_log
    
    print(f"{param}: [{lower_lin:.4e}, {upper_lin:.4e}]")
