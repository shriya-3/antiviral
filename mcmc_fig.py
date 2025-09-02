import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt

df = pd.read_csv('format_purple_Kfixed1_new.csv')
df.columns = df.columns.str.strip()

if "K" in df.columns:
    df = df.drop(columns=["K"])

data = df.values 
print(df.shape)  # should be (4000+, 6)


fig = corner.corner(data, 
                    labels=["$λ$", "$β$", "$k$", "$δ$", "$p$", "$c$"],
                    plot_contours=True,
                    bins=[100]*6)

fig.suptitle("ERDRP-0519 (3 dpi)", fontsize=16)
#plt.tight_layout(pad=2.0)
fig.savefig("erdrp1.png", dpi=300)
plt.show()
