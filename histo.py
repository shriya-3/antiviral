import matplotlib.pyplot as plt
import pandas as pd

# List of CSV file paths (vehicle is now last)
#file_paths = [
#    "format_purple_Kfixed1_new.csv",
#    "format_green_Kfixed.csv",
#    "format_yellow_Kfixed2_new.csv",
#    "format_red_Kfixed1_new.csv",
#    "format_grey_Kfixed1_new.csv"  # Vehicle last
#]

'''file_paths = [
    "new_format_purple.csv",
    "new_format_green.csv",
    "new_format_yellow.csv",
    "new_format_red.csv",
    "new_format_grey.csv" 
]'''

'''file_paths = [
    "samecons_format_purple.csv",
    "samecons_format_green.csv",
    "samecons_format_yellow.csv",
    "samecons_format_red.csv",
    "samecons_format_grey.csv" 
]'''

'''file_paths = [
    "boot_purple.csv",
    "boot_green.csv",
    "boot_yellow.csv",
    "boot_red.csv",
    "boot_grey.csv" 
]'''

'''file_paths = [
    "ei_format_purple.csv",
    "ei_format_green.csv",
    "samecons_format_yellow.csv",
    "ei_format_red.csv",
    "samecons_format_grey.csv" 
]'''

file_paths = [
    "ei_format_purple.csv",
    "ei_format_green.csv",
    "samecons_format_yellow.csv",
    "yellowguess_format_red.csv",
    "samecons_format_grey.csv" 
]


# Corresponding names (vehicle last)
dataset_names = [
    'ERDRP (3 dpi) (n=3)',
    'GHP (5 dpi) (n=3)',
    'GHP (7 dpi) (n=3)',
    'GHP (3 dpi) (n=3)',
    'Vehicle (n=3)'
]

# Read CSVs (ignore last column)
datasets = [pd.read_csv(f).iloc[:, :-1] for f in file_paths]

# Colors (vehicle = black, plotted last)
colors = ['purple', 'green', 'yellow', 'red', 'teal']

# Extract parameter names (first 6 columns)
parameter_names = datasets[0].columns[:4]
# Create 2x3 subplot
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()

# Map for Greek replacements (customize as needed)
axis_labels = [
    'β',
    'δ',
    'p',
    'c',
]

for i, param in enumerate(parameter_names):
    handles = []
    labels = []

    for j in range(len(datasets) - 1):
        h = axes[i].hist(datasets[j][param], bins=30, alpha=0.5, color=colors[j], label=dataset_names[j])
        handles.append(h[2][0])
        labels.append(dataset_names[j])

    # vehicle last
    j = len(datasets) - 1
    h = axes[i].hist(datasets[j][param], bins=30, alpha=0.5, color=colors[j], label=dataset_names[j])
    handles.append(h[2][0])
    labels.append(dataset_names[j])

    axes[i].set_title("")
    axes[i].set_xlabel(axis_labels[i], fontsize=20)
    axes[i].set_ylabel('Frequency', fontsize=15)

    if i == 0:
        axes[i].legend(handles, labels)


plt.tight_layout()
plt.show()
