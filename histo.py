import matplotlib.pyplot as plt
import pandas as pd

# List of CSV file paths (vehicle is now last)
file_paths = [
    "format_purple_Kfixed1_new.csv",
    "format_green_Kfixed.csv",
    "format_yellow_Kfixed2_new.csv",
    "format_red_Kfixed1_new.csv",
    "format_grey_Kfixed1_new.csv"  # Vehicle last
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
parameter_names = datasets[0].columns

# Create 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Map for Greek replacements (customize as needed)
axis_labels = [
    'λ',
    'β',
    'k',
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
