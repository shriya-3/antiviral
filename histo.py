import matplotlib.pyplot as plt
import pandas as pd

# List of CSV file paths
file_paths = ["format_grey_Kfixed1_new.csv", "format_purple_Kfixed1_new.csv", "format_green_Kfixed.csv", "format_yellow_Kfixed2_new.csv", "format_red_Kfixed1_new.csv"]
datasets = [pd.read_csv(f).iloc[:, :-1] for f in file_paths]  # Read CSV and ignore the last column
dataset_names = ['Vehicle (n=3)', 'ERDRP (3 dpi) (n=3)', 'GHP (5 dpi) (n=3)', 'GHP (7 dpi) (n=3)', 'GHP (3 dpi) (n=3)']

# Extract parameter names (first 6 columns)
parameter_names = datasets[0].columns  
colors = ['black', 'purple', 'green', 'yellow', 'red']  # Unique color for each dataset

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2x3 grid for 6 histograms
axes = axes.flatten()  # Flatten the 2D array for easy iteration

for i, param in enumerate(parameter_names):
    handles = []  # Store legend handles for each parameter's subplot
    labels = []   # Store corresponding labels

    for j, df in enumerate(datasets):
        h = axes[i].hist(df[param], bins=30, alpha=0.5, color=colors[j], label=dataset_names[j])
        handles.append(h[2][0])  # Add the Patch object from hist to handles
        labels.append(dataset_names[j])  # Use manually defined names

    axes[i].set_title(f'Histogram of {param}')
    axes[i].set_xlabel(param)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(handles, labels)  # Set unique legend per histogram

plt.tight_layout()
plt.show()