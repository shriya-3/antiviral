from __future__ import print_function, division
import numpy as np
import corner
from matplotlib import pyplot as plt

# Load the saved MCMC samples and true parameter values (result)
samples = np.load('emcee_samples_red.npy')  # Load previously saved samples
#result = np.load('true_values.npy')     # Load true values or initial guesses, if needed

# Define parameter labels (if you have a specific ordering)
labels = ["$lamb$", "$beta$", "$k$", "$delta$", "$p$", "$c$", "$K$"]

#PURPLE
#result = np.log10([1.78E-01, 1.80E-06, 1.68E+02, 2.47E+00, 5.99E-01, 1.61E+00, 1.89E+07]) #PURPLE
#param_range=[(-2.4,0.8),(-9.5,-7),(1.2,2.6),(-0.8,0.8),(-2,1),(-1,0.8),(9.98,10)] #PURPLE

#RED
result = np.log10([1.36703974e-01, 2.20334893e-05, 2.15432783e+00, 2.89684226e+18, 3.12119482e+18, 8.89698760e+01, 8.02762543e+06]) #RED
param_range=[(-3.2,-0.9),(-9,-5),(1.2,3),(16.8,19.2),(12,18.2),(0.8,1.8),(9.6,10)] #RED



# Create the corner plot
fig = corner.corner(samples, labels=labels, truths=result, plot_contours=True,
                    bins=[100, 100, 100, 100, 100, 100, 100], range=param_range, markersize = 100)

# Save the figure
fig.savefig("red_2.png")
# Show the plot
plt.show()

