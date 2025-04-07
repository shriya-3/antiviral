from __future__ import print_function, division
#from IPython.display import display, Math

#import os
#import sys
import numpy as np
import corner
from scipy.integrate import odeint
from matplotlib import pyplot as plt

import emcee
global par
global data

# Load in data
#virus = np.loadtxt('data/virus_purple.dat')
#cells = np.loadtxt('data/cells_purple.dat')

#virus = np.loadtxt('data/virus_green.dat')
#cells = np.loadtxt('data/cells_green2.dat')

#virus = np.loadtxt('data/virus_red.dat')
#cells = np.loadtxt('data/cells_red2.dat')

#virus = np.loadtxt('data/virus_yellow.dat')
#cells = np.loadtxt('data/cells_yellow2.dat')

virus = np.loadtxt('data/virus_grey.dat')
cells = np.loadtxt('data/cells_grey2.dat')

vdata = np.log10(virus[:, 1])
tdata = np.log10(cells[:, 1])
print(f"data",vdata)
t = virus[:, 0]
#K_fixed = 74767840.060099503
#K_fixed = 4767840.060099503 #PURPLE
#K_fixed = 3123092.42921152 #GREEN
#K_fixed = 4502249.959989419 #RED
#K_fixed = 5142701.720558477 #YELLOW
K_fixed = 5972735.535321577 #GREY

#K_fixed = 2.46272672e+07

# ODE function describing the virus and cell equations
def model(y, t):
    global par
    lamb = par[0]
    beta = par[1]
    k = par[2]
    delta = par[3]
    p = par[4]
    c = par[5]
    #K = 4767840.060099503 #PURPLE
    #K = 3123092.42921152 #GREEN
    #K = 4502249.959989419 #RED
    #K = 5142701.720558477 #YELLOW
    K = 5972735.535321577 #GREY


    T, E, I, V = y
    #lamb, beta, k, delta, p, c, K = params

    dTdt = lamb * T * (1-T/K) - beta * T * V
    dEdt = beta * T * V - k * E
    dIdt = k * E - delta * I
    dVdt = p * I - c * V

    return np.array([dTdt, dEdt, dIdt, dVdt])

# Integrate the ODE function and return the virus
def solveeqs(lamb, beta, k, delta, p, c, t):#pp):
    global par
    global data

    fpar=np.power(10,[lamb, beta, k, delta, p, c]) 
    par = list(fpar) 
    #par.extend(fpar[:])
    #upd = []
  
    #y=[fpar[-1], 0.0, 0.0, fpar[-2]] 
    y = [K_fixed, 0.0, 0.0, fpar[-2]] #should this be fpar[-1] or the K value
    ysol = odeint(model, y, t)
    
    #findindex=lambda x:np.where(t==x)[0][0]
    #mindex=list(map(findindex,t))
    #Vmodel=ysol[1:,-1]
    #Tmodel=ysol[1:,-4]
    Tm=ysol[:,0]
    Vm=ysol[:,3]
    #Vm=Vmodel[mindex]

    return np.array([Tm,Vm])

# This is the initial guess --- you can change this (b,p,c,d,k,v0)
#result= np.log10([1.35653818e-02, 2.67324539e-06, 4.53607591e+00, 4.26398866e+00, 2.45679026e+00, 1.30694794e+00, 4.37516494e+00, 1.01332200e+13]) 
#result = np.log10([7.17316553e-01, 3.36823860e-04, 7.83678424e-01, 2.00291307e+00, 7.61300436e-03, 8.72944403e-01, 4.52239206e+06])
#result = np.log10([1.02605046e-16, 1.20989507e-06, 8.49060327e+10, 2.99309184e+01, 1.07597258e+01, 2.60597208e+00, 2.09812514e+01])

#result = np.log10([1.78E-01, 1.80E-06, 1.68E+02, 2.47E+00, 5.99E-01, 1.61E+00, 1.89E+07]) #PURPLE
#result = np.log10([1.36703974e-01, 2.20334893e-05, 2.15432783e+00, 2.89684226e+18, 3.12119482e+18, 8.89698760e+01, 8.02762543e+06]) #RED
#result = np.log10([1.33811643e-25, 1.51975522e-06, 2.38853749e+02, 1.65858268e+02, 4.75308073e+01, 2.77989318e-01, 2.87466633e-18]) #YELLOW
#result = np.log10([7.45030763e-05, 6.67527894e-78, 1.07337535e+00, 5.94586265e+04, 8.61326107e+77, 6.52842748e+01, 2.67172008e+02]) #GREEN


#After re-running initial guesses

#PURPLE DATASET
#result = np.log10([3.34131606e-03, 7.97858286e-07, 1.61285348e+02, 5.17873197e+00, 6.14732745e+01, 2.75111791e+01, 3.44613946e+05]) #PURPLE
#result = np.log10([7.24985555e-03, 2.50715804e-08, 2.61706640e+01, 5.24540588e+01, 8.67834551e+03, 1.44740750e+01, 7.64463148e+05]) #PURPLE NEW
#result = np.log10([5.20019145e-04, 6.21685867e-07, 2.22759472e+02, 1.44617993e+02, 3.03191439e+02, 4.67172092e+00, 4767840.060099503]) #PURPLE K(INITIAL CELLS)
#result = np.log10([5.76489543e-01, 1.00211170e-04, 1.59401898e+02, 2.44051237e+02, 1.70054323e+00, 1.70663815e+00])
#result = np.log10([9.58646156e-02, 2.15340282e-06, 9.65755873e+01, 3.18465532e+01, 9.67901147e+00, 1.31840638e+00]) #MOST RECENT

#GREEN DATASET
#result = np.log10([2.29008838e-01, 3.33848192e-05, 5.48977445e+00, 2.75132844e+02, 1.74369880e+01, 1.67546080e+00])

#RED DATASET
#result = np.log10([1.15360747e-01, 1.04649218e-05, 3.32467872e+00, 2.30330430e+17, 4.07116880e+17, 4.22678961e+01]) #most receent?
#result = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01])
#result = np.log10([9.40540949e-02, 1.71571339e-05, 2.86281663e+03, 3.01583306e+01, 2.61835030e+00, 4.06336670e+00]) #MOST RECENT

#YELLOW DATASET
#result = np.log10([1.48251874e-01, 1.84039128e-06, 3.42223960e+07, 1.26325461e+08, 3.18247380e+07, 5.77409907e-01])
#result = np.log10([4.68045301e-01, 2.45361792e-06, 9.57476446e+01, 2.22159842e+01, 4.59566061e+00, 5.37628973e-01]) #MOST RECENT

#GREY DATASET
result = np.log10([5.65271052e-01, 9.70303786e-06, 9.72892766e+01, 2.82402642e+02, 1.45084101e+01, 5.60841578e-01]) #GREY


#result = np.log10([2.15090516e-01, 3.45657502e-05, 1.52896846e+00, 1.69064399e+07, 6.14777220e+06, 9.91326364e+00, 2.99780262e+06]) #GREEN
#result = np.log10([1.30662809e-01, 1.60994300e-05, 1.99801204e+01, 3.95619435e+00, 1.63277801e+00, 2.67318059e+01, 5.40522939e+06]) #RED
#result = np.log10([5.27414353e-01, 1.37599351e-02, 1.00271020e+00, 1.24340295e+00, 6.43668547e-05, 7.50468323e-01, 4.96716462e+06]) #YELLOW

#lamb, beta, k, delta, p, c, K


# Define the SSR
def lnlike(theta, t, tdata, vdata):
    lamb, beta, k, delta, p, c= theta
    model = solveeqs(lamb, beta, k, delta, p, c, t)
    model = np.log10(model)
    print(model[0])
    x = np.where(np.isnan(model))
    model[x] = 0
    return -np.sum((vdata-model[1])**2) - np.sum((tdata-model[0])**2)

# Define the range for all the parameters
def lnprior(theta):
    lamb, beta, k, delta, p, c = theta
    #if -4 < lamb < 0 and -8 < beta < -2  and -3 < k < 3 and -3 < delta < 3 and -3 < p < 4 and -3 < c < 2: #PURPLE (MOST RECENT)
    #if -3 < lamb < 1 and -5 < beta < -1  and -3 < k < 3 and -3 < delta < 3 and -6 < p < 2 and -3 < c < 2: #RED
    #if -3 < lamb < 1 and -7 < beta < -2  and -3 < k < 4 and -3 < delta < 3 and -3 < p < 3 and -3 < c < 2: #RED (Most recent?)
    #if -3 < lamb < 1 and -8 < beta < -2  and -3 < k < 3 and -3 < delta < 3 and -3 < p < 4 and -3 < c < 2: #yellow (Most recent)
    #if -6 < lamb < 1 and -8 < beta < -2  and -3 < k < 3 and 1 < delta < 7 and -2 < p < 4 and -3 < c < 2: #GREEN (MOST RECENT)
    if -3 < lamb < 1 and -6 < beta < -2  and -3 < k < 3 and -3 < delta < 3 and -3 < p < 4 and -3 < c < 2: #GREY (MOST RECENT)

    #if -7 < lamb < -1 and -80 < beta < -74  and -3 < k < 3 and -1 < delta < 6 and 71 < p < 79 and -3 < c < 2 and -3 < K < 3:
        return 0.0
    return -np.inf

# Defines the new parameters
def lnprob(theta, t, tdata, vdata):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, t, tdata, vdata)


print('Start')
# You can change the number of walkers (second number)
ndim, nwalkers = 6, 50
# Sets up all the walkers in a ball around the initial guess
pos = [(result) + 1e-6*np.random.randn(ndim) for i in range(nwalkers)]

# Runs the Markov chain Monte Carlo sampling
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, tdata, vdata))
sampler.run_mcmc(pos, 1000); # The second argument is the number of steps taken, you can change this

res=plt.plot(sampler.chain[:,:,1].T, '-', color='k', alpha=0.3)
plt.show()

# Creates the corner plot
samples =sampler.chain[:, 100:, :].reshape((-1, ndim))
print('Done', samples.shape)
samples=samples[0::10]
print(samples.shape)  

fig = corner.corner(samples, labels=["$lamb$", "$beta$", "$k$","$delta$", "$p$", "$c$"],
                      truths=result, plot_contours="False", bins=[100,100,100,100,100,100])
#fig.savefig("purple_Kfixed7.png")
#fig.savefig("green_Kfixed.png")
#fig.savefig("red_Kfixed_new1.png")
#fig.savefig("yellow_Kfixed_new2.png")
fig.savefig("grey_Kfixed1.png")


plt.show()


print('Done')


K_fixed_column = np.full((samples.shape[0], 1), np.log10(K_fixed))  # Create a column of log10(K_fixed)
#samples = np.hstack((samples, K_fixed_column))  # Append to samples
#print("Shape of modified samples:", samples.shape)  # Should be (num_samples, 7)
#np.save('emcee_samples_purple_Kfixed.npy', samples)
'''samples_with_K = np.hstack([samples, np.full((samples.shape[0], 1), K_fixed)]) 
np.save('emcee_samples_purple_Kfixed1.npy', samples_with_K)
# Save the results
fd = open('emcee_control_purple_Kfixed1.dat', 'w')
fd.write(str(np.power(10, samples_with_K).tolist()))  # Convert back from log scale before writing
fd.close()'''

# Ensure `samples_with_K` is properly formatted
samples_with_K = np.hstack([samples, np.full((samples.shape[0], 1), K_fixed)])

# Save as CSV directly
#csv_filename = 'format_purple_Kfixed7.csv'
#csv_filename = 'format_green_Kfixed.csv'
#csv_filename = 'format_red_Kfixed1 .csv'
#csv_filename = 'format_purple_Kfixed1_new.csv'
#csv_filename = 'format_red_Kfixed1_new.csv'
#csv_filename = 'format_yellow_Kfixed2_new.csv'
csv_filename = 'format_grey_Kfixed1_new.csv'

np.savetxt(csv_filename, samples_with_K, delimiter=',',
           header='lamb, beta, k, delta, p, c, K', comments='')

print(f"Data saved to {csv_filename}")



x=sampler.chain
print('Done', samples.shape, x.shape)


'''np.save('emcee_samples_purple_Kfixed.npy', samples)
# Save the results

fd=open('emcee_control_purple_Kfixed.dat','w')
fd.write(str(np.power(10,samples).tolist()))#,str(samples))
fd.close()'''