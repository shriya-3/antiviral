import numpy as np
import corner
from scipy.integrate import odeint
from matplotlib import pyplot as plt

import emcee
global par
global data

virus = np.loadtxt('data/virus_grey.dat')
cells = np.loadtxt('data/cells_grey2.dat')

vdata = np.log10(virus[:, 1])
tdata = np.log10(cells[:, 1])
print(f"data",vdata)
t = virus[:, 0]
K_fixed = 5972735.535321577 #GREY

# ODE function describing the virus and cell equations
def model(y, t):
    global par
    lamb = par[0]
    beta = par[1]
    k = par[2]
    delta = par[3]
    p = par[4]
    c = par[5]

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

  
    y = [K_fixed, 0.0, 0.0, fpar[-2]] 
    ysol = odeint(model, y, t)
    

    Tm=ysol[:,0]
    Vm=ysol[:,3]

    return np.array([Tm,Vm])


result = np.log10([5.65271052e-01, 9.70303786e-06, 9.72892766e+01, 2.82402642e+02, 1.45084101e+01, 5.60841578e-01]) #GREY



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
    if -3 < lamb < 1 and -6 < beta < -2  and -3 < k < 3 and -3 < delta < 3 and -3 < p < 4 and -3 < c < 2: #GREY (MOST RECENT)
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

fig.savefig("grey_Kfixed1.png")


plt.show()


print('Done')


K_fixed_column = np.full((samples.shape[0], 1), np.log10(K_fixed))  # Create a column of log10(K_fixed)

samples_with_K = np.hstack([samples, np.full((samples.shape[0], 1), K_fixed)])


csv_filename = 'format_grey_Kfixed1_new.csv'

np.savetxt(csv_filename, samples_with_K, delimiter=',',
           header='lamb, beta, k, delta, p, c, K', comments='')

print(f"Data saved to {csv_filename}")



x=sampler.chain
print('Done', samples.shape, x.shape)

