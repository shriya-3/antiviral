from __future__ import print_function, division

import numpy as np
import corner
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import emcee

global par
global data


COLOR = "red"


configs = {

"purple": {
    "virus_file": "data/virus_purple.dat",
    "cells_file": "data/cells_purple.dat",
    "K": 4767840.060099503,
    #"initial_guess":[5.34848694e-06, 1.00000000e+02, 1.98769229e+01, 1.35770017e+00],
    "initial_guess":[2.02235251e-06, 3.30084146e+01, 3.05103531e+01, 3.85528847e+00],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"ei_fig_purple.png",
    "csv":"ei_format_purple.csv",
    "title":"ERDRP-0519 (3 dpi)"
},

"green": {
    "virus_file": "data/virus_green.dat",
    "cells_file": "data/cells_green2.dat",
    "K": 3123092.42921152,
    #"initial_guess":[4.06356553e-04,1.00000000e+02,3.10613835e-01,9.93663321e-01],
    "initial_guess":[4.19289797e-04, 1.00000000e+02, 3.02418659e-01, 9.69013533e-01],
    "prior":[-8,-2,1,7,-2,4,-3,2],
    "fig":"ei_fig_green.png",
    "csv":"ei_format_green.csv",
    "title":"GHP-88309 (5 dpi)"
},

"red": {
    "virus_file": "data/virus_red.dat",
    "cells_file": "data/cells_red2.dat",
    "K": 4502249.959989419,
    #"initial_guess":[5.00455586e-05,9.99882682e+01,1.76178104e+01,2.17512702e+01],
    #"initial_guess":[4.99362058e-05, 9.99908520e+01, 1.76249545e+01, 2.16918653e+01],
    "initial_guess":[1.55083578e-06, 9.99908520e+01, 1.76249545e+01, 2.16918653e+01],
    "prior":[-7,-2,-3,3,-3,3,-3,2],
    "fig":"yellowguess_fig_red.png",
    "csv":"yellowguess_format_red.csv",
    "title":"GHP-88309 (3 dpi)"
},

"yellow": {
    "virus_file": "data/virus_yellow.dat",
    "cells_file": "data/cells_yellow2.dat",
    "K": 5142701.720558477,
    #"initial_guess":[1.45794847e-06,9.99997040e+01,5.25328929e+01,8.64861350e-01],
    "initial_guess":[1.55083578e-06, 9.99998252e+01, 5.04847866e+01, 8.35543344e-01],
    "prior":[-8,-2,-3,3,-3,4,-3,2],
    "fig":"samecons_fig_yellow.png",
    "csv":"samecons_format_yellow.csv",
    "title":"GHP-88309 (7 dpi)"
},

"grey": {
    "virus_file": "data/virus_grey.dat",
    "cells_file": "data/cells_grey2.dat",
    "K": 5972735.535321577,
    #"initial_guess":[5.34848694e-06,1.00000000e+02,1.98769229e+01,1.35770017e+00],
    "initial_guess":[5.42211170e-06, 1.00000000e+02, 2.01959362e+01, 1.31955504e+00],
    "prior":[-6,-2,-3,3,-3,4,-3,2],
    "fig":"samecons_fig_grey.png",
    "csv":"samecons_format_grey.csv",
    "title":"Vehicle"
}

}

cfg = configs[COLOR]


###########################################################
# LOAD DATA
###########################################################

virus = np.loadtxt(cfg["virus_file"])
cells = np.loadtxt(cfg["cells_file"])

vdata = np.log10(virus[:,1])
tdata = np.log10(cells[:,1])
t = virus[:,0]

print("data",vdata)

K_fixed = cfg["K"]

lamb = 0.085
k = 3.0


###########################################################
# ODE MODEL
###########################################################

def model(y,t):

    global par

    beta = par[0]
    delta = par[1]
    p = par[2]
    c = par[3]

    K = K_fixed

    lamb = 0.085
    k = 3.0

    T,E,I,V = y

    dTdt = lamb*T*(1-T/K) - beta*T*V
    dEdt = beta*T*V - k*E
    dIdt = k*E - delta*I
    dVdt = p*I - c*V

    return np.array([dTdt,dEdt,dIdt,dVdt])


###########################################################
# SOLVE MODEL
###########################################################

def solveeqs(beta,delta,p,c,t):

    global par

    fpar = np.power(10,[beta,delta,p,c])
    par = list(fpar)

    y=[K_fixed,0.0,0.0,1.0]

    ysol = odeint(model,y,t)

    Tm = ysol[:,0]
    Vm = ysol[:,3]

    return np.array([Tm,Vm])


###########################################################
# INITIAL GUESS
###########################################################

result = np.log10(cfg["initial_guess"])


###########################################################
# LIKELIHOOD
###########################################################

def lnlike(theta,t,tdata,vdata):

    beta,delta,p,c = theta

    model = solveeqs(beta,delta,p,c,t)
    model = np.log10(model)

    x=np.where(np.isnan(model))
    model[x]=0

    return -np.sum((vdata-model[1])**2) - np.sum((tdata-model[0])**2)


###########################################################
# PRIORS
###########################################################

# Define the range for all the parameters
def lnprior(theta):
    beta, delta, p, c = theta
    if -8 < beta < -2  and -3 < delta < 3 and -2 < p < 10 and -3 < c < 3:
        return 0.0
    return -np.inf


###########################################################
# POSTERIOR
###########################################################

def lnprob(theta,t,tdata,vdata):

    lp = lnprior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta,t,tdata,vdata)


###########################################################
# RUN MCMC
###########################################################

print("Start")

ndim,nwalkers = 4,50

pos = [result + 1e-6*np.random.randn(ndim) for i in range(nwalkers)]


stretch_move = emcee.moves.StretchMove(a=2.5)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, moves=stretch_move, args=(t,tdata,vdata))

#sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(t,tdata,vdata))

sampler.run_mcmc(pos,1000)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))


###########################################################
# TRACE PLOT
###########################################################

plt.plot(sampler.chain[:,:,1].T,'-',color='k',alpha=0.3)
plt.show()


###########################################################
# CORNER PLOT
###########################################################

samples = sampler.chain[:,100:,:].reshape((-1,ndim))

print("Done",samples.shape)

samples = samples[0::10]

fig = corner.corner(
    samples,
    labels=["$beta$","$delta$","$p$","$c$"],
    truths=result,
    plot_contours="False",
    bins=[100,100,100,100]
)

fig.suptitle(cfg["title"],fontsize=16)

fig.savefig(cfg["fig"])

plt.show()


###########################################################
# SAVE CSV
###########################################################

K_col = np.full((samples.shape[0],1),K_fixed)
lamb_col = np.full((samples.shape[0],1),lamb)
k_col = np.full((samples.shape[0],1),k)

samples_with_params = np.hstack([samples,lamb_col,k_col,K_col])

np.savetxt(
    cfg["csv"],
    samples_with_params,
    delimiter=',',
    header='beta, delta, p, c, lamb, k, K',
    comments=''
)

print("Data saved to",cfg["csv"])


x=sampler.chain
print("Done",samples.shape,x.shape)