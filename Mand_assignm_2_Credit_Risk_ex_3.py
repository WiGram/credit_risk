# -*- coding: utf-8 -*-
"""
Created on Wed May  9 07:09:58 2018

@author: William
"""

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.special import factorial

#plt.style.available
plt.style.use("ggplot")


# =========================================================================== #
# ================================ Functions ================================ #
# =========================================================================== #

# Exercise 3 and 4
def binom_fct(n, k, p):
    choose = factorial(n) / (factorial(k)*factorial(n - k))
    return choose * np.power(p,k) * np.power(1-p, n-k)

# The n over k function. Factorial is loaded in the scipy.special package
def mixing_dist_function(correlation, default_probability, theta):
    root      = np.sqrt(1 - correlation * correlation)
    ppf_theta = norm.ppf(theta) # inverse dist functio, ppf: percent point function
    ppf_prob  = norm.ppf(default_probability)
    return norm.cdf(1 / correlation * ( root * ppf_theta - ppf_prob))

def mixing_density(correlation, default_probability, theta):
    root        = np.sqrt(1 - correlation * correlation)
    a           = (root * norm.ppf(theta) - norm.ppf(default_probability)) / correlation
    dens_a      = norm.pdf(a)
    ppf_theta_d = norm.pdf(norm.ppf(theta))
    return dens_a * root / (correlation * ppf_theta_d )

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# Exercise 3

# --------------------------------------------------------------------------- #
# (a) Model parameters
corr = np.array((0.1,0.3,0.5))
d_prob = np.array((0.01,0.05))
theta = np.arange(0.001,0.1,0.001)
matrix = np.zeros((len(theta),len(corr)*len(d_prob)))
matrix2 = np.zeros((len(theta),len(corr)*len(d_prob)))

# ----------------------------- Density function ---------------------------- #
for h in d_prob:
    for i in corr:
        if list(d_prob).index(h) == 0:
            matrix[:,list(corr).index(i)] =\
            mixing_density(i,h, theta)
        else:
            matrix[:,list(corr).index(i)+3] =\
            mixing_density(i,h, theta)
            
# --------------------------- Distribution function ------------------------- #
for h in d_prob:
    for i in corr:
        if list(d_prob).index(h) == 0:
            matrix2[:,list(corr).index(i)] =\
            mixing_dist_function(i,h, theta)
        else:
            matrix2[:,list(corr).index(i)+3] =\
            mixing_dist_function(i,h, theta)

# =============================== Plotting ================================== #
fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
ax1.plot(theta, matrix[:,0], label = 'p = 0.01, rho = 0.1')
ax1.plot(theta, matrix[:,1], label = 'p = 0.01, rho = 0.3')
ax1.plot(theta, matrix[:,2], label = 'p = 0.01, rho = 0.5')
ax1.plot(theta, matrix[:,3], label = 'p = 0.05, rho = 0.1')
ax1.plot(theta, matrix[:,4], label = 'p = 0.05, rho = 0.3')
ax1.plot(theta, matrix[:,5], label = 'p = 0.05, rho = 0.5')
ax1.set_title('Density function')
ax1.legend(loc = 'upper right', shadow = False)
ax1.set_ylabel('Density')
ax1.set_xlabel('Loss fraction')
ax1.set_xlim(xmin = theta[0], xmax = theta[-1])
ax2.plot(theta, matrix2[:,0], label = 'p = 0.01, rho = 0.1')
ax2.plot(theta, matrix2[:,1], label = 'p = 0.01, rho = 0.3')
ax2.plot(theta, matrix2[:,2], label = 'p = 0.01, rho = 0.5')
ax2.plot(theta, matrix2[:,3], label = 'p = 0.05, rho = 0.1')
ax2.plot(theta, matrix2[:,4], label = 'p = 0.05, rho = 0.3')
ax2.plot(theta, matrix2[:,5], label = 'p = 0.05, rho = 0.5')
ax2.set_title('Distribution function')
ax2.legend(loc = 'lower right', shadow = False)
ax2.set_ylabel('Probability')
ax2.set_xlabel('Loss fraction')
ax2.set_xlim(xmin = theta[0], xmax = theta[-1])
fig.tight_layout()


# =========================================================================== #
# =================================== Fin =================================== #
# =========================================================================== #