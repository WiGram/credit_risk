# -*- coding: utf-8 -*-
"""
Created on Wed May  9 07:09:58 2018

@author: William
"""

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.special import factorial
from numba import jit

#plt.style.available
plt.style.use("ggplot")


# Note that numba does not work with scipy distribution functions for the time
# being, meaning the time increase from using jit (just in time: numba) is at
# most a factor 1.25.

# =========================================================================== #
# ================================ Functions ================================ #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Assisting functions for the big boy below
# --------------------------------------------------------------------------- #

def binom_fct(n, k, p):
    choose = factorial(n) / (factorial(k)*factorial(n - k))
    return choose * np.power(p,k) * np.power(1-p, n-k)

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
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The big boy below: Calculates payoffs and spreads for each tranche
# --------------------------------------------------------------------------- #
    
@jit
def payoff_matrix(n, d_prob, corr, max_theta, detachment, recovery, face_value_percentage, small_or_large_portfolio):
    k = np.arange(0, n+1, 1)
    k_payoff = np.zeros((len(k)))
    theta = np.arange(0.00001,max_theta, 0.00001)
    p_tilde = np.zeros((len(k), len(d_prob)*len(corr)))
    tranche_payoff = np.zeros((len(detachment), len(d_prob)*len(corr)))
    loss_fraction = recovery * face_value_percentage
    if small_or_large_portfolio == 'small':
        for m in detachment:
            mx = list(detachment).index(m)
            if mx == 0:
                for p in d_prob:
                    px = list(d_prob).index(p)
                    if px == 0: 
                        for i in corr:
                            ix = list(corr).index(i)
                            for j in k:
                                jx = list(k).index(j)
                                p_tilde[jx,ix] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix],d_prob[px],theta))/len(theta)
                                k_payoff[jx] = max(m - loss_fraction*j, 0) * p_tilde[jx,ix]
                            tranche_payoff[mx,ix] = sum(k_payoff)
                    else:
                        for i in corr:
                            ix = list(corr).index(i)
                            for j in k:
                                jx = list(k).index(j)
                                p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                                k_payoff[jx] = max(m - loss_fraction*j,0) * p_tilde[jx, ix + 2]
                            tranche_payoff[mx,ix + 2] = sum(k_payoff)
            else:
                last_detachment = detachment[mx - 1]
                for p in d_prob:
                    px = list(d_prob).index(p)
                    if px == 0: 
                        for i in corr:
                            ix = list(corr).index(i)
                            for j in k:
                                jx = list(k).index(j)
                                p_tilde[jx,ix] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix],d_prob[px],theta))/len(theta)
                                k_payoff[jx] = (max(m - loss_fraction*j, 0) - max(last_detachment - loss_fraction*j, 0)) * p_tilde[jx,ix]
                            tranche_payoff[mx,ix] = sum(k_payoff)
                    else:
                        for i in corr:
                            ix = list(corr).index(i)
                            for j in k:
                                jx = list(k).index(j)
                                p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                                k_payoff[jx] = (max(m - loss_fraction*j, 0) - max(last_detachment - loss_fraction*j, 0)) * p_tilde[jx, ix + 2]
                            tranche_payoff[mx,ix + 2] = sum(k_payoff)
    else:
        for m in detachment:
            mx = list(detachment).index(m)
            if mx == 0:
                for p in d_prob:
                    px = list(d_prob).index(p)
                    if px == 0:
                        for i in corr:
                            ix = list(corr).index(i)
                            tranche_payoff[mx,ix] = sum(np.maximum(m - 100 * theta,0)*mixing_density(i, p, theta))/len(theta)
                    else:
                        for i in corr:
                            ix = list(corr).index(i)
                            tranche_payoff[mx,ix + 2] = sum(np.maximum(m - 100 * theta, 0)*mixing_density(i,p,theta))/len(theta)
            else:
                last_detachment = detachment[mx - 1]
                for p in d_prob:
                    px = list(d_prob).index(p)
                    if px == 0:
                        for i in corr:
                            ix = list(corr).index(i)
                            tranche_payoff[mx,ix] = sum((np.maximum(m - 100 * theta,0) - np.maximum(last_detachment - 100 * theta, 0))*mixing_density(i, p, theta))/len(theta)
                    else:
                        for i in corr:
                            ix = list(corr).index(i)
                            tranche_payoff[mx,ix + 2] = sum((np.maximum(m - 100 * theta,0) - np.maximum(last_detachment - 100 * theta, 0))*mixing_density(i,p,theta))/len(theta)

    spread = np.zeros((len(detachment), len(d_prob)*len(corr)))
    k = 0

    for m in detachment:
        mx = list(detachment).index(m)
        diff = m - k
        spread[mx] = diff / tranche_payoff[mx] - 1
        k = m

    return tranche_payoff, spread
# --------------------------------------------------------------------------- #
    

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Initial model parameters
# --------------------------------------------------------------------------- #
    
n = 125
d_prob = np.array((0.01,0.05))
corr = np.array((0.3,0.5))
max_theta = 1
recovery = 1
face_value_percentage = 0.8
detachment = np.array((3,6,9,12,22,100))

# --------------------------------------------------------------------------- #


# =========================================================================== #
# =============================== Estimation ================================ #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Exercise 4: Small portfolio
# --------------------------------------------------------------------------- #

matrix_4 = payoff_matrix(n, d_prob, corr, max_theta, detachment, recovery, face_value_percentage, 'small')

tranche_payoffs_4 = matrix_4[0]
spreads_4 = matrix_4[1]

np.savetxt("tranche_payoff_4.csv", tranche_payoffs_4)
np.savetxt("credit_spread_4.csv", spreads_4)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Exercise 5: Large portfolio approximation (hence no 'small' in the function)
# --------------------------------------------------------------------------- #

matrix_5 = payoff_matrix(n, d_prob, corr, max_theta, detachment, recovery, face_value_percentage, 'large')

tranche_payoffs_5 = matrix_5[0]
spreads_5 = matrix_5[1]

np.savetxt("tranche_payoff_5.csv", tranche_payoffs_5)
np.savetxt("credit_spread_5.csv", spreads_5)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Exercise 6: Small portfolio, recovery != 0.
# --------------------------------------------------------------------------- #

# Adjustment to recovery rate abd default probability: 
recovery = 0.5
d_prob2 = np.array((0.02, 0.10))

matrix_6 = payoff_matrix(n, d_prob2, corr, max_theta, detachment, recovery, face_value_percentage, 'small')

tranche_payoffs_6 = matrix_6[0]
spreads_6 = matrix_6[1]
   
np.savetxt("tranche_payoff_6.csv", tranche_payoffs_6)
np.savetxt("credit_spread_6.csv", spreads_6)
# --------------------------------------------------------------------------- #


# =========================================================================== #
# ================================== Plots ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Exercise 4: spreads
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spreads_4[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spreads_4[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spreads_4[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spreads_4[:,3], bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='P = 0.05')

plt.xlabel('Tranche')
plt.ylabel('Credit spread')
plt.title('Credit spread by tranche and default probability')
plt.xticks(index + bar_width / 2, ('Equity', 'Junior Mezzanine', 'Senior Mezzanine', 'Super Senior Mezzanine', 'Super Secure'))
plt.legend()

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Exercise 5: Spreads
# --------------------------------------------------------------------------- #

fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spreads_5[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spreads_5[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spreads_5[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spreads_5[:,3], bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='P = 0.05')

plt.xlabel('Tranche')
plt.ylabel('Credit spread')
plt.title('Credit spread by tranche and default probability')
plt.xticks(index + bar_width / 2, ('Equity', 'Junior Mezzanine', 'Senior Mezzanine', 'Super Senior Mezzanine', 'Super Secure'))
plt.legend()

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Exercise 6: Spreads
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spreads_6[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spreads_6[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spreads_6[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spreads_6[:,3], bar_width,
                 alpha=opacity,
                 color='y',
                 error_kw=error_config,
                 label='P = 0.05')

plt.xlabel('Tranche')
plt.ylabel('Credit spread')
plt.title('Credit spread by tranche and default probability')
plt.xticks(index + bar_width / 2, ('Equity', 'Junior Mezzanine', 'Senior Mezzanine', 'Super Senior Mezzanine', 'Super Secure'))
plt.legend()

plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------- #

# =========================================================================== #
# =================================== Fin =================================== #
# =========================================================================== #