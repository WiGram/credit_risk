# -*- coding: utf-8 -*-
"""
Created on Thu May  3 20:44:15 2018

@author: William
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.append('C:/Users/William/.spyder-py3/')
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
# ================================== Model ================================== #
# =========================================================================== #

# Exercise 4

# --------------------------------------------------------------------------- #
# (b) Further model parameters
n = 125
k = np.arange(0,n+1,1)
detachment = np.array((3,6,9,12,22,100))

# (a) Model parameters
corr = np.array((0.3,0.5))
d_prob = np.array((0.01,0.05))
theta = np.arange(0.05,1,0.05)

theta = np.arange(0.00001,1,0.00001)
p_tilde = np.zeros((len(k),len(d_prob)*len(corr)))
k_payoff = np.zeros((len(k)))
tranche_payoff = np.zeros((len(detachment),len(d_prob)*len(corr)))

detachment
detachment[0]
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
                        k_payoff[jx] = max(m - 0.8*j, 0) * p_tilde[jx,ix]
                    tranche_payoff[mx,ix] = sum(k_payoff)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                        k_payoff[jx] = max(m - 0.8*j,0) * p_tilde[jx, ix + 2]
                    tranche_payoff[mx,ix + 2] = sum(k_payoff)
    else:
        for p in d_prob:
            px = list(d_prob).index(p)
            if px == 0: 
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        last_detachment = detachment[mx - 1]
                        p_tilde[jx,ix] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix],d_prob[px],theta))/len(theta)
                        k_payoff[jx] = (max(m - 0.8*j, 0) - max(last_detachment - 0.8*j, 0)) * p_tilde[jx,ix]
                    tranche_payoff[mx,ix] = sum(k_payoff)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                        k_payoff[jx] = (max(m - 0.8*j, 0) - max(last_detachment - 0.8*j, 0)) * p_tilde[jx, ix + 2]
                    tranche_payoff[mx,ix + 2] = sum(k_payoff)
                    
np.savetxt("tranche_payoffs.csv", tranche_payoff)

spread_4 = np.zeros((len(detachment), len(d_prob)*len(corr)))
k = 0

for m in detachment:
    mx = list(detachment).index(m)
    diff = m - k
    spread_4[mx] = diff / tranche_payoff[mx] - 1
    k = m
   
np.savetxt("credit_spread_4.csv", spread_4)

fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spread_4[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spread_4[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spread_4[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spread_4[:,3], bar_width,
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

# Exercise 5

# --------------------------------------------------------------------------- #
# (a) remaining model parameters
theta = np.arange(0.00001,1,0.00001)               
place_holder = np.zeros((len(theta))) 
payoff = np.zeros((len(detachment),len(d_prob)*len(corr)))                   

for m in detachment:
    mx = list(detachment).index(m)
    if mx == 0:
        for p in d_prob:
            px = list(d_prob).index(p)
            if px == 0:
                for i in corr:
                    ix = list(corr).index(i)
                    payoff[mx,ix] = sum(np.maximum(m - 100 * theta,0)*mixing_density(i, p, theta))/len(theta)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    payoff[mx,ix + 2] = sum(np.maximum(m - 100 * theta, 0)*mixing_density(i,p,theta))/len(theta)
    else:
        last_detachment = detachment[mx - 1]
        for p in d_prob:
            px = list(d_prob).index(p)
            if px == 0:
                for i in corr:
                    ix = list(corr).index(i)
                    payoff[mx,ix] = sum((np.maximum(m - 100 * theta,0) - np.maximum(last_detachment - 100 * theta, 0))*mixing_density(i, p, theta))/len(theta)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    payoff[mx,ix + 2] = sum((np.maximum(m - 100 * theta,0) - np.maximum(last_detachment - 100 * theta, 0))*mixing_density(i,p,theta))/len(theta)

np.savetxt("tranche_payoff_5.csv", payoff)

spread_5 = np.zeros((len(detachment), len(d_prob)*len(corr)))
k = 0

for m in detachment:
    mx = list(detachment).index(m)
    diff = m - k
    spread_5[mx] = diff / payoff[mx] - 1
    k = m
   
np.savetxt("credit_spread_5.csv", spread_5)

fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spread_5[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spread_5[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spread_5[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spread_5[:,3], bar_width,
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

# Exercise 6

# --------------------------------------------------------------------------- #
# (a) remaining model parameters
recovery = 0.5
loss     = 0.8
# --------------------------------------------------------------------------- #

# (b) Further model parameters
n = 125
k = np.arange(0,n+1,1)
detachment = np.array((3,6,9,12,22,100))

# (a) Model parameters
corr = np.array((0.3,0.5))
d_prob = np.array((0.01,0.05))
theta = np.arange(0.05,1,0.05)

theta = np.arange(0.00001,1,0.00001)
p_tilde = np.zeros((len(k),len(d_prob)*len(corr)))
k_payoff = np.zeros((len(k)))
tranche_payoff2 = np.zeros((len(detachment),len(d_prob)*len(corr)))

detachment
detachment[0]
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
                        k_payoff[jx] = max(m - loss*recovery*j, 0) * p_tilde[jx,ix]
                    tranche_payoff2[mx,ix] = sum(k_payoff)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                        k_payoff[jx] = max(m - loss*recovery*j,0) * p_tilde[jx, ix + 2]
                    tranche_payoff2[mx,ix + 2] = sum(k_payoff)
    else:
        for p in d_prob:
            px = list(d_prob).index(p)
            if px == 0: 
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        last_detachment = detachment[mx - 1]
                        p_tilde[jx,ix] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix],d_prob[px],theta))/len(theta)
                        k_payoff[jx] = (max(m - loss*recovery*j, 0) - max(last_detachment - loss*recovery*j, 0)) * p_tilde[jx,ix]
                    tranche_payoff2[mx,ix] = sum(k_payoff)
            else:
                for i in corr:
                    ix = list(corr).index(i)
                    for j in k:
                        jx = list(k).index(j)
                        p_tilde[jx,ix + 2] = sum(binom_fct(n,j,theta) * mixing_density(corr[ix], d_prob[px], theta)) / len(theta)
                        k_payoff[jx] = (max(m - loss*recovery*j, 0) - max(last_detachment - loss*recovery*j, 0)) * p_tilde[jx, ix + 2]
                    tranche_payoff2[mx,ix + 2] = sum(k_payoff)
                    
np.savetxt("tranche_payoff_6.csv", tranche_payoff2)

spread_6 = np.zeros((len(detachment), len(d_prob)*len(corr)))
k = 0

for m in detachment:
    mx = list(detachment).index(m)
    diff = m - k
    spread_6[mx] = diff / tranche_payoff[mx] - 1
    k = m
   
np.savetxt("credit_spread_5.csv", spread_6)

fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spread_6[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spread_6[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

rects1 = plt.bar(index + 2*bar_width, spread_6[:,2], bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + 3*bar_width, spread_6[:,3], bar_width,
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