# -*- coding: utf-8 -*-
"""
Created on Tue May  1 19:46:07 2018

@author: William
"""

import numpy as np
# from scipy.stats import norm
from matplotlib import pyplot as plt

from scipy.stats import binom

#plt.style.available
plt.style.use("seaborn-white")

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) Model parameters
n           = 125
k           = np.arange(0,n+1,1)
p           = np.array((0.01,0.05))  # Must contain exactly two elements until I produce more flexible code.
detachment  = np.array((3,6,9,12,22,100))
exp_payoffs = np.zeros((len(detachment),len(p)))
bin_prob    = np.zeros((len(p), len(k)))


for i in p:
    bin_prob[list(p).index(i),:] = binom.pmf(k, n, i)
    for j in detachment:
        if list(detachment).index(j) == 0:
            exp_payoffs[list(detachment).index(j),list(p).index(i)] = sum(np.maximum(j - 0.8 * k,0) * bin_prob[list(p).index(i)])
        else:
            exp_payoffs[list(detachment).index(j),list(p).index(i)] = sum(\
                (np.maximum(j - 0.8 * k,0) - np.maximum(detachment[list(detachment).index(j)-1] - 0.8 * k,0))*\
                bin_prob[list(p).index(i)])

np.savetxt("exp_payoffs.csv", exp_payoffs)

# ============================== Credit spread ============================== #

spread = np.zeros((len(detachment), len(p)))
k = 0

for m in detachment:
    mx = list(detachment).index(m)
    diff = m - k
    spread[mx] = diff / exp_payoffs[mx] - 1
    k = m

np.savetxt("credit_spreads.csv", spread)


fig, ax = plt.subplots()

index = np.arange(len(detachment))
bar_width = 0.35

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, spread[:,0], bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='P = 0.01')

rects2 = plt.bar(index + bar_width, spread[:,1], bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='P = 0.05')

plt.xlabel('Tranche')
plt.ylabel('Credit spread')
plt.title('Credit spread by tranche and default probability')
plt.xticks(index + bar_width / 2, ('Equity', 'Junior Mezzanine', 'Senior Mezzanine', 'Super Senior Mezzanine', 'Super Secure'))
plt.legend()

plt.tight_layout()
plt.show()


# =========================================================================== #
# =================================== Fin =================================== #
# =========================================================================== #