# -*- coding: utf-8 -*-
"""
Created on Mon May 14 12:38:16 2018

@author: William
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

np.random.seed(21432)

# Brownian motion

spot = 100.0
vol  = 0.2
mat  = 10
strike = 100
r    = 0.05
mu   = 0.07
q    = 0
nsteps = 100

def bm_fct(spot, vol, mat, return_rate, dividend,nsteps):
    spot = np.repeat(spot, mat * nsteps)
    dt = 1 / (mat * nsteps)
    for i in range(1,len(spot)):
        spot[i] = spot[i-1] * np.exp((return_rate - dividend - 0.5 * vol*vol ) * dt + vol * np.sqrt(dt)*np.random.normal(0,1))
    return spot

time_to_mat = np.arange(0, mat, 1.0 / nsteps)

asset_price = bm_fct(spot, vol, mat, r, q, nsteps)
initial_barrier = 90
barrier_decay = 0.02
barrier = initial_barrier * np.exp( -barrier_decay * (mat - time_to_mat))

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(time_to_mat, asset_price, label = 'Asset value')
ax.plot(time_to_mat, barrier, label = 'Barrier')
ax.plot(time_to_mat, strike*time_to_mat / time_to_mat, label = 'Strike')
ax.set_title('Asset value, strike and credit barrier')
ax.legend(loc = 'lower right', shadow = False)
ax.set_ylabel('Value (USD)')
ax.set_xlabel('Time to maturity')
ax.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/bc_barriere.png', bbox_inches='tight')
#plt.close(fig)

90 * np.exp(-2)

def ret_fct(spot):
    log_ret = np.repeat(0., len(spot))
    for i in range(1, len(spot)):
        log_ret[i] = np.log(spot[i]/spot[i-1])
    return log_ret

asset_return = ret_fct(asset_price)
sample_mean = sum(asset_return)/len(asset_price)

test3 = np.abs(asset_return - sample_mean)

sample_vol  = np.sqrt(sum(asset_return**2)/len(asset_return))

test_vol = np.sqrt((asset_return[1] - sample_mean)**2)

plt.plot(test3)

count, bins, ignored = plt.hist(asset_return, 30, normed=True)
plt.plot(bins, 1 / (np.sqrt(2 * sample_vol**2 * np.pi)) * np.exp( - (bins - sample_mean)**2 / (2 * sample_vol**2)), linewidth=2, color='r')
plt.show()

stats.probplot(asset_return, dist="norm", plot=plt)
plt.title("Normal Q-Q plot")
plt.show()