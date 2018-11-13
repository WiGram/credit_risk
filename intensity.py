# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:56:50 2018

@author: William
"""

import numpy as np
from matplotlib import pyplot as plt
np.random.seed(1245)

# Brownian motion
def bm_fct(spot, vol, mat, return_rate, dividend,nsteps):
    spot = np.repeat(spot, mat * nsteps)
    dt = 1 / (mat * nsteps)
    for i in range(1,len(spot)):
        spot[i] = spot[i-1] * np.exp((return_rate - dividend - 0.5 * vol*vol ) * dt + vol * np.sqrt(dt)*np.random.normal(0,1))
    return spot

# Model parameters
spot = 100.0
vol  = 0.2
mat  = 10
r    = 0.05
mu   = 0.07
q    = 0
nsteps = 100

time_to_mat = np.arange(0, mat, 1.0 / nsteps)
asset_price = bm_fct(spot, vol, mat, r, q, nsteps)



# Discrete step function
y = np.zeros(len(time_to_mat))

y = np.where(time_to_mat < 2, 0,
             np.where(time_to_mat < 4, 1,
                      np.where(time_to_mat < 6, 2,
                               np.where(time_to_mat < 8, 3,
                                        np.where(time_to_mat < 10, 4, 5)
                                       )
                              )
                     )
             )

arealabels = ['E1','E2','E3','E4','E5']
fig, (ax1, ax2) = plt.subplots(ncols = 1, nrows = 2, figsize = (6,9))
ax1.plot(time_to_mat, asset_price, label = 'Asset value')
ax1.set_ylabel('lambda(t)')
ax1.set_xlabel('Time, t')
ax1.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
ax1.fill_between(time_to_mat, 0, asset_price,
                where = time_to_mat <= 10,
                facecolor = 'blue',
                alpha = 0.5)
ax1.fill_between(time_to_mat, 0, asset_price,
                where = time_to_mat <= 8,
                facecolor = 'red',
                alpha = 0.5)
ax1.fill_between(time_to_mat, 0, asset_price,
                where = time_to_mat <= 6,
                facecolor = 'yellow',
                alpha = 0.5)
ax1.fill_between(time_to_mat, 0, asset_price,
                where = time_to_mat <= 4,
                facecolor = 'green',
                alpha = 0.5)
ax1.fill_between(time_to_mat, 0, asset_price,
                where = time_to_mat <= 2,
                facecolor = 'purple',
                alpha = 0.5)
ax1.text(1, 50, arealabels[0])
ax1.text(3, 50, arealabels[1])
ax1.text(5, 50, arealabels[2])
ax1.text(7, 50, arealabels[3])
ax1.text(9, 50, arealabels[4])
ax2.plot(time_to_mat, y, label = 'Defaults')
ax2.set_ylabel('Defaults')
ax2.set_xlabel('Time, t')
ax2.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/jumps.png', bbox_inches='tight')
#plt.close(fig)
