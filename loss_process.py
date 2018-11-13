# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 20:10:20 2018

@author: William
"""

import numpy as np
from matplotlib import pyplot as plt

np.random.seed(54321)

def cir(x, mr_lvl, mr_spd, vol, time_to_mat, n_steps):
    x = np.repeat(x, time_to_mat * n_steps)
    dt = 1 / (time_to_mat * n_steps)
    for i in range(1, len(x)):
        x[i] = x[i-1] + mr_spd * (mr_lvl - x[i-1]) * dt +\
        vol * np.sqrt(x[i-1]) * np.random.normal(0,1) * np.sqrt(dt)
    return x


def bm_fct(spot, vol, mat, return_rate,nsteps):
    spot = np.repeat(spot, mat * nsteps)
    dt = 1 / (mat * nsteps)
    for i in range(1,len(spot)):
        spot[i] = spot[i-1] * np.exp((return_rate - 0.5 * vol*vol ) * dt + vol * np.sqrt(dt)*np.random.normal(0,1))
    return spot

# CIR process
loss_0 = 0.01
mr_lvl = 0.01
mr_spd = 0.40
vol    = 0.12
mat    = 10
nsteps = 1000

loss_frc = cir(loss_0, mr_lvl, mr_spd, vol, mat, nsteps) * 100
exp_loss = np.repeat(np.average(loss_frc),len(loss_frc))
time_to_mat = np.arange(0, mat, 1.0 / nsteps)

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(time_to_mat, loss_frc, label = 'Tabsandel')
ax.plot(time_to_mat, exp_loss, label = 'Forventede tab')
ax.set_title('Tab og forventede tab')
ax.legend(loc = 'upper right', shadow = False)
ax.set_ylabel('Tabsandel (pct.)')
ax.set_xlabel('Tidshorisont')
ax.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/loss_fraction.png', bbox_inches='tight')
#plt.close(fig)

# Alternativt kunne man lave porteføljeværdien stokastisk:
spot = 100.0
vol  = 0.2
mat  = 10
r    = 0.05
nsteps = 1000

time_to_mat = np.arange(0, mat, 1.0 / nsteps)
loss_on_assets = 0.01


time_to_mat = np.arange(0, mat, 1.0 / nsteps)
asset_price = bm_fct(spot, vol, mat, r, nsteps)
asset_loss  = bm_fct(spot, vol, mat, r, nsteps) * loss_frc/100

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(time_to_mat, asset_price, label = 'Udlånsportefølje DKK Mio.')
#ax.plot(time_to_mat, exp_loss, label = 'Forventede tab')
ax.set_title('Udlånsportefølje')
ax.set_ylabel('Udlån Mio. DKK')
ax.set_xlabel('Tidshorisont')
ax.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/loss_fraction.png', bbox_inches='tight')

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(time_to_mat, asset_loss, label = 'Tab')
#ax.plot(time_to_mat, exp_loss, label = 'Forventede tab')
ax.set_title('Tab og forventede tab')
ax.set_ylabel('Tab (DKK Mio.)')
ax.set_xlabel('Tidshorisont')
ax.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/loss_fraction.png', bbox_inches='tight')
