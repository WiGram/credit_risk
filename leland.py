# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:36:12 2018

@author: William
"""

import numpy as np
from matplotlib import pyplot as plt

#plt.style.available
plt.style.use("seaborn-white")

# =========================================================================== #
# ================================ Functions ================================ #
# =========================================================================== #


def beta_fct(r, dividend, vol):
    mu = r - dividend
    kappa = mu - 0.5 * vol**2
    return -(kappa + vol * np.sqrt(kappa**2 + 2 * r * vol ** 2)) / (vol**2)

def kappa_fct(r, dividend, vol, tax):
    beta = beta_fct(r, dividend, vol)
    return beta / (beta - 1) * (1 - tax) / r

def coupon_fct(spot, r, dividend, vol, tax):
    beta  = beta_fct(r, dividend, vol)
    kappa = kappa_fct(r, dividend, vol, tax)
    return spot * kappa**(-1) * (((1 - beta)*tax - alpha * beta * (1 - tax))/tax)**(1/beta)

def barrier_fct(spot, r, dividend, vol, tax):
    kappa = kappa_fct(r, dividend, vol, tax)
    coupon = coupon_fct(spot, r, dividend, vol, tax)
    return kappa * coupon

def p_b_fct(spot, asset, r, dividend, vol, tax):
    beta = beta_fct(r, dividend, vol)
    barrier = barrier_fct(spot, r, dividend, vol, tax)
    return (asset / barrier)**beta

def debt_fct(spot, asset, r, dividend, vol, tax, alpha):
    coupon = coupon_fct(spot, r, dividend, vol, tax)
    barrier = barrier_fct(spot, r, dividend, vol, tax)
    p = p_b_fct(spot, asset, r, dividend, vol, tax)
    return np.where(asset <= barrier, 
                    barrier * (1 - alpha),
                    coupon / r * (1 - p) + (1 - alpha) * barrier * p)

def equity_fct(spot, asset, r, dividend, vol, tax, alpha):
    coupon = coupon_fct(spot, r, dividend, vol, tax)
    barrier = barrier_fct(spot, r, dividend, vol, tax)
    p = p_b_fct(spot, asset, r, dividend, vol, tax)
    debt = debt_fct(spot, asset, r, dividend, vol, tax, alpha)
    return np.where(asset <= barrier,
                    0,
                    asset + coupon * (1 - p) * tax / r - debt - alpha * barrier * p)
    

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) Model parameters
# --------------------------------------------------------------------------- #
spot     = 100
V        = np.arange(30,80,1)
tax      = 0.35
vol      = 0.2
r        = 0.05
dividend = 0
alpha    = 0.5

# --------------------------------------------------------------------------- #
# (a.2) Helper functions
# --------------------------------------------------------------------------- #
mu       = r - dividend
kappa    = (mu - 0.5 * vol**2)
beta     = -(kappa + vol * np.sqrt(kappa**2 + 2 * r * vol ** 2)) / (vol**2)
gamma    = beta/(beta - 1) * (1 - tax)/r

# --------------------------------------------------------------------------- #
# (a.3) Optimal coupon and boundary
# --------------------------------------------------------------------------- #
coupon  = coupon_fct(spot, r, dividend, vol, tax)
barrier = barrier_fct(spot, r, dividend, vol, tax)

spot_vec    = np.arange(80,120,1)
coupon_vec  = coupon_fct(spot_vec, r, dividend, vol, tax)
barrier_vec = barrier_fct(spot_vec, r, dividend, vol, tax)

# --------------------------------------------------------------------------- #
debt   = debt_fct(spot, V, r, dividend, vol, tax, alpha)
y      = coupon / debt
s      = 100 * (y - r)
equity = equity_fct(spot, V, r, dividend, vol, tax, alpha)
# --------------------------------------------------------------------------- #

# =========================================================================== #
# =========================== Comparative Statics =========================== #
# =========================================================================== #
spot     = 100
V        = np.arange(30,80,1)
dividend = 0

tax_1    = 0.15
tax_2    = 0.3

vol_1    = 0.2
vol_3    = 0.3

alpha_1  = 0.5
alpha_4  = 0.7

r_1      = 0.03
r_5      = 0.05

# --------------------------------------------------------------------------- #
c_1 = coupon_fct(spot, r_1, dividend, vol_1, tax_1)
d_1 = debt_fct(spot, V, r_1, dividend, vol_1, tax_1, alpha_1)
y_1 = c_1 / d_1
s_1 = 100 * (y_1 - r_1)

c_2 = coupon_fct(spot, r_1, dividend, vol_1, tax_2)
d_2 = debt_fct(spot, V, r_1, dividend, vol_1, tax_2, alpha_1)
y_2 = c_2 / d_2
s_2 = 100 * (y_2 - r_1)

c_3 = coupon_fct(spot, r_1, dividend, vol_3, tax_1)
d_3 = debt_fct(spot, V, r_1, dividend, vol_3, tax_1, alpha_1)
y_3 = c_3 / d_3
s_3 = 100 * (y_3 - r_1)

c_4 = coupon_fct(spot, r_1, dividend, vol_1, tax_1)
d_4 = debt_fct(spot, V, r_1, dividend, vol_1, tax_1, alpha_4)
y_4 = c_4 / d_4
s_4 = 100 * (y_4 - r_1)

c_5 = coupon_fct(spot, r_5, dividend, vol_1, tax_1)
d_5 = debt_fct(spot, V, r_5, dividend, vol_1, tax_1, alpha_1)
y_5 = c_5 / d_5
s_5 = 100 * (y_5 - r_5)

# =========================================================================== #
# ================================== Plots ================================== #
# =========================================================================== #

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16,5))
ax1.plot(V, equity, label = 'Equity')
ax1.plot(V, debt, label = 'Debt')
ax1.set_title('Value of equity and debt')
ax1.legend(loc = 'lower right', shadow = False)
ax1.set_ylabel('Value (USD)')
ax1.set_xlabel('Value of assets')
ax2.plot(V, s, label = 'Credit spread')
ax2.set_title('Credit spread on debt')
ax2.set_ylabel('Spread (pct.)')
ax2.set_xlabel('Value of assets')
ax3.plot(spot_vec, coupon_vec, label = 'Coupons')
ax3.plot(spot_vec, barrier_vec, label = 'Barrier')
ax3.set_title('Value in coupons and barrier')
ax3.legend(loc = 'upper left', shadow = False)
ax3.set_ylabel('Value (USD)')
ax3.set_xlabel('Spot value of firm assets')
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/leland.png', bbox_inches='tight')

# (b) Comparative statics

#---- Komparativ statik - m_1 til m_2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (16,8))
ax1.plot(V, s_1, label = 'Tax = 0.15')
ax1.plot(V, s_2, label = 'Tax = 0.3')
ax1.set_title('Yield spread (bp)')
ax1.legend(loc = 'upper right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Value of assets')
ax1.set_xlim(xmin = V[0], xmax = V[-1])
ax2.plot(V, s_1, label = 'Volatility = 0.2')
ax2.plot(V, s_3, label = 'Volatility = 0.3')
ax2.set_title('Yield spread (bp)')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Value of assets')
ax2.set_xlim(xmin = V[0], xmax = V[-1])
ax3.plot(V, s_1, label = 'Bankruptcy costs = 0.4')
ax3.plot(V, s_4, label = 'Bankruptcy costs = 0.5')
ax3.set_title('Yield spread (bp)')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('Spread')
ax3.set_xlabel('Value of assets')
ax3.set_xlim(xmin = V[0], xmax = V[-1])
ax4.plot(V, s_1, label = 'Interest rate = 0.3')
ax4.plot(V, s_5, label = 'Interest rate = 0.5')
ax4.set_title('Yield spread (bp)')
ax4.legend(loc = 'upper right', shadow = False)
ax4.set_ylabel('Spread')
ax4.set_xlabel('Value of assets')
ax4.set_xlim(xmin = V[0], xmax = V[-1])
fig.tight_layout()
fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/leland_comparative_statics.png', bbox_inches='tight')
#plt.close(fig)


# =========================================================================== #
# =================================== Fin =================================== #
# =========================================================================== #