# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 09:26:50 2018

@author: William
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.append('C:/Users/William/.spyder-py3/')
import module1 as m1
from matplotlib import pyplot as plt

#plt.style.available
plt.style.use("seaborn-white")

# =========================================================================== #
# ================================ Functions ================================ #
# =========================================================================== #

def r_fct(interest, vol):
    return interest - 0.5*vol*vol

def d1_fct(spot, vol, mat, interest, bound_or_strike):
    return (np.log(spot/bound_or_strike) + (interest + 0.5*vol*vol) * mat) / (vol * np.sqrt(mat))

def d2_fct(spot, vol, mat, interest, bound_or_strike):
    return d1_fct(spot, vol, mat, interest, bound_or_strike) - vol * np.sqrt(mat)

def f_lo(spot, vol, mat, interest, bound, bound_rate):
    spot_adj = bound*bound/spot
    d1_1     = d1_fct(spot, vol, mat, interest, bound)
    d1_2     = d1_fct(spot_adj, vol, mat, interest, bound)
    r_tilde  = interest + 0.5 * vol * vol
    factor   = np.power(bound/spot,2*r_tilde/(vol*vol))
    return np.where(spot > bound,\
                    spot * (norm.cdf(d1_1) -factor * norm.cdf(d1_2)),\
                    bound*np.exp(-bound_rate*mat))
    
def c_lo(spot, vol, mat, strike, interest, bound, bound_rate):
    spot_adj      = bound*bound/spot
    r_tilde_plus  = interest + 0.5 * vol * vol
    r_tilde       = interest - 0.5 * vol * vol
    factor_minus  = np.power((bound/spot),2*r_tilde/(vol*vol))
    factor_plus   = np.power((bound/spot),2*r_tilde_plus/(vol*vol))
    strike_factor = strike * np.exp(-interest * mat)
    d1_1          = d1_fct(spot, vol, mat, interest, strike)
    d1_2          = d1_fct(spot_adj, vol, mat, interest, strike)
    d2_1          = d1_1 - vol*np.sqrt(mat)
    d2_2          = d1_2 - vol*np.sqrt(mat)
    return np.where(bound < strike, spot * (norm.cdf(d1_1) - factor_plus * norm.cdf(d1_2)) -\
     strike_factor * (norm.cdf(d2_1) - factor_minus * norm.cdf(d2_2)),0)

def c_lo_bs(spot, vol, mat, strike, interest, dividends, bound, bound_rate):
    spot_adj      = bound * bound / spot
    r_tilde       = interest - 0.5 * vol * vol
    factor        = np.power(bound / spot, 2 * r_tilde / (vol*vol))
    return np.where(bound < strike, m1.bsOptionPrice(spot, vol, mat, strike, interest, dividends, "call") -\
            factor * m1.bsOptionPrice(spot_adj, vol, mat, strike, interest, dividends, "call"),0)

# Book manipulation:
    # (a) spot is not discounted by gamma.
def B_m(spot, vol, mat, strike, interest, dividend, bound, bound_rate):
    r_hat        = interest - dividend - bound_rate
    gt           = np.exp(-bound_rate*mat)
    # spot_tilde   = spot*gt # Wrong graph is producet ammending the spot, but must be correct to do so
    spot_tilde   = spot
    strike_tilde = strike*gt
    bound_tilde  = bound*gt
    return np.exp(-dividend * mat) * (f_lo(spot_tilde, vol, mat, r_hat, bound_tilde, bound_rate) -\
      c_lo_bs(spot_tilde, vol, mat, strike_tilde, r_hat, 0, bound_tilde, bound_rate))

# Følgende giver kvalitativt resultaterne i bogen:
    # (a) alpha i m_tilde indeholder kun renten
    # (b) exp1 er ufølsom overfor bound_rate.
def B_b(spot, vol, mat, interest, dividend, bound, bound_rate):
    b       = (np.log(bound/spot) - bound_rate * mat)/vol
    r_tilde = interest - dividend - bound_rate - 0.5 * vol * vol
    m       = r_tilde / vol
    m_tilde = np.sqrt(m*m + 2*(interest - 0* bound_rate))
    exp1    = np.exp((m - m_tilde)*b)*np.exp(-0*bound_rate*mat)
    exp2    = np.exp(2*m_tilde*b)
    d1      = (b - m_tilde*mat)/np.sqrt(mat)
    d2      = (b + m_tilde*mat)/np.sqrt(mat)
    return bound * exp1 * (norm.cdf(d1) + exp2 * norm.cdf(d2))

# The next formula, is the final one, which prices bond payoffs in a barrier setting
def B(spot, vol, mat, strike, interest, dividend, bound, bound_rate):
    return B_m(spot, vol, mat, strike, interest, dividend, bound, bound_rate) +\
     B_b(spot, vol, mat, interest, dividend, bound, bound_rate)

# The stock price without dividends is a down-and-out-call
def stock(spot, vol, mat, strike, interest, bound):
    spot_adj          = bound*bound/spot
    r_tilde           = r_fct(interest, vol)
    factor            = np.power((bound/spot),2*r_tilde/(vol*vol))
    strike_factor     = strike * np.exp(-interest * mat)
    d1_1              = d1_fct(spot, vol, mat, interest, strike)
    d1_2              = d1_fct(spot_adj, vol, mat, interest, strike)
    d2_1              = d1_1 - vol*np.sqrt(mat)
    d2_2              = d1_2 - vol*np.sqrt(mat)
    d2_1_bound        = d2_fct(spot, vol, mat, interest, bound)
    d2_2_bound        = d2_fct(spot_adj, vol, mat, interest, bound)
    strike_over_bound = spot * (norm.cdf(d1_1) - factor * norm.cdf(d1_2)) -\
     strike_factor * (norm.cdf(d2_1) - factor * norm.cdf(d2_2))
    bound_over_strike = strike_over_bound + (bound - strike) * np.exp(-interest * mat) *\
     (norm.cdf(d2_1_bound) - factor * norm.cdf(d2_2_bound))
    return np.where(bound < strike, strike_over_bound, bound_over_strike)
    

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) Model parameters
# --------------------------------------------------------------------------- #
spot   = 120
vol    = 0.2
mat    = np.arange(0.01,10.01,0.01)
debt   = 100
r      = 0.05
bound  = 90
gamma  = 0.02
q      = 0

# --------------------------------------------------------------------------- #
# (b) Merton with exogenous barrier
# --------------------------------------------------------------------------- #

bond_m = B_m(spot, vol, mat, debt, r, q, bound, gamma)
bond_b = B_b(spot, vol, mat, r, q, bound, gamma)

# --------------------------------------------------------------------------- #

bond_barrier  = B(spot, vol, mat, debt, r, q, bound, gamma)
y_barrier     = 1 / mat * np.log(debt / bond_barrier)
s_barrier     = 10000 * (y_barrier - r)

# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (c) Merton base case
# --------------------------------------------------------------------------- #

bond_standard = spot - m1.bsOptionPrice(spot, vol, mat, debt, r, q, "call")
y_standard    = 1 / mat * np.log(debt / bond_standard)
s_standard    = 10000 * (y_standard - r)
# --------------------------------------------------------------------------- #

# =========================================================================== #
# =========================== Comparative Statics =========================== #
# =========================================================================== #

spot   = 120
debt   = 100
mat    = np.arange(0.01,10.01,0.01)

vol_1 = 0.2
vol_2 = 0.3

r_1   = 0.03
r_3   = 0.05

bound_1 = 85
bound_4 = 95

gamma_1 = 0.02
gamma_5 = 0.04

# Base case: bb = bond_barrier, yb = yield, sb = spread
bb_1  = B(spot, vol_1, mat, debt, r_1, q, bound_1, gamma_1)
yb_1     = 1 / mat * np.log(debt / bb_1)
sb_1     = 10000 * (yb_1 - r_1)

bb_2  = B(spot, vol_2, mat, debt, r_1, q, bound_1, gamma_1)
yb_2     = 1 / mat * np.log(debt / bb_2)
sb_2     = 10000 * (yb_2 - r_1)

bb_3  = B(spot, vol_1, mat, debt, r_3, q, bound_1, gamma_1)
yb_3    = 1 / mat * np.log(debt / bb_3)
sb_3     = 10000 * (yb_3 - r_3)

bb_4  = B(spot, vol_1, mat, debt, r_1, q, bound_4, gamma_1)
yb_4     = 1 / mat * np.log(debt / bb_4)
sb_4     = 10000 * (yb_4 - r_1)

bb_5  = B(spot, vol_1, mat, debt, r_1, q, bound_1, gamma_5)
yb_5     = 1 / mat * np.log(debt / bb_5)
sb_5     = 10000 * (yb_5 - r_1)


# =========================================================================== #
# ================================== Plots ================================== #
# =========================================================================== #

# (a) Barrier bond_m and bond_b

fig, ax = plt.subplots(figsize = (6,5))
ax.plot(mat, bond_m, label = 'Bond maturity')
ax.plot(mat, bond_b, label = 'Bond default')
ax.set_title('Bond payoff')
ax.legend(loc = 'upper right', shadow = False)
ax.set_ylabel('Value (dollars)')
ax.set_xlabel('Time to maturity')
ax.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_values.png', bbox_inches='tight')
#plt.close(fig)


# (b) Merton bond values 

fig, ax = plt.subplots(figsize = (6,5))
ax.plot(mat, bond_standard, label = 'Standard Merton')
ax.plot(mat, bond_barrier, label = 'Merton with Barrier')
ax.set_title('Payoffs')
ax.legend(loc = 'upper right', shadow = False)
ax.set_ylabel('Value (dollars)')
ax.set_xlabel('Time to maturity')
ax.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_values.png', bbox_inches='tight')
#plt.close(fig)


# (c) Merton Barrier spreads

fig, ax = plt.subplots(figsize = (8,5))
ax.plot(mat, s_standard, label = 'Standard Merton')
ax.plot(mat, s_barrier, label = 'Merton with Barrier')
ax.set_title('Credit spreads')
ax.legend(loc = 'upper right', shadow = False)
ax.set_ylabel('Yield spread (bps)')
ax.set_xlabel('Time to maturity')
ax.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/bc_m_spread.png', bbox_inches='tight')
#plt.close(fig)


# (d) Comparative statics

#---- Komparativ statik - m_1 til m_2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (16,8))
ax1.plot(mat, sb_1, label = 'Volatility = 0.2')
ax1.plot(mat, sb_2, label = 'Volatility = 0.3')
ax1.set_title('Yield spread (bp)')
ax1.legend(loc = 'upper right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = mat[0], xmax = mat[-1])
ax2.plot(mat, sb_1, label = 'Interest rate = 0.3')
ax2.plot(mat, sb_3, label = 'Interest rate = 0.5')
ax2.set_title('Yield spread (bp)')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = mat[0], xmax = mat[-1])
ax3.plot(mat, sb_1, label = 'Default barrier = 85')
ax3.plot(mat, sb_4, label = 'Default barrier = 95')
ax3.set_title('Yield spread (bp)')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('Spread')
ax3.set_xlabel('Time to maturity')
ax3.set_xlim(xmin = mat[0], xmax = mat[-1])
ax4.plot(mat, sb_1, label = 'Gamma = 0.2')
ax4.plot(mat, sb_5, label = 'Gamma = 0.4')
ax4.set_title('Yield spread (bp)')
ax4.legend(loc = 'upper right', shadow = False)
ax4.set_ylabel('Spread')
ax4.set_xlabel('Time to maturity')
ax4.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/bar_comparative_statics.png', bbox_inches='tight')
#plt.close(fig)


# =========================================================================== #
# =================================== Fin =================================== #
# =========================================================================== #