# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 13:59:37 2018

@author: William
"""

import numpy as np
import sys
sys.path.append('C:/Users/William/.spyder-py3/')
import module1 as m1
from matplotlib import pyplot as plt

#plt.style.available
plt.style.use("seaborn-white")

# =========================================================================== #
# ============================= Functions =================================== #
# =========================================================================== #

# I here define the factorial function.
def factorial(x):
    if x == 0:
        return 1
    else:
        return x * factorial(x-1)

# The stock price is still that of a call price, now in the merton jump model
def merton_option_price(spot, vol, mat, strike, r, lambdaa, m, v):
    price_element = 1
    price = 0
    n = 0
    # jump_stat = m + 0.5*v
    # comp      = lambdaa * (np.exp(jump_stat) - 1)
    lambda_dt = lambdaa*(1+m)*mat
    gamma     = np.log(1+m)
    
    while price_element > 0.00001:
        r_tilde    = r + (n * gamma)/mat - lambdaa*m
        # spot_tilde = spot*np.exp(n*jump_stat - comp * mat)
        vol_tilde  = np.sqrt(vol*vol + n*v / mat)
        # Without hashtags spot below should be replaced by spot_tilde
        cond_price = m1.bsOptionPrice(spot, vol_tilde, mat, strike,\
                                      r_tilde, 0, "call")
        weight     = np.exp(-lambda_dt) * np.power(lambda_dt, n) / factorial(n)
        price_element = cond_price * weight
        price += price_element
        n += 1
    
    return price

m_jmp_func = np.vectorize(merton_option_price)


# =========================================================================== #
# =============================== Model ===================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) Model parameters
r           = 0.0 # risk free return rate
q           = 0.0  # Dividends
vol         = 0.25 # Volatility of firm value

lambdaa     = 0.2  # Jump intensity 
m           = -0.5 # Mean jump size
v           = 0.16 # Jump variance

debt_jnr = 50
debt_snr = 50
debt_ttl  = debt_jnr + debt_snr

# s = stock payoff, b = bond payoff, y = yield (spread)
spot      = (90,100,130,180)
mat       = np.arange(1,31,1)
s_jmp     = np.zeros((len(spot),len(mat)))
b_jmp_snr = np.zeros((len(spot),len(mat)))
b_jmp_jnr = np.zeros((len(spot),len(mat)))
y_jmp_snr = np.zeros((len(spot),len(mat)))
y_jmp_jnr = np.zeros((len(spot),len(mat)))

s_std     = np.zeros((len(spot),len(mat)))
b_std_snr = np.zeros((len(spot),len(mat)))
b_std_jnr = np.zeros((len(spot),len(mat)))
y_std_snr = np.zeros((len(spot),len(mat)))
y_std_jnr = np.zeros((len(spot),len(mat)))

intra_jmp = np.zeros((len(spot),len(mat)))
inter_snr = np.zeros((len(spot),len(mat)))
inter_jnr = np.zeros((len(spot),len(mat)))
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# (b) Deriving stock payoff, bond payoff and yield spread.
for i in spot:
    ix = spot.index(i)
    
    # Payoffs: jmp
    s_jmp[ix,:]     = m_jmp_func(i, vol, mat, debt_ttl, r, lambdaa, m, v)
    b_jmp_snr[ix,:] = i - m_jmp_func(i, vol, mat, debt_snr, r, lambdaa, m, v)
    b_jmp_jnr[ix,:] = m_jmp_func(i, vol, mat, debt_snr, r, lambdaa, m, v) -\
                       m_jmp_func(i, vol, mat, debt_ttl, r, lambdaa, m, v)
    # Payoffs: std
    s_std[ix, :]     = m1.bsOptionPrice(i, vol, mat, debt_ttl, r, q, 'call')
    b_std_snr[ix, :] = i - m1.bsOptionPrice(i, vol, mat, debt_snr, r, q, 'call')
    b_std_jnr[ix, :] = m1.bsOptionPrice(i, vol, mat, debt_snr, r, q, 'call') -\
                        m1.bsOptionPrice(i, vol, mat, debt_ttl, r, q, 'call')

    # Credit spreads: jmp
    y_jmp_snr[ix,:] = 100 * (1 / mat * np.log(debt_snr / b_jmp_snr[ix,:]) - r)
    y_jmp_jnr[ix,:] = 100 * (1 / mat * np.log(debt_jnr / b_jmp_jnr[ix,:]) - r)
    # Credit spreads: std
    y_std_snr[ix,:] = 100 * (1 / mat * np.log(debt_snr / b_std_snr[ix,:]) - r)
    y_std_jnr[ix,:] = 100 * (1 / mat * np.log(debt_jnr / b_std_jnr[ix,:]) - r)

    # Factors    
    intra_jmp[ix,:] = y_jmp_jnr[ix,:] / y_jmp_snr[ix,:]
    inter_snr[ix,:] = y_jmp_snr[ix,:] / y_std_snr[ix,:]
    inter_jnr[ix,:] = y_jmp_jnr[ix,:] / y_std_jnr[ix,:]
# --------------------------------------------------------------------------- #

# =========================================================================== #
# ========================= Komparativ Statik =============================== #
# =========================================================================== #

r_1       = 0.0

q         = 0.0  # Dividends
vol_1     = 0.15 # Volatility of firm value
vol_3     = 0.35 

lambdaa_1 = 0.2  # Jump intensity
lambdaa_5 = 0.1

m_1       = -0.1 # Mean jump size
m_2       = -0.5 # Mean jump size

v_1       = 0.16 # Jump variance
v_4       = 0.26

debt      = 100
spot      = 130
mat       = np.arange(1,31,1)
    
# Payoffs: jmp
b_jmp_1 = spot - m_jmp_func(spot, vol_1, mat, debt, r_1, lambdaa_1, m_1, v_1)
y_jmp_1 = 100 * (1 / mat * np.log(debt / b_jmp_1) - r_1)

b_jmp_2 = spot - m_jmp_func(spot, vol_1, mat, debt, r_1, lambdaa_1, m_2, v_1)
y_jmp_2 = 100 * (1 / mat * np.log(debt / b_jmp_2) - r_1)

b_jmp_3 = spot - m_jmp_func(spot, vol_3, mat, debt, r_1, lambdaa_1, m_1, v_1)
y_jmp_3 = 100 * (1 / mat * np.log(debt / b_jmp_3) - r_1)

b_jmp_4 = spot - m_jmp_func(spot, vol_1, mat, debt, r_1, lambdaa_1, m_1, v_4)
y_jmp_4 = 100 * (1 / mat * np.log(debt / b_jmp_4) - r_1)

b_jmp_5 = spot - m_jmp_func(spot, vol_1, mat, debt, r_1, lambdaa_5, m_1, v_1)
y_jmp_5 = 100 * (1 / mat * np.log(debt / b_jmp_5) - r_1)
    
# =========================================================================== #
# ============================== Plotting =================================== #
# =========================================================================== #
        
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16,5))
ax1.plot(mat, y_jmp_snr[0,:], label = 'V = 90')
ax1.plot(mat, y_jmp_snr[1,:], label = 'V = 100')
ax1.plot(mat, y_jmp_snr[2,:], label = 'V = 130')
ax1.plot(mat, y_jmp_snr[3,:], label = 'V = 180')
ax1.set_title('Senior yield spread (bp)')
ax1.legend(loc = 'lower right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = mat[0], xmax = mat[-1])
ax2.plot(mat, y_jmp_jnr[0,:], label = 'V = 90')
ax2.plot(mat, y_jmp_jnr[1,:], label = 'V = 100')
ax2.plot(mat, y_jmp_jnr[2,:], label = 'V = 130')
ax2.plot(mat, y_jmp_jnr[3,:], label = 'V = 180')
ax2.set_title('Junior yield spread (bp)')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = mat[0], xmax = mat[-1])
ax3.plot(mat[4:], intra_jmp[0,4:], label = 'V = 90')
ax3.plot(mat[4:], intra_jmp[1,4:], label = 'V = 100')
ax3.plot(mat[4:], intra_jmp[2,4:], label = 'V = 130')
ax3.plot(mat[4:], intra_jmp[3,4:], label = 'V = 180')
ax3.set_title('Spread factor between junior debt and senior debt')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('Factor')
ax3.set_xlabel('Time to maturity')
ax3.set_xlim(xmin = mat[4], xmax = mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/jmp_risk_structure_int_rates.png', bbox_inches='tight')
#plt.close(fig)

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
ax1.plot(mat[1:], inter_snr[0,1:], label = 'V = 90')
ax1.plot(mat[1:], inter_snr[1,1:], label = 'V = 100')
ax1.plot(mat[1:], inter_snr[2,1:], label = 'V = 130')
ax1.plot(mat[1:], inter_snr[3,1:], label = 'V = 180')
ax1.set_title('Senior yield spread (bp)')
ax1.legend(loc = 'lower right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = mat[0], xmax = mat[-1])
ax2.plot(mat, inter_jnr[0,:], label = 'V = 90')
ax2.plot(mat, inter_jnr[1,:], label = 'V = 100')
ax2.plot(mat, inter_jnr[2,:], label = 'V = 130')
ax2.plot(mat, inter_jnr[3,:], label = 'V = 180')
ax2.set_title('Junior yield spread (bp)')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_jmp_std_factor_comparison.png', bbox_inches='tight')
#plt.close(fig)



#---- Komparativ statik - m_1 til m_2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (16,8))
ax1.plot(mat, y_jmp_1, label = 'Mean jump size = -0.1')
ax1.plot(mat, y_jmp_2, label = 'Mean jump size = -0.5')
ax1.set_title('Yield spread (bp)')
ax1.legend(loc = 'upper right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = mat[0], xmax = mat[-1])
ax2.plot(mat, y_jmp_1, label = 'Vol = 0.15')
ax2.plot(mat, y_jmp_3, label = 'Vol = 0.35')
ax2.set_title('Yield spread (bp)')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = mat[0], xmax = mat[-1])
ax3.plot(mat, y_jmp_1, label = 'Jump variance = 0.16')
ax3.plot(mat, y_jmp_4, label = 'Jump variance = 0.16')
ax3.set_title('Yield spread (bp)')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('Spread')
ax3.set_xlabel('Time to maturity')
ax3.set_xlim(xmin = mat[0], xmax = mat[-1])
ax4.plot(mat, y_jmp_1, label = 'Jump intensity = 0.2')
ax4.plot(mat, y_jmp_5, label = 'Jump intensity = 0.1')
ax4.set_title('Yield spread (bp)')
ax4.legend(loc = 'upper right', shadow = False)
ax4.set_ylabel('Spread')
ax4.set_xlabel('Time to maturity')
ax4.set_xlim(xmin = mat[0], xmax = mat[-1])
fig.tight_layout()
fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/jmp_comparative_statics.png', bbox_inches='tight')
#plt.close(fig)