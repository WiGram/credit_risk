# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:43:32 2018

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
# =============================== Model ===================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# 1. Payoffs to stocks and bonds

# Model parameters
spot        = 100
r           = 0.01 # risk free return rate
q           = 0.0  # Dividends
mu          = 0.05 # real return rate (not used here)
vol         = 0.25
mat         = 1    # Being at t = 0 this functions also as time_to_mat.
debt_junior = 50
debt_senior = 50
debt_total  = debt_junior + debt_senior

# Firm value is denoted by FV
FV = np.arange(1,201,1)

# S = Stock, B = bond, where debt_i functions as strike price
s = m1.bsOptionPrice(FV, vol, mat, debt_total, r, q, "call")
b_senior = FV - m1.bsOptionPrice(FV, vol, mat, debt_senior, r, q, "call")
b_junior = m1.bsOptionPrice(FV, vol, mat, debt_senior, r, q, "call") -\
            m1.bsOptionPrice(FV, vol, mat, debt_total, r, q, "call")

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 2. Risk structure of stocks and bonds

# New model parameters (Firm value is now given by V)
V   = (50,100,150)
mat = 30
time_to_mat = np.arange(1,mat + 1, 1)
s_rs           = np.zeros((mat, 2*len(V)))
b_senior_rs    = np.zeros((mat, 2*len(V)))
b_junior_rs    = np.zeros((mat, 2*len(V)))

for k in V:
    s_rs[:,V.index(k)] = \
      m1.bsOptionPrice(k, vol, time_to_mat, debt_total, r, q, "call")
    b_senior_rs[:,V.index(k)] = \
      k - m1.bsOptionPrice(k, vol, time_to_mat, debt_senior, r, q, "call")
    b_junior_rs[:, V.index(k)] = \
      m1.bsOptionPrice(k, vol, time_to_mat, debt_senior, r, q, "call") -\
       m1.bsOptionPrice(k, vol, time_to_mat, debt_total, r, q, "call")

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 3. Risk Structure of interest rates

# New model parameters
V   = (90,130,180)
mat = 30
time_to_mat = np.arange(1,mat + 1, 1)
b_rs = np.zeros((mat, 2*len(V)))
y  = np.zeros((mat, 2*len(V)))
    
for k in V:
    b_rs[:,V.index(k)] =\
      k - m1.bsOptionPrice(k, vol, time_to_mat, debt_senior, r, q, "call")
    b_rs[:,V.index(k) + len(V)] =\
      m1.bsOptionPrice(k, vol, time_to_mat, debt_senior, r, q, "call") -\
       m1.bsOptionPrice(k, vol, time_to_mat, debt_total, r, q, "call")
    y[:,V.index(k)] =\
      100 * ( 1 / time_to_mat * np.log(debt_senior / b_rs[:,V.index(k)]) - r)
    y[:,V.index(k) + len(V)] =\
      100 * ( 1 / time_to_mat * np.log(debt_junior / b_rs[:,V.index(k) + len(V)]) - r)

# --------------------------------------------------------------------------- #
# 3.b factor difference between senior and junior debt for each firm value
      
y_factor = np.zeros((mat, len(V)))

for k in V:
    y_factor[:,V.index(k)] = y[:,V.index(k) + len(V)] / y[:,V.index(k)]



# =========================================================================== #
# ============================== Plotting =================================== #
# =========================================================================== #

# 1. Payoffs
fig, ax = plt.subplots(figsize = (6,5))
ax.plot(FV, s,        label = 'Stock payoff')
ax.plot(FV, b_senior, label = 'Senior bond payoff')
ax.plot(FV, b_junior, label = 'Junior bond payoff')
ax.set_title('Development in stock and bond payoffs')
ax.legend(loc = 'upper left', shadow = False)
ax.set_ylabel('Value')
ax.set_xlabel('Firm value')
ax.set_xlim(xmin = FV[0], xmax = FV[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_values.png', bbox_inches='tight')
#plt.close(fig)

# 2. Risk structure on stocks and bonds
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (18,5))
ax1.plot(time_to_mat, s_rs[:,0], label = 'V = 50')
ax1.plot(time_to_mat, s_rs[:,1], label = 'V = 100')
ax1.plot(time_to_mat, s_rs[:,2], label = 'V = 150')
ax1.set_title('Payoff to the stock')
ax1.legend(loc = 'upper left', shadow = False)
ax1.set_ylabel('Value')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
ax2.plot(time_to_mat, b_senior_rs[:,0], label = 'V = 50')
ax2.plot(time_to_mat, b_senior_rs[:,1], label = 'V = 100')
ax2.plot(time_to_mat, b_senior_rs[:,2], label = 'V = 150')
ax2.set_title('Payoff to senior debt')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Value')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
ax3.plot(time_to_mat, b_junior_rs[:,0], label = 'V = 50')
ax3.plot(time_to_mat, b_junior_rs[:,1], label = 'V = 100')
ax3.plot(time_to_mat, b_junior_rs[:,2], label = 'V = 150')
ax3.set_title('Payoff to junior debt')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('Value')
ax3.set_xlabel('Time to maturity')
ax3.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_risk_structure_s_and_b.png', bbox_inches='tight')
#plt.close(fig)

# 3. Risk structure on interest rates
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (16,5))
ax1.plot(time_to_mat, y[:,0], label = 'V = 90')
ax1.plot(time_to_mat, y[:,1], label = 'V = 130')
ax1.plot(time_to_mat, y[:,2], label = 'V = 180')
ax1.set_title('Credit spread on Senior debt')
ax1.legend(loc = 'lower right', shadow = False)
ax1.set_ylabel('Spread')
ax1.set_xlabel('Time to maturity')
ax1.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
ax2.plot(time_to_mat, y[:,3], label = 'V = 90')
ax2.plot(time_to_mat, y[:,4], label = 'V = 130')
ax2.plot(time_to_mat, y[:,5], label = 'V = 180')
ax2.set_title('Credit spread on Junior debt')
ax2.legend(loc = 'upper right', shadow = False)
ax2.set_ylabel('Spread')
ax2.set_xlabel('Time to maturity')
ax2.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
ax3.plot(time_to_mat[4:], y_factor[4:,0], label = 'V = 90')
ax3.plot(time_to_mat[4:], y_factor[4:,1], label = 'V = 130')
ax3.plot(time_to_mat[4:], y_factor[4:,2], label = 'V = 180')
ax3.set_title('Spread factor between junior debt and senior debt')
ax3.legend(loc = 'upper right', shadow = False)
ax3.set_ylabel('factor')
ax3.set_xlabel('Time to maturity')
ax3.set_xlim(xmin = time_to_mat[0], xmax = time_to_mat[-1])
fig.tight_layout()
#fig.savefig('C:/Users/William/Dropbox/Code/Python/Finance/merton_risk_structure_int_rates.png', bbox_inches='tight')
#plt.close(fig)