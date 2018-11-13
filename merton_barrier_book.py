# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:11:54 2018

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

def m_fct(interest, vol):
    return interest - 0.5*vol*vol

def d1_fct(spot, vol, mat, interest, bound_or_strike):
    m = m_fct(interest, vol)
    return (np.log(spot/bound_or_strike) + m*mat)/(vol * np.sqrt(mat))

def f_lower_bound_and_out(spot, vol, mat, interest, bound):
    spot_adj = bound*bound/spot
    m        = m_fct(interest, vol)
    d1_1     = d1_fct(spot, vol, mat, interest, bound)
    d1_2     = d1_fct(spot_adj, vol, mat, interest, bound)
    factor   = np.power(bound/spot,2*m/(vol*vol))
    return bound * np.exp(-interest * mat) * (norm.cdf(d1_1) - factor * norm.cdf(d1_2))

def call_lower_bound_and_out(spot, vol, mat, strike, interest, bound):
    m        = m_fct(interest, vol)
    factor   = np.power((bound/spot),2*m/(vol*vol))
    spot_adj = bound*bound/spot
    call_1   = m1.bsOptionPrice(spot, vol, mat, bound, interest, 0, "call")
    call_2   = m1.bsOptionPrice(spot_adj, vol, mat, bound, interest, 0, "call")
    call_3   = m1.bsOptionPrice(spot, vol, mat, strike, interest, 0, "call")
    call_4   = m1.bsOptionPrice(spot_adj, vol, mat, strike, interest, 0, "call")
    return call_1 - factor*call_2 - (call_3 - factor * call_4)

def b_lower_bound(spot, vol, mat, strike, interest, bound):
     return f_lower_bound_and_out(spot, vol, mat, interest, bound) +\
      call_lower_bound_and_out(spot, vol, mat, strike, interest, bound)

def B_m(spot, vol, mat, strike, interest, dividend, boundary, bound_rate):
    gt             = np.exp(-bound_rate*mat) #bound_rate is parametrised as (g)amma
    r_tilde        = interest - dividend - bound_rate
    spot_tilde     = gt*strike
    strike_tilde   = gt*strike
    bound_tilde    = gt*boundary
    #exp            = np.exp((bound_rate*time - dividend)*(mat-time)) -> time = 0. 
    return b_lower_bound(spot_tilde, vol, mat, strike_tilde, r_tilde, bound_tilde)

def B_b(spot, vol, mat, strike, interest, dividend, bound, bound_rate):
    b       = (np.log(bound/spot) - bound_rate * mat)/vol
    r_tilde = interest - dividend - bound_rate
    m       = m_fct(r_tilde, vol) / vol
    m_tilde = np.sqrt(m*m + 2*dividend)
    exp1    = bound * np.exp((m - m_tilde) * b)
    exp2    = np.exp(2*m_tilde*b)
    d1      = (b - m_tilde*mat)/np.sqrt(mat)
    d2      = (b + m_tilde*mat)/np.sqrt(mat)
    return exp1*(norm.cdf(d1) + exp2 * norm.cdf(d2))

# The next formula, is the final one, which prices bond payoffs in a barrier setting
def B(spot, vol, mat, strike, interest, dividend, bound, bound_rate):
    return B_m(spot, vol, mat, strike, interest, dividend, bound, bound_rate) +\
     B_b(spot, vol, mat, strike, interest, dividend, bound, bound_rate)

# =========================================================================== #
# ================================== Model ================================== #
# =========================================================================== #

# --------------------------------------------------------------------------- #
# (a) Model parameters
spot       = 120
vol        = 0.2
# mat        = np.arange(1,31,1)
mat        = 1
strike     = 80
# strike     = np.arange(70,150,10)
r          = 0.05
dividend   = 0
# bound      = 90
bound      = np.arange(70, 150, 10)
bound_rate = 0.02


b_mat = B_m(spot, vol, mat, strike, r, dividend, bound, bound_rate)
b_no_mat = B_b(spot, vol, mat, strike, r, dividend, bound, bound_rate)
test = B(spot, vol, mat, strike, r, dividend, bound, bound_rate)

# --------------------------------------------------------------------------- #