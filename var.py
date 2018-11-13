# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 22:25:53 2018

@author: William
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

def var_fct(alpha, correlation, uncond_def_prob):
    root      = np.sqrt(1 - correlation * correlation)
    ppf_alpha = norm.ppf(alpha) # inverse dist functio, ppf: percent point function
    ppf_prob  = norm.ppf(uncond_def_prob)
    return norm.cdf(( correlation * ppf_alpha + ppf_prob)/ root)

def alpha_fct(theta, correlation, uncond_def_prob):
    root      = np.sqrt(1 - correlation * correlation)
    ppf_theta = norm.ppf(theta) # inverse dist functio, ppf: percent point function
    ppf_prob  = norm.ppf(uncond_def_prob)
    return norm.cdf(( root * ppf_theta - ppf_prob)/ correlation)

x = np.arange(0.01,1,0.001)
rho = 0.2
def_prob = 0.04

plt.plot(var_fct(x,0.2,0.05))
var_fct(0.001,0.2,0.05)
