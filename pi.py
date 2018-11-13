# -*- coding: utf-8 -*-
"""
Created on Mon May 21 09:50:16 2018

@author: William
"""

from math import pi
import random

circle, square = 0, 0
ratios = []

precision = 0.1

simulating = True

while precision >= 0.000001:
    print("Precision is: ", precision)
    while simulating:
        x, y = random.random(), random.random()
        if (0.5 - x)**2 + (0.5 - y)**2 <= 0.25:
            circle += 1
        # Regardless, any point will belong to the unit square
        # so every time we will add the point here too
        square += 1
        
        # The circle area is pi and the square area is 2 * 2 = 4
        # hence the ratio n / d will be pi / 4, so we multiply by 4.
        ratio = 4.0 * circle / square
                
        # For each c I measure how many draws it takes to become
        # sufficiently accurate
        if abs(ratio - pi) / pi <= precision:
            print("Draws needed: ", square)
            print("Ratio is: ", ratio)
            print('Error is: ', abs(ratio - pi))
            break
    
    precision = precision / 10