# Function to be integrated: x * e^(-x^2)                                   
# input:                                                                       
#    x: the variable to integrate                                                                                     
# output:                                                                      
#    returns the value of the function   
# @author: Isha                                         


import numpy as np                     # get exp function
from scipy.integrate import quad       # get the integration function


# Function to calculate infinite integral using substitution
# Subsiting x with z / (1-z) and integrating from 0 to 1
def exponential_function(z):
    # Substituting x with z/1-z
    x = z / (1-z)                     # From Newman 5.67
    exp_function = x * np.exp(-x**2)  # Function to resolve
    z_dz = 1 / ( (1-z)**2 )        # dx = dz/(1-z)**2
    return z_dz * exp_function    # From newman 5.68


# Function to call numpy's quad and exponential_function and print output
def begin():
    res_1, err = quad(exponential_function, 0, 1, args=())  # Call Numpy's quad to integrate with range from 0 to 1
    print("Value computed is "+'{0:.8f}'.format(res_1))    # {0:.8f} would print 8 decimal numbers in res_1
    
# Starting point of the program
begin()
