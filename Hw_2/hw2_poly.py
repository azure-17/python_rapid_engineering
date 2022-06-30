# Function to be integrated: a(x^2) + bx + c                                   
# input:                                                                       
#    x: the variable to integrate                                              
#    a, b and c : constants to be used                                         
# output:                                                                      
#    returns the value of the function   
# @author: Isha                                      

# importing relevant libraries
import numpy as np                     # get array functions
from scipy.integrate import quad       # get the integration function
import matplotlib.pyplot as plt        # get plotting functions


# Function to create the quadratic equation
# Input x, a, b and c
# Output: a (x^2) + bx + c
def quad_function(x, a, b, c):
    x_square = x**2
    return (a * x_square) + (b * x) + c

# Function to loop from 0 to 5 in steps of 0.01 and calling quad_function to integrate, 
# finally plotting the results.
def begin():    
    step_list = []              # List to store step at which integral is calculated
    integrals_list_1 = []       # Storing integral values of first case (a, b, c) = (2, 3, 4)
    integrals_list_2 = []       # Storing integral values of second case (a, b, c) = (2, 1, 1)
    
    
    for step in np.arange(0, 5.01, 0.01):                # start from 0, in step size of 0.01 and upto 5.01 (so that the last picked value is 5) 
        step_list.append(step)                             # store the step value to be plotted as horizonatal axis
    
        # calculate the integral - note two return values (result and error)
        res_1, err1 = quad(quad_function, 0, step, args=(2,3,4))    # Calculate first integral using numpy.quad, from 0 to step with 2, 3, 4 as constants
        res_2, err2 = quad(quad_function, 0, step, args=(2,1,1))    # Calculate first integral using numpy.quad, from 0 to step with 2, 1, 1 as constants
    
        integrals_list_1.append(res_1)          # Collect res_1 to integrals_list_1
        integrals_list_2.append(res_2)          # Collect res_2 to integrals_list_2
    
    # now generate the plot
    plt.plot(step_list, integrals_list_1, label='a=2,b=3,c=4')           # data to plot, step value vs function value
    plt.plot(step_list, integrals_list_2, label='a=2,b=1,c=1')           # data to plot, step value vs function value
    plt.xlabel("Value of step")           # x-axis
    plt.ylabel("Integral of: ax^2 + bx + c")           # y-axis
    plt.title('Quadratic integral - HW2')               # a title
    plt.legend()                            # Create legends based on labels above
    plt.show()                           # and plot it


# Starting point of the program
begin()