# -*- coding: utf-8 -*-
################################################################################
# Created on Tuesday November 24 13:36:53 2021                                 #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Solving first and second order derivatives using odeint and plotting         #
################################################################################


# Organize the imports
import numpy as np                           # Used for numpy libraries for calculation
import matplotlib.pyplot as plt              # Used for plotting
from scipy.integrate import odeint           # Used to solve derivatives, this is the ODE solver

# Constants
TIME_VEC = np.linspace(0, 7, 700)            # create the time steps
Y_0_PROBLEM_1 = 1                            # Initial value for problem1
Y_0_PROBLEM_2 = 0                            # Initial value for problem2
Y_0_PROBLEM_3 = [1, 1]                       # Initial value for problem3

###############################################################################
# Problem 1: # y’ = cos(t); initial value: y(0)=1
###############################################################################

# Method to return the derivative for first problem
# Inputs:                                                                      
#    ypos - current position                                                   
#    time - current time                                                       
# Outputs:                                                                     
#    returns the derivative  
def problem1_derivative(ypos, time):
 return np.cos(time)                          # return the derivative


# Main method for problem1
# This invokes the python odeint solver and then calls the common plot function
def problem1_main():
    yvec = odeint(problem1_derivative, Y_0_PROBLEM_1, TIME_VEC)     #Call the ode solver
    plot(" Function: y’ = cos(t) ", yvec, False)     #Plot the graph
    

###############################################################################
# Problem 2: # y’ = -y + t2e-2t +10; initial value y(0) = 0
###############################################################################

# Method to return the derivative for second problem
# Inputs:                                                                      
#    ypos - current position                                                   
#    time - current time                                                       
# Outputs:                                                                     
#    returns the derivative  
def problem2_derivative(ypos, time):
     return -ypos + (time**2) * np.exp(-2*time) + 10                  # return the derivative


# Main method for problem2
# This invokes the python odeint solver and then calls the common plot function
def problem2_main():
    yvec = odeint(problem2_derivative, Y_0_PROBLEM_2, TIME_VEC)     #Call the ode solver
    plot(" Function: y’ = -y + (t**2) * (e**-2t) +10 ", yvec, False)     #Plot the graph



###############################################################################
# Problem 3: # y''+4y'+4y = 25cos(t) + 25sin(t); initial values y(0) = 1, y’(0) =1
###############################################################################

# Method to return the derivative for the third problem
# Inputs:                                                                      
#    ypos - current position                                                   
#    time - current time                                                       
# Outputs:                                                                     
#    returns the derivative  
def problem3_derivative(ypos, time):
    return  [ypos[1], 25*np.cos(time) + 25*np.sin(time) - 4*ypos[1] - 4*ypos[0]]          # return the derivative
         

# Main method for problem3
# This invokes the python odeint solver and then calls the common plot function
def problem3_main():    
    yvec = odeint(problem3_derivative, Y_0_PROBLEM_3, TIME_VEC)     #Call the ode solver
    plot(" Function: y'' = 25cos(t) + 25sin(t) - 4y' - 4y ", yvec, True)     #Plot the graph


###############################################################################
# Common functions
###############################################################################

# Method to plot the graphs
# Inputs:                                                                      
#    title - title for the plot                                                   
#    yvec - values of y to be plotted against the time
#    is_second_order - a boolean to determine whether yvec is for second order or not                                                       
# Outputs:                                                                     
#    creates the plot
def plot(title, yvec, is_second_order):    
    
    if (is_second_order):                           # Check if the plot is for second order
        plt.plot(TIME_VEC, yvec[:,0] , label= 'y')     # If yes, plot the y vs t
        plt.plot(TIME_VEC, yvec[:,1], label= "y'")     # Also plot the y' vs t on the same plot
    else:
        plt.plot(TIME_VEC,yvec, label='y')          # If plot is not for second order, just plot y vs t
    
    plt.title(title)                # Add the title to the plot
    plt.xlabel('Time')              # Add x label
    plt.ylabel('y')                 # Add y label
    plt.legend()                    # Add legends
    plt.show()                      # Show th eplot
    
# Function to invoke all three mains for respective problems
def main():    
    problem1_main()
    problem2_main()
    problem3_main()
    
    
# This is the starting point of the program.
main()