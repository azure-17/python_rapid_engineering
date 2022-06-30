# -*- coding: utf-8 -*-
#################################################################################################
# Created on Thursday November 27 13:36:53 2021                                                 #
#                                                                                               #
# @author: Isha Joshi                                                                           #
#                                                                                               #
# Compute the average value of pi found for the successful attempts at each precision.          #
#################################################################################################


import random         # For using random numbers
import numpy as np           # for using the actual value of pi

# Constants
precision_list = [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07] # Defined precisions to try on
MAX_ITERATIONS = 100
MAX_TRIALS = 10000
RADIUS_OF_CIRCLE = 1

# Function to get the radius of a circle at origin
# Circle's equation at origin = x**2 + y**2 = r**2
# r = np.sqrt(x**2 + y**2)
# input x and y 
# output radius
def get_radius(x, y):
    return np.sqrt(x**2 + y**2)

# Function to copute the value of pi
# input - points inside and total points checked
# output computed pi
def compute_pi(points_inside, total_points):
    return 4 * (points_inside/total_points)


# Function to print the values in the required format
# input - precision, valid trials and sum of the calculated pi
# output - prints
def print_output(precision, valid_trials, pi_sum):
    if valid_trials > 0:
        print ("{0:} success {1:} times {2:}".format (precision, valid_trials, pi_sum / valid_trials))
    else:
        print ("{0:} no success".format (precision))

# Main function to try calculating pi value using random numbers
# It invokes the other helper functions
def main():
    
    random.seed()      # Seed the random number
    
    for precision_value in precision_list:      # Iterate over the different precision values to try
        valid_trials = 0                        # A temporary variable to store the valid trial count
        calculated_pi_sum = 0                   # A temporary variable to store the sum of calculated pi's value
    
        for iteration in range(MAX_ITERATIONS):         # Iterate MAX_ITERATIONS time
            points_in_cirlce = 0.0                  # A temporary variable to store the count of points lying inside the circle
            total_points = 0.0                      # A temporary variable to store the total points that we check
        
            for trial in range (MAX_TRIALS):        # Iterate for upto MAX_TRIALS
            
                total_points += 1                   # increment thet total points temporary variable by 1
                x = random.random()                     # Get the next random number in the range [0.0, 1.0)
                y = random.random()                     # Get the next random number in the range [0.0, 1.0)
                r = get_radius(x, y)                # Call the get radius function to get the radius

                if r <= RADIUS_OF_CIRCLE:                          
                    points_in_cirlce += 1           # If the calculated value of radius is <=1 then the point is inside, increment the temp
                
                pi_compute = compute_pi(points_in_cirlce, total_points)    # Call the compute_pi function to get the computed value of pi

                if abs(pi_compute - np.pi) <= precision_value:     # If the difference of computed pi is in precesion range of np.pi
                    valid_trials += 1                                       # We mark the trial as valid by incrementing the counter
                    calculated_pi_sum += pi_compute                        # We also add the calculated pi value to the sum variable
                    break                                                  # Since it is valid, no need to go test further, we break
    
        print_output(precision_value, valid_trials, calculated_pi_sum)      # Call the print output function to actually print the data in required format

# This is the starting point of the program        
main()