#Python for Rapid Engineering Solutions
#            Homework #1
#
# Finding optimal configuration for a circus performer to be launched as a
# projectile 
# @author: Isha Joshi.

import numpy as np              # Importing numpy for sin/cos/arange/round etc.


INITIAL_HEIGHT = 1              # Initial height of the canon's exit
GRAVITY = 9.81                  # Gravitantional force, used as acceleration
PI = np.pi                      # Numpy.pi value = 3.14.. used for angle iteration.


# Function to calculate the maximum range
# Input: theta - angle at which the projectile is thrown at.
#        initial_velocity - velocity from the cannon, provided as input.
# Output: float range - maximum horizontal distance, in (m), upto 1 decimal place.
# Formula: (Vi**2 * sin2(0) ) / g (Where Vi is initial velocity, g is gravity)
def get_max_range(theta, initial_velocity):
    velocity_squared = pow(initial_velocity, 2)               # Will provide velocity to the power 2
    sin_2_theta = np.sin(2 * theta)                           # 2sin(theta)*cost(theta) = sin(2*theta) since range is max at theta = 45
    max_range = (velocity_squared * sin_2_theta) / GRAVITY
    return np.round(max_range, 1)                     #Return range rounded upto 1 decimal place


# Function to calculate the maximum height
# Input: theta - angle at which the projectile is thrown at.
#        initial_velocity - velocity from the cannon, provided as input.
# Output: float height - maximum vertical distance, in (m), upto 1 decimal place.
# Formula: Hi + (Vi**2 * sin(0)**2 ) / 2*g (Where Vi is initial velocity, g is gravity and Hi is initial height)
def get_max_height(theta, initial_velocity):
    velocity_squared = pow(initial_velocity, 2)        # Will provide velocity to the power 2
    sin_theta_squared = pow(np.sin(theta), 2)          # Will provide sin(0) to the power 2
    max_height = INITIAL_HEIGHT + ((velocity_squared * sin_theta_squared) / (2 * GRAVITY))      # Caputing cannon's height towards max height
    return np.round(max_height, 1)                     # Return height rounded upto 1 decimal place.



# This function performs the constraints validations
# It then prints out the optimal configurations in the required format.
def optimal_configurations_validator(range_list, height_list, angle_list):
    max_range = -1;                                 # initialize temporary maximum range to -1
    max_height = -1;                                 # initialize temporary maximum height to -1
    optimal_angle = -1;                                 # initialize temporary maximum angle to -1
    
    for range_index in range(len(range_list)):                                 # iterate the range_list from 0 to length of the range_list
        range_val = range_list[range_index]                                     # get the value from the range_list at the index
        
        if (range_val == -1):                                           # If the range value was -1, we have nothing to do
            continue
        
        elif (range_val == max_range):                                  # If value is not -1, and the range_value is equal to our max_range value (Meaning 2 angles have same range)
            height = height_list[range_index]                           # We get the height at that same index
            if (height > max_height):                                   # Compare which height is greater
                max_height = height                                     # We pick the greater of the two heights
                optimal_angle = angle_list[range_index]                 # We update our angle to be the angle that gave max_height
                
        elif (range_val > max_range):                                   # If value is not -1, and the range_value is greater than our max_range value
            max_range = range_val                                   # Simply update our max range to new max range
            max_height = height_list[range_index]                                   # update the new height corresponding to the max range
            optimal_angle = angle_list[range_index]                                   # update the new angle corresponding to the max range

    if (max_range <= 0):                                   # If max_range is still -1 or 0, means none of the constraints were fulfilled and print the error condition
        print("")
        print("Requirements can't be met. Please reduce initial velocity.")
    else:                                       # Otherwise, print the optimal configuration
        print("")    
        print("optimal settings:")
        print("angle: "+ str(optimal_angle) +" radians")
        print("landing: "+ str(max_range) +" m")
        print("max height: "+ str(max_height) +" m")
        print("")
        print("possible landing spots:")
        print(np.round(range_list,1))                       # Need to round it here again because this did not work on a mac
    

# This function takes the required inputs - height and length of the circus and initial velocity.
# It then creates 3 objects that store different configurations at each angle.
def begin():
    height_input = float(input("enter height: "))             # User input - maximum height of the celiing
    length_input = float(input("enter length: "))             # User input - maximum length of the circus' landing net
    initial_velocity_input = float(input("enter velocity: ")) # User input - initial velocity of the performer

    range_list=[]                                           # To store different range values at each angle
    height_list=[]                                           # To store different height values at each angle
    angle_list=[]                                           # To store different angles

    for theta in np.arange(PI/24, PI/2, PI/24):                         # start a variable theta from PI/24, go until PI/2 (not including) in increments of PI/24
        angle_list.append(np.round(theta, 3))                            # adding angles to the angle list, rounded up to 3 decimal places
        max_range = get_max_range(theta, initial_velocity_input)        # call get_max_range to calculate max range at the theta value
        max_height = get_max_height(theta, initial_velocity_input)      # call get_max_height to calculate max height at the theta value
        if (max_range <= length_input) and (max_height <= height_input):    # range is valid only if circus performer lands within bounds of celiing and landing net
            range_list.append(max_range)                            # adding range to range list if range conditions are met
            height_list.append(max_height)                            # adding height to height list if range conditions are met
        else:
            range_list.append(-1)                            # adding -1 to range list if range conditions are NOT met

    optimal_configurations_validator(range_list, height_list, angle_list)                            # call optimal_configurations_validator function to do constraint checks and print output        


# This is the starting point of the script, invoking the begin function here
begin()