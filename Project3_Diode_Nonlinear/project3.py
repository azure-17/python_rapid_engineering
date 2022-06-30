# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday October 28 13:36:53 2021                                 #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Project3: Measuring diode current, using optimize and fsolve                 #
################################################################################


# Our usual set of imports
import numpy as np                            # For numpy functions like exp, initializing lists etc
from scipy.optimize import fsolve             # Using fsolve to solve our non linear equations
import matplotlib.pyplot as plt               # Used for plotting
from scipy import optimize                    # Used to optimize the non linear equation variables


# Constants defined here, mostly relevant to part 1 and some common
VDD_START = 0.1                             # Starting VDD for part 1 iteration
VDD_END = 2.5                               # Ending VDD for part 1 iteration
P1_VDD_STEP = 0.1                           # Step provided for iterating Vdd
I_S = 1E-9                                  # Saturation current constant for part 1
N = 1.7                                     # Ideality value for part 1
Q = 1.6021766208e-19                        # Constant value of the charge - provided in the project desc
KB = 1.380648e-23                            # Boltzman constant - provided in the project desc
R = 11000                                   # Resistance in Ohm
T = 350                                     # Absolute temperature Kelvin, used for Part 1
INITIAL_GUESS = 0.5                         # Initial guess value for Vd for part 1

# Constants relevant to part 2 of the project
AREA = 1.0E-8                           # Crossectional Area of the diode
P2_T = 375                              # Absolute Temperature for the part 2
P2_VDD_STEP = 0.6                       # A constant step value, used in the opt functions as initialization variable for V
TOL = 1e-9                              # Tolerance for error, we iterate until error lies within this tolerance
R_VALUE_INITIAL = 10000                 # Initial value of R to be used for Part 2
PHI_VALUE_INITIAL = 0.8                 # Initial value of Phi to be used for part 2
N_VALUE_INITIAL = 1.5                   # Initial value of Ideality to be used for part 2
MAX_ITERATIONS = 1000                    # Max iterations, this is like a guard, incase tolerance is never met, we don't wanna iterate forever.

# Read the txt file and organize data
DIODE_DATA = np.loadtxt('DiodeIV.txt', dtype='float64')            # Read the input file for source_v and measured_i
SOURCE_VOLTAGE = DIODE_DATA[:, 0]                                  # Create the Diode Voltage Array 1st column
MEASURED_CURRENT = DIODE_DATA[:, 1]                                # Create the Diode Current Array 2nd column
X_TOL = 1e-12                                                      # Tolerance for fsolve
INITIAL_VOLTAGE_GUESS = 0.9                                        # Initial voltage guess for solving the equation with optimized values


# This function computes the current in the diode [USED for both the parts]
# Inputs - voltage in diode, ideality, absolute temperature and saturation current
# Output - current in the diode
def compute_diode_current(voltage_in_diode, n, temp, i_s):
    return i_s * (np.exp( (Q * voltage_in_diode) / (n * KB * temp)) - 1)


# This function returns difference between diode current and the resistor current [USED for both the parts]
# Inputs - voltage in diode, initial voltage, resistance, ideality, temperature and saturation current
# Output - current diff
def solve_diode_v(v_d, v_initial, r, n, temp, i_s):
    return ((v_d - v_initial)/r) + compute_diode_current(v_d, n, temp, i_s)


# This is the starting function for part 1 of the project
def begin_part_1():
    diode_voltage = []                          # Temporary list to store the diode voltage, for plotting
    diode_current = []                          # Temporary list to store the diode current, for plotting
    source_voltage = []                         # Temporary list to store the source voltage, for plotting
    
    # Iterate from the provided start voltage, to the end voltage (inclusive), in steps provided for part 1
    for i in np.arange(VDD_START, VDD_END + P1_VDD_STEP, P1_VDD_STEP):
        roots = fsolve(solve_diode_v, INITIAL_GUESS, (i, R, N, T, I_S))              # Call fsolve to solve our non linear equation for each 'i' initial voltage
        diode_voltage.append(roots[0])                                               # Store the 0 value in the return from fsolve
        
        current = compute_diode_current(roots[0], N, T, I_S)                         # Calculate the actual current at the root provided by the fsolve above
        diode_current.append(current)                                                # Store the current for plotting
        
        source_voltage.append(i)                                                     # Store the iteration voltage.
        
    plot_1(source_voltage, diode_voltage, diode_current)                             #Invoke the plot function, and we're done.


# This function plots the source voltage against the diode current and diode voltage
# Input - source voltage, diode voltage and diode current
# Output - void, creates a plot
def plot_1(source_voltage, diode_voltage, diode_current):
    plt.plot(source_voltage, np.log(diode_current), color = 'blue', label='log(Diode Current) vs Source Voltage')   # Create plot for source voltage and diode current
    plt.plot(diode_voltage, np.log(diode_current), color = 'red', label='log(Diode Current) vs Diode Voltage')      # Create plot for diode voltage and diode current
    plt.title(' Diode Current ')                  # Add a title to the plot 
    plt.ylabel(' Log(Diode Current) ')              # The y label
    plt.xlabel(' Voltages (0.0 - 2.5) ')            # The X label
    axes = plt.gca()                                # Get the axes
    axes.yaxis.grid()                               # Draw the horizontal lines
    plt.legend()                                    # Draw the legends
    plt.show()                                      # Show the plot



# PART 2 functions begin here

################################################################################
# This function does the optimization for the resistor                         #
# Inputs:                                                                      #
#    r_value   - value of the resistor                                         #
#    ide_value - value of the ideality                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################
# NO CHANGE MADE TO THIS FUNCTION
def opt_r(r_value,ide_value,phi_value,area,temp,src_v,meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P1_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return meas_i - diode_i

################################################################################
# This function does the optimization for the ideality (n)                     #
# Inputs:                                                                      #
#    ide_value - value of the ideality                                         #
#    r_value   - value of the resistor                                         #
#    phi_value - value of phi                                                  #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################
def opt_n(ide_value, r_value, phi_value, area,temp, src_v, meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P2_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,(src_v[index],r_value,ide_value,temp,is_value),xtol=X_TOL)[0] # Call fsolve
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return calculate_residual_error_value(meas_i, diode_i)   # Calculate residual values

################################################################################
# This function does the optimization for the ideality (phi)                   #
# Inputs:                                                                      #
#    phi_value - value of phi                                                  #
#    ide_value - value of the ideality                                         #
#    r_value   - value of the resistor                                         #
#    area      - area of the diode                                             #
#    temp      - temperature                                                   #
#    src_v     - source voltage                                                #
#    meas_i    - measured current                                              #
# Outputs:                                                                     #
#    err_array - array of error measurements                                   #
################################################################################
def opt_phi(phi_value, ide_value, r_value, area,temp, src_v, meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = P2_VDD_STEP                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,(src_v[index],r_value,ide_value,temp,is_value),xtol=X_TOL)[0] # Call fsolve
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return calculate_residual_error_value(meas_i, diode_i)            # Calculate residual values

# Calculate residual values
# A simple function to calculate residual values by dividing the meas - diode currents with their sums and a delta
# Added a guard to avoid division by zero
def calculate_residual_error_value(meas_i, diode_i):
    diff = (meas_i - diode_i)
    total = (meas_i + diode_i +1e-15)
    residual = np.divide(diff, total, out=np.zeros_like(diff), where=total!=0)
    return residual

# Begin the part2 of the project
def begin_part_2():
    # We need to run a for loop to keep optipizing values
    # Initializing these variables with the constant init values
    r_value = R_VALUE_INITIAL
    phi_value = PHI_VALUE_INITIAL
    n_value = N_VALUE_INITIAL
    
    p2_voltage_diodes = np.zeros_like(SOURCE_VOLTAGE) # Create an empty array to store diode voltages for plotting

    # Iterate as long as we want (defined above), until the error tolerance is met
    for i in range(MAX_ITERATIONS):
        # Find the R value to optimize
        r_val_opt = optimize.leastsq(opt_r,r_value,args=(n_value, phi_value, AREA, P2_T, SOURCE_VOLTAGE, MEASURED_CURRENT))
        r_value = r_val_opt[0][0]  # Update the r_value to be used in next iteration and below
    
        # Find the n value to optimize
        n_val_opt = optimize.leastsq(opt_n,n_value, args=(r_value,phi_value, AREA, P2_T, SOURCE_VOLTAGE, MEASURED_CURRENT))
        n_value = n_val_opt[0][0] # Update the n_value to be used in next iteration and below
        
        # Find the phi value to optimize
        phi_val_opt = optimize.leastsq(opt_phi,phi_value, args=(n_value,r_value,AREA, P2_T, SOURCE_VOLTAGE, MEASURED_CURRENT))
        phi_value = phi_val_opt[0][0]  # Update the phi_value to be used in next iteration and below
    
        # Calculate error based on the residual value sum divided by total residual values by calling opt_phi
        error = opt_phi(phi_value, n_value, r_value, AREA, P2_T, SOURCE_VOLTAGE, MEASURED_CURRENT)
        # Calculate the average error
        avg_error = np.sum(np.abs(error))/len(error)
        
        # Print all upto 10 decimal places
        print('\nIteration Number: ', i)        # Print the iteration
        print('r_val: %.10f' %r_value)        # Print the r value
        print('ide_val: %.10f' %n_value)        # Print the n value
        print('phi_val: %.10f' %phi_value)               # Print the phi
        print('Average Error: %.10f' %avg_error)             # Print the avg error
        
        # Our break condition, if avg error is less than tolerance, we break free
        if avg_error < TOL:
            break

    Is = AREA * P2_T * P2_T * np.exp(-phi_value * Q / (KB * P2_T ) )   # calculate the saturation current for our newly otimized phi
    for i in range(len(SOURCE_VOLTAGE)):                 # Iterate for all source voltages
        P2_V_diode = optimize.fsolve(solve_diode_v, INITIAL_VOLTAGE_GUESS, (SOURCE_VOLTAGE[i],r_value,n_value, P2_T, Is), xtol=X_TOL)[0]  # Solve for diode voltage using our new optimized r, n and phi
        p2_voltage_diodes[i] = P2_V_diode               # Store for plotting in a list

    diode_current = compute_diode_current(p2_voltage_diodes, n_value, P2_T, Is)          # Calculate the diode current for all diode voltages
    plot_2(diode_current)         # Call the plot 2 function to plot this all


# Plots the estimated current vs measured current
# Input- diode current
def plot_2(diode_current):    
    plt.plot(SOURCE_VOLTAGE, np.log(diode_current), 'rx', label="estimated current", markersize=10) # Create a log scaled diode_current
    plt.plot(SOURCE_VOLTAGE, np.log(MEASURED_CURRENT), '-bo', label="measured current")            # Create a log scaled measured current
    axes = plt.gca()         # Get the axes
    axes.yaxis.grid()           # Draw horizontal lines
    plt.xlabel('Source voltage')       # x label
    plt.ylabel('Diode current (log)')   # y label
    plt.legend(loc="center right")    # legend
    plt.title("Model vs Measured Diode Current vs Voltage")  # plot title
    plt.show()   # Show the plot


# Starting point for part1
begin_part_1()

# Starting point for part2
begin_part_2()
