# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday November 14 13:36:53 2021                                #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Invoking HSPICE to find optimal combinations of                              #
# inverters and fans to minimize delays                                        #
################################################################################

import numpy as np  # For using numpy functions
import subprocess   # For invoking hspice
import shutil       # For copying the files over
import string       # Creating lower case ascii dictionary

MAX_FAN = 10        # Maximum number of fans to try
MAX_INVERTERS = 15  # Maximum number of inverters to try
INDEX_ASCII_DICT = dict(zip(range(1, 27), string.ascii_lowercase))  # A dictionary to store mappings between indexes and ascii characters
Z_INDEX = 26        # The last ascii index (for z)
HEADER_FILE = "header.sp"   # Header file, used to create the Hspice input file
SPICE_EXTENSION = ".sp"   # Extension for hspice files"
OUTPUT_FILE_EXTENSION = ".mt0.csv"     # HSpice output extension"
TPHL = "tphl_inv"      # HSpice tphl variable name in the output file


# Main function to iterate the combinations of fan and inverters
# It creates a Hspice file for each combination, invokes Hspice, reads output and gets the minimum delay
def main():
    min_tph = 100               # Initialize the delay to be 100 (so that the next one's would be less than this)
    optimal_fan = 1             # let's assume 1 fan is optimal
    optimal_inverter = 1        # let's assume 1 inverter is optimal
    for fan in range(2, MAX_FAN + 1):           # Iterate for Fans starting 2, since we will have a min of 1
        for inv in range(1, MAX_INVERTERS + 1, 1):    # Iterate all the combinations of inverter for the fan value above

            iteration_name = "inv" + str(inv) + "-" + str(fan)     # A temporary variable to identify the iteration that's happening, used as file name
            shutil.copy(HEADER_FILE, iteration_name + SPICE_EXTENSION)   # Copy the header file into the iteration_name.sp file for appending inverter/fan data

            f = open(iteration_name + SPICE_EXTENSION, 'a')             # Open the newly copied file in append mode.
            text = get_fan_text(fan)                            # Call the get_fan_text function to get fan string for input file
            f.write(text)                                       # write the data to the file

            # For a single inverter we go directly to 'z' from a
            if (inv == 1):
                text = get_single_inverter_text()      # Get the single inverter text
                tphl = write_file_and_print(f, text, iteration_name, fan, inv)    # Write it to the file we created, do required printing

                # Compare the time delay, if lesser then we take the fan and inverter values as optimal
                if float(tphl) < min_tph:
                    min_tph = tphl
                    optimal_fan = fan
                    optimal_inverter = inv

            # For > 1 inverters we go from a to b first and then loop over.
            if (inv > 1):
                counter = 2         # A counter to maintain inverter/fan numbers for the input file
                text = get_first_inverter_in_multiple()     # Get the single inverter text for multiple inverter scenario
                f.write(text)           # Write the data to the file
                if inv % 2 == 0:        # After first inverter, we only add even number of inverters as discussed in the lectures
                    for index in range(2, inv + 1, 1):      # Iterate from 2 to the inverter count
                        text = get_middle_inverters(counter, INDEX_ASCII_DICT[index], INDEX_ASCII_DICT[index + 1], counter - 1)    # Get inverter text for the inverter count
                        counter = counter + 1       # Increment the counter as we are moving on to the next inverter
                        f.write(text)               # Write to the file

                        # We need an additional check for the last inverter, since it has to end with 'z'
                        if (index == inv):
                            text = get_middle_inverters(counter, INDEX_ASCII_DICT[index + 1], INDEX_ASCII_DICT[Z_INDEX], counter - 1)  # Get text for the last inverter
                            tphl = write_file_and_print(f, text, iteration_name, fan, inv) # Write it to the file we created, do required printing

                            # Compare the time delay, if lesser then we take the fan and inverter values as optimal
                            if tphl < min_tph:
                                min_tph = tphl
                                optimal_fan = fan
                                optimal_inverter = inv

    # Finally print the optimal configurations obtained
    print_optional_configurations(min_tph, optimal_fan, optimal_inverter)

# Function to run the HSPICE tool
# Input - iteration name: used as the file name for reading input and processing output
# Returns the time delay obtained for the input file
def run_hspice(iteration_name):
    proc = subprocess.Popen(["hspice", iteration_name + SPICE_EXTENSION], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)     # Invoke hspice using subprocess
    output, err = proc.communicate()            # Get the output and errors
    data = np.recfromcsv(iteration_name + OUTPUT_FILE_EXTENSION, comments="$", skip_header=3) # Process the output, ignoring 3 line headers and comments
    tphl = data[TPHL]   # Extract the time delay
    return tphl         # Return the time delay


# Function to print the configurations for fan, inv and time delay for each combination
def print_configurations(fan, inv, tphl):
    if (inv == 1):
        print("N ", inv, "\t Fan ", fan,  "\t tphl ", format(tphl, '.15f'))
    else:
        print("N ", inv+1, "\t Fan ", fan, "\t tphl ", format(tphl, '.15f'))


# Function to print the final optimal configurations for the time delay, fan and inverter count
def print_optional_configurations(min_tph, optimal_fan, optimal_inverter):
    print("Best values were:")
    print("\t fan ",optimal_fan )
    print("\t num_inverters ", optimal_inverter+1)
    print("\t tphl", min_tph)

# Function to get the fan string in the Hspice input file
def get_fan_text(fan):
    return '\n.param fan = {}\n'.format(fan)

# Function to get the inverter string for a single inverter scenario
def get_single_inverter_text():
    return 'Xinv1 a z inv M=1\n'

# Function to get the inverter string for the first inverter for multiple inverter scenario
def get_first_inverter_in_multiple():
    return 'Xinv1 a b inv M=1\n'

# Function to get the end string for the inverter block
def get_end_text():
    return '.end\n'

# Function to create middle inverter strings for the inverter_number variable. They go from 'from_ascii' to 'to_ascii' for the fan_number
def get_middle_inverters(inverter_number, from_ascii, to_ascii, fan_number):
    return 'Xinv{} {} {} inv M=fan**{}\n'.format(inverter_number, from_ascii,
                                                                         to_ascii, fan_number)

# Function to write all the text to the file 'f' by the name 'iteration_name'
# It also prints out the configuration
def write_file_and_print(f, text, iteration_name, fan, inv):
    f.write(text)               # Write the data
    f.write(get_end_text())     # Get the end text
    f.close()                   # Close the file, otherwise this would not work
    tphl = run_hspice(iteration_name)       # Invoke hspice function
    print_configurations(fan, inv, tphl)    # Invoke print configurations function
    return tphl                 # Return the calculated time delay


# Starting point of the program
main()