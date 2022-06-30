################################################################################
# Created on Mon Sep 2021                                                      #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist

    node_cnt = 0                       # initialize node count to 0
    volt_cnt = 0                       # initialize voltage source count
    
    for comp in netlist:               # for each component in netlist
        
        # Get the i and j values to calculate the node count
        i_val = comp[COMP.I]
        j_val = comp[COMP.J]
                
        # node_cnt is the maximum of i, j and node_cnt, as they are starting from 0 .. N sequentially, and we ignore 0
        node_cnt = max(node_cnt, i_val, j_val)
        
        if ( comp[COMP.TYPE] == COMP.VS ):   # If the component is a voltage source, increase voltage count
            volt_cnt += 1

    #print(' Nodes ', node_cnt, ' Voltage sources ', volt_cnt)
    return node_cnt,volt_cnt

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add,netlist,currents,num_nodes):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...
    
    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1
        
        if ( comp[COMP.TYPE] == COMP.R ):           # a resistor
            if (i >= 0):                                # negative index ignored
                y_add[i][i] += 1.0/comp[COMP.VAL]       # add 1/R to admittance[i,i]
            
            if (j >= 0):                                # negative index ignored
                y_add[j][j] += 1.0/comp[COMP.VAL]       # add 1/R to admittance[j,j]

            if (j >= 0 and i >= 0):                     # ignore if either of i or j is negative       
                y_add[i][j] -= 1.0/comp[COMP.VAL]       # subtract 1/R from admittance[i, j]
                y_add[j][i] -= 1.0/comp[COMP.VAL]       # subtract 1/R from admittance[j, i]
            
        elif ( comp[COMP.TYPE] == COMP.IS ):           # a current source
            if (i >= 0):                                # negative index ignored
                currents[i] -= comp[COMP.VAL]           # subtract value of current from currents[i]
            if (j >= 0):                                # negative index ignored
                currents[j] += comp[COMP.VAL]           # add value of current to currents[j]
                
        elif ( comp[COMP.TYPE] == COMP.VS ):           # a voltage source
        
            # Part1 of stamping for admittance matrix
            node_count = num_nodes                              # node_count = number of rows in admittance matrix

            if (i >= 0):                                # ignore negative indexes
                y_add[node_count][ i] = 1                        # set admittance[node_count, i] = 1
                y_add[i][ node_count] = 1                        # set admittance[i, node_count] = 1
            
            if (j >= 0):                                # ignore negative indexes
                y_add[node_count][ j] = -1                       # set admittance[node_count, j] = -1
                y_add[j][ node_count] = -1                       # set admittance[j, node_count] = -1
            
            #Part2 of stamping current matrix
            
            currents[node_count] = comp[COMP.VAL]                # Set the node_count element as current value
            num_nodes += 1                               # Nn has increased by 1 now
            
            # Part3
            # Voltage matrix automatically gets appended with 0 with solve 
            # because the solution would always have same dimensions as admittance and currents
        
        


    voltage_vector = solve(y_add, currents)               # use np.solve to solve for voltage
    print(voltage_vector)
    return num_nodes  # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################


# Main function to start execution
def main():    
    # Read the netlist!
    netlist = read_netlist()
    
    # Print the netlist so we can verify we've read it correctly
    # for index in range(len(netlist)):
    #    print(netlist[index])
    # print("\n")
    
    #EXTRA STUFF HERE!
    node_cnt,volt_cnt = get_dimensions(netlist)     # get netlist dimensions
    
    total_nodes = node_cnt + volt_cnt                         # total_nodes = size of rows = size of columns = Nn + Nv
    
    #print(total_nodes)
    
    currents = np.zeros(total_nodes) # create a current matrix N x 1 (N = number of nodes)
    
    y_add = [ [ 0 for k in range(total_nodes) ] for l in range(total_nodes) ] # create an admittance matrix with 0 values (total_nodes X total_nodes)
    
    
    # Begin stamping for admittance matrix, current matrix using netlist and node count variables
    stamper(y_add, netlist, currents, node_cnt)


# This is the starting point of the program
main()