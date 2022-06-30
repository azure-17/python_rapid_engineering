# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday October 14 13:36:53 2021                               #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Project1 - part1: Read dataset, correlation, covariance and pairplots        #
################################################################################


import numpy as np                # Used for tri function of numpy
from pandas import read_csv       # Used for reading csv with pandas and other pandas operations
import seaborn as sns             # Used for creating pair plots
import matplotlib.pyplot as plt   # Used for showing the plots


# Input file for the dataset
DATASET_INPUT_FILE_NAME = "heart1.csv"

# Variable of interest - heart attack in this case
CLASS_DEPENDENT_VARIABLE = "a1p2"

# Print statements for correlations/covariance
PRINT_TOP_5_ALL_VARIABLE_CORRELATIONS = "Top 5 heart1.csv correlations\n"
PRINT_TOP_5_DEPENDENT_VARIABLE_CORRELATIONS = "Top 5 heart disease (a1p2) correlation\n"
PRINT_TOP_5_ALL_VARIABLE_COVARIANCE = "Top 5 heart1.csv covariances\n"
PRINT_TOP_5_DEPENDENT_VARIABLE_COVARIANCE = "Top 5 heart disease (a1p2) correlation\n"

COUNT = 5               # For printint out top 5 features for correlation or covariance

# Function to calculate correlation between data elements
# Input - dataset : Dataframe object
# Output - prints out correlations between all variables in Dataframe
# and between varibles of interest and returns the correlations
def get_correlation(data):
    
    # Calculate the correlation in the data
    # take the absolute value since large negative are as useful as large positive
    corr = data.corr().abs()
    
    # set the correlations on the diagonal or lower triangle to zero,
    # We clear the diagonal since the correlation with itself is always 1. And upper and lower matrix elements are repetitive.
    corr *= np.tri(*corr.values.shape, k=-1).T
    
    #unstack the corr object so we can sort things
    corr_unstack = corr.unstack()
    
    #Sort the unstacked corr in descending order
    corr_unstack.sort_values(inplace=True,ascending=False)

    #Print the top 5 values
    print(PRINT_TOP_5_ALL_VARIABLE_CORRELATIONS,corr_unstack.head(COUNT))
    print("")
    
    #filter correltations with our dependent variable
    with_type = corr_unstack.get(key = CLASS_DEPENDENT_VARIABLE)
    
    #Print the top 5 correlations with our dependent variable
    print(PRINT_TOP_5_DEPENDENT_VARIABLE_CORRELATIONS, with_type.head(COUNT))
    print("")
    #Return correlation dataframe (incase needed for pairplot)
    return corr


# Function to calculate covariance between data elements
# Input - dataset : Dataframe object
# Output - prints out covariances between all variables in Dataframe 
# and between varibles of interest and returns the covariance
def get_covariance(data):
    
    #Calculate the covariance
    # take the absolute value since large negative are as useful as large positive
    cov = data.cov().abs()
    
    # set the correlations on the diagonal or lower triangle to zero,
    # We clear the diagonal since the correlation with itself is always 1. And upper and lower matrix elements are repetitive.
    cov *= np.tri(*cov.values.shape, k=-1).T
    
    #unstack the corr object so we can sort things
    cov_unstack = cov.unstack()
    
    #Sort the unstacked cov in descending order
    cov_unstack.sort_values(inplace=True,ascending=False)
    
    
    #Print top 5 covariance
    print(PRINT_TOP_5_ALL_VARIABLE_COVARIANCE,cov_unstack.head(COUNT))
    print("")
    
    #filter covariance with our dependent variable
    with_type_cov = cov_unstack.get(key = CLASS_DEPENDENT_VARIABLE)
    
    #Print top 5 covariance with our variable of interest
    print(PRINT_TOP_5_DEPENDENT_VARIABLE_COVARIANCE,with_type_cov.head(COUNT))
    print("")
    
    #Return covariance dataframe (incase needed for pairplot)
    return cov


# Function to create the pair plot
# Input - dataset : Dataframe object
# Output - prints out covariances between all variables in Dataframe 
def create_pair_plot(dataset):
    #Create a pair plot
    sns.set(style = 'whitegrid', context = 'notebook')   # set the apearance
    sns.pairplot(dataset, height = 1.5, plot_kws={"s": 1})                    # create the pair plots
    plt.show()                                           # and show them

# Main function to read the dataset and call all helper functions    
def main():
    heart_data = read_csv(DATASET_INPUT_FILE_NAME)           # Read the sample data in as a dataframe
    correlation = get_correlation(heart_data)                # Call get_correlation to print correlations
    # create_pair_plot(correlation)                             # Uncomment to plot the pair plot for correlation
    covariance = get_covariance(heart_data)                  # Call get_covariance to print covariance and store it in a temp variable
    # create_pair_plot(covariance)                             # Uncomment to plot the pair plot for covariance
    create_pair_plot(heart_data)                             # Call the create_pair_plot function to get the pair plot for data

# This is the starting point of the program - invoking the main function
main()