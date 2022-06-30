# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday October 16 13:36:53 2021                                 #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Project2:                                                                    #
################################################################################

import numpy as np                                     # needed for arrays
import pandas as pd                                    # data frame
import matplotlib.pyplot as plt                        # modifying plot
from sklearn.model_selection import train_test_split   # splitting data
from sklearn.preprocessing import StandardScaler       # scaling data
from sklearn.neural_network import MLPClassifier       # learning algorithm for neural networks
from sklearn.decomposition import PCA                  # PCA package
from sklearn.metrics import accuracy_score             # grading
from sklearn.metrics import confusion_matrix           # generate the  matrix
from warnings import filterwarnings                    # To ignore convergence warnings


MAX_ITERATIONS = 2000                                   # Max iterations for MLPClassifier
RANDOM_STATE = False                                    # Variable to determine whether to use random seed or not
TEST_SIZE_SPLIT = 0.3                                   # Split percentage for dataset
TOTAL_FEATURES = 60                                     # Total features in the dataset
CLASS_INDEX = -2                                        # Identifier for class in the dataset (second last column)
HIDDEN_LAYER_SIZES = 100                                # Hiddent layer size for MLPClassifier
ALPHA = 0.00001                                         # Const alpha for MLPClassifier
TOL = 0.0001                                            # Const tol for MLPClassifier

SONAR_DATASET = "sonar_all_data_2.csv"                  # Sonar dataset 

filterwarnings('ignore')              # Invoke filterwarnings to ignore the convergence warnings for low iterations


# Function to run neural network analysis on a dataset
# Input - standardized test and train data, testing set train and test data
# components to use for PCA, max_iterations for MLPClassifier and random_state to determine the random seed variable
# Output - Returns the test accuracy and the confusion matrix for the component_count that was passed
def run_neural_analysis(x_train_std, x_test_std, y_train, y_test,
                        component_count, max_iterations, random_state):
    
    random_state_int = None                                    # Initialize random state to None, meaning it will use random seed
    if (random_state == False):                                # if random_state is false then set random seed to 0
        random_state_int = 0
    
    pca = PCA(n_components=component_count)                    # only keep the features passed in from component_count variable
    x_train_pca = pca.fit_transform(x_train_std)               # apply to the train data
    x_test_pca = pca.transform(x_test_std)                     # do the same to the test data
    
    # now create a MLPClassifier and train on it
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(HIDDEN_LAYER_SIZES), activation='logistic', max_iter=max_iterations, 
                       random_state=random_state_int, alpha=ALPHA, solver='adam', tol=TOL )
    
    mlp_classifier.fit(x_train_pca,y_train)                    # Fit the data in the classifier
    
    y_pred = mlp_classifier.predict(x_test_pca)                # Do the predictions
    
    test_accuracy = accuracy_score(y_test, y_pred)             # collect the accuracy score between the test set
    
    # now combine the train and test data for creating the confusion matrix
    x_comb_pca = np.vstack((x_train_pca, x_test_pca))
    y_comb = np.hstack((y_train, y_test))
    
    y_comb_pred = mlp_classifier.predict(x_comb_pca)           # Do the prediction on the combined dataset
    
    combined_accuracy = accuracy_score(y_comb, y_comb_pred)    # Calculate the combined accuracy for plotting purposes
    
    confuse = confusion_matrix(y_comb,y_comb_pred)             # Create a confusion matrix variable with confusion matrix
    
    return test_accuracy, confuse, combined_accuracy           # Return the test and combined accuracy for the components and the confusion matrix
    

# Function to create the plot for the component and accuracies arrays
# Input - component array and accuracy array
# Output - void, plots the graph for component vs accuracy
def draw_component_accuracy_plot(component_array, accuracy_array, combined_accuracy_array):
    plt.plot(component_array, accuracy_array, label="test accuracy")                   # data to plot, component vs accuracy
    plt.plot(component_array, combined_accuracy_array, label="combined accuracy")
    plt.xlabel("Number of components")                          # x-axis label
    plt.ylabel("Accuracy achieved")                             # y-axis label
    plt.title('Accuracy achieved for # of components in MLP')   # a title for the plot
    plt.grid(False)
    plt.legend()
    plt.show()                                                  # Show the plot
    

# Main function to read the dataset, invoke the analysis function and finally plot the result
# Input - None
# Output - Prints and plots the required attributes
def main():        
    
    # read the database. Since it lacks headers, put them in
    df_wine = pd.read_csv(SONAR_DATASET, header=None)
        
    x = df_wine.iloc[:,:CLASS_INDEX].values       # features are in columns 0:(N-CLASS_INDEX)
    y = df_wine.iloc[:,CLASS_INDEX].values        # classes are in column CLASS_INDEX!
    
    # now split the data
    x_train, x_test, y_train, y_test = \
             train_test_split(x, y, test_size=TEST_SIZE_SPLIT, random_state=0)
    
    stdsc = StandardScaler()                     # Create a standard scalar
    x_train_std = stdsc.fit_transform(x_train)   # Apply the standardization on train data
    x_test_std = stdsc.transform(x_test)         # Apply the standardization on test data
    
    max_accuracy_component = 1                   # Temporary variable to store the compnent count with max accuracy, intialized to 1 component
    max_accuracy = 0                   # Temporary variable to store the max accuracy
    test_accuracies = []                   # Temporary list to store the accuracies for plotting later
    combined_accuracies = []                # Temporary list to store combined accuracies
    components = []                   # Temporary list to store the compnent count for plotting later
    confuse_for_max = []                   # Temporary variable to store the confusion matrix for max accuarcy components

    for component_count in range(1,TOTAL_FEATURES + 1):                   # Iterate over the total number of features starting from 1
        
        # Invoke the analysis function to get the accuracy and confusion matrix for the compnent count
        test_accuracy, confuse, combined_accuracy = run_neural_analysis(x_train_std, x_test_std, y_train, y_test,
                        component_count, MAX_ITERATIONS, RANDOM_STATE)
        
        # Print the values during iteration
        print('Number of components ',component_count, ', Accuracy achieved: %.2f' % test_accuracy )

        # Collect the data for accuracy and components in the list
        test_accuracies.append(test_accuracy)
        components.append(component_count)
        combined_accuracies.append(combined_accuracy)
        
        # Updating the maximum 
        if (test_accuracy > max_accuracy):              # If test_accuarcy is greater than max accuracy, we update our temp variables
            max_accuracy = test_accuracy                # update the accuracy
            max_accuracy_component = component_count    # update the compnoent count
            confuse_for_max = confuse                   # update the confusion matrix
    
    # Print final details for the component count that provided maximum accuracy
    print("")
    print("--------------------------------------------------")
    print("")
    print('Number of components for Max accuracy ',max_accuracy_component, ', Max Accuracy achieved: %.2f' % max_accuracy )
    print('Confusion matrix for max accuracy:')
    print(confuse_for_max)
    
    # Invoke the plotting function
    draw_component_accuracy_plot(components, test_accuracies, combined_accuracies)


# This is the starting point of the program, invoke the main function
main()