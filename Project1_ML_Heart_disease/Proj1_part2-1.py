# -*- coding: utf-8 -*-
################################################################################
# Created on Thursday October 14 13:36:53 2021                                 #
#                                                                              #
# @author: Isha Joshi                                                          #
#                                                                              #
# Project1 - part2: Apply various ML algorithms to predict heart disease       #
################################################################################


from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.tree import DecisionTreeClassifier        # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.svm import SVC                            # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm
from sklearn.metrics import accuracy_score             # grade the results

import numpy as np                                     # Numpy functions
from pandas import read_csv                            # Reading the dataset using pandas

TEST_DATA_PERCENTAGE = 0.3                             # Threshold for splitting the dataset into training and testing


# Best parameter constants for all algorithms
PERCEPTRON_MAX_ITERATIONS = 10      # Perceptron algo
LOGISTIC_REGRESSION_C = 0.16      # Logistic regression algo
SVM_C = 0.23      # SVM algo
DECISION_TREE_DEPTH = 6      # Decision tree algo
RANDROM_FOREST_TREES = 69      # Random forest algo
KNN_K = 35          # KNN algo

# Function to implement perceptron analysis
# Input - Standardized train/test sets and class train test set and max iterations for algo
# Output - Prints out dataset accuracy and combined accuracy
def perceptron(x_train_std, x_test_std, y_train, y_test, max_iterations):
    
    # Create the perceptron analyzer
    # We do not want to enable debug logs so verbose = False
    ppn = Perceptron(max_iter=max_iterations, tol=1e-3, eta0=0.001,
                     fit_intercept=True, random_state=0, verbose=False)

    ppn.fit(x_train_std, y_train)  # do the training using scaled and class test data
    
    y_pred = ppn.predict(x_test_std)  # now try with the test data
    
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    X_combined_std = np.vstack((x_train_std, x_test_std))  # doing combo
    y_combined = np.hstack((y_train, y_test))
    
    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = ppn.predict(X_combined_std)
    print('Misclassified combined samples: %d' % \
          (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.4f' % accuracy_score(y_combined, y_combined_pred))


# Function to implement logistic regression analysis
# Input - Standardized train/test sets and class train test set and c_val for algo
# Output - Prints out dataset accuracy and combined accuracy
def logistic_regression(x_train_std, x_test_std, y_train, y_test, c_val):   
        
    # apply the algorithm to training data
    lr = LogisticRegression(C=c_val, solver='liblinear', \
                                multi_class='ovr', random_state=0)
    lr.fit(x_train_std, y_train)  # do the training
    
    y_pred = lr.predict(x_test_std)  # now try with the test data
    
    # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    
    X_combined_std = np.vstack((x_train_std, x_test_std))  # doing combo
    y_combined = np.hstack((y_train, y_test))
    
    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = lr.predict(X_combined_std)
    print('Misclassified combined samples: %d' % \
                                    (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.4f' % accuracy_score(y_combined, y_combined_pred))
    

# Function to implement SVM analysis
# Input - Standardized train/test sets and class train test set and c_val for algo
# Output - Prints out dataset accuracy and combined accuracy
def support_vector_machine(x_train_std, x_test_std, y_train, y_test, c_val): 

    # apply the algorithm to training data
    svm = SVC(kernel='linear', C=c_val, random_state=0)
    svm.fit(x_train_std, y_train)  # do the training
    
    y_pred = svm.predict(x_test_std)  # work on the test data
    
    # show the results
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.4f' % accuracy_score(y_test, y_pred))
    
    # combine the train and test sets
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # and analyze the combined sets
    y_combined_pred = svm.predict(X_combined_std)
    print('Misclassified combined samples: %d' % \
          (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.4f' % \
          accuracy_score(y_combined, y_combined_pred))


# Function to implement decision tree analysis
# Input - Standardized train/test sets and class train test set and depth for algo
# Output - Prints out dataset accuracy and combined accuracy
def decision_tree_learning(x_train, x_test, y_train, y_test, depth):    
    
    # apply the algorithm to training data
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=depth ,random_state=0)
    tree.fit(x_train,y_train)   # do the training
    
    
    y_pred = tree.predict(x_test)  # now try with the test data
    
    
     # Note that this only counts the samples where the predicted value was wrong
    print('Misclassified samples: %d' % (y_test != y_pred).sum())  # how'd we do?
    print('Accuracy: %.7f' % accuracy_score(y_test, y_pred))
    
    # vstack puts first array above the second in a vertical stack
    # hstack puts first array to left of the second in a horizontal stack
    # NOTE the double parens!
    x_combined = np.vstack((x_train, x_test))  # doing combo
    y_combined = np.hstack((y_train, y_test))
    
    # we did the stack so we can see how the combination of test and train data did
    y_combined_pred = tree.predict(x_combined)
    print('Misclassified combined samples: %d' % \
                                        (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.7f' % accuracy_score(y_combined, y_combined_pred))
    


# Function to implement random forest analysis
# Input - Standardized train/test sets and class train test set and trees for algo
# Output - Prints out dataset accuracy and combined accuracy
def random_forest(x_train, x_test, y_train, y_test, trees):
    
    # apply the algorithm to training data
    forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                                    random_state=1, n_jobs=4)
    forest.fit(x_train, y_train)   # do the training
    
    y_pred = forest.predict(x_test)  # see how we do on the test data
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    
    print('Accuracy: %.7f \n' % accuracy_score(y_test, y_pred))
    
        # combine the train and test data
    X_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    
        # see how we do on the combined data
    y_combined_pred = forest.predict(X_combined)
    print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.7f' % \
                accuracy_score(y_combined, y_combined_pred))
    


# Function to implement perceptron analysis
# Input - Standardized train/test sets and class train test set and neights for neibors for algo
# Output - Prints out dataset accuracy and combined accuracy
def k_nearest_neighbors(x_train_std, x_test_std, y_train, y_test, neightbors):

    # apply the algorithm to training data
    knn = KNeighborsClassifier(n_neighbors=neightbors,p=2,metric='minkowski')
    knn.fit(x_train_std,y_train) # do the training
    
        # run on the test data and print results and check accuracy
    y_pred = knn.predict(x_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.7f ' % accuracy_score(y_test, y_pred))
    
        # combine the train and test data
    X_combined_std = np.vstack((x_train_std, x_test_std))
    y_combined = np.hstack((y_train, y_test))
    
        # check results on combined data
    y_combined_pred = knn.predict(X_combined_std)
    print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.7f' % \
               accuracy_score(y_combined, y_combined_pred))
    


def main():
        
    # load the data set
    heart_db = read_csv('heart1.csv')
    
    # We want to use all the features
    x = heart_db[['age','sex','cpt','rbp','sc','fbs','rer','mhr',\
       'eia',	'opst',	'dests'	,'nmvcf',	'thal']]
    
    # The class we care about is a1p2 for heart disease 
    # We extract all values for that
    y = heart_db.loc[:,'a1p2']
    
    # split the problem into train and test
    # this will yield 70% training and 30% test
    # random_state allows the split to be reproduced
    x_train, x_test, y_train, y_test = \
             train_test_split(x,y,test_size = TEST_DATA_PERCENTAGE,random_state=0)
    
    # scale X by removing the mean and setting the variance to 1 on all features.
    sc = StandardScaler()                    # create the standard scalar
    sc.fit(x_train)                          # compute the required transformation
    x_train_std = sc.transform(x_train)      # apply to the training data
    x_test_std = sc.transform(x_test)        # and SAME transformation of test data
    
    
    # for i in range(1, 50):
    #     print("i = ", i)
    #     k_nearest_neighbors(x_train_std, x_test_std, y_train, y_test, i)   # For getting the best params
    #     print()
    #     print()
    #     print()
        
    print('********PERCEPTRON***********\n')
    perceptron(x_train_std, x_test_std, y_train, y_test, PERCEPTRON_MAX_ITERATIONS)        # Call perceptron analysis with scaled data, test data and max_iterations
    
    print('*******************\n')
    print('*********LOGISTIC_REGRESSION**********\n')
    logistic_regression(x_train_std, x_test_std, y_train, y_test, LOGISTIC_REGRESSION_C)   # Call logistic_regression analysis with scaled data, test data and max_iterations
    
    print('*******************\n')
    print('*********SVM**********\n')
    support_vector_machine(x_train_std, x_test_std, y_train, y_test, SVM_C)  # Call support_vector_machine analysis with scaled data, test data and max_iterations
    
    print('*******************\n')
    print('*******DECISION_TREE_LEARNING************\n')
    decision_tree_learning(x_train, x_test, y_train, y_test, DECISION_TREE_DEPTH)     # Call decision_tree_learning analysis with scaled data, test data and max_iterations
    
    print('*******************\n')
    print('********RANDOM_FOREST***********\n')
    random_forest(x_train, x_test, y_train, y_test, RANDROM_FOREST_TREES)    # Call random_forest analysis with scaled data, test data and max_iterations
    
    print('*******************\n')
    print('********KNN***********\n')
    k_nearest_neighbors(x_train_std, x_test_std, y_train, y_test, KNN_K)     # Call k_nearest_neighbors analysis with scaled data, test data and max_iterations
    

main()