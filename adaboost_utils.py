#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def read_data(file_name):
    '''
    This function reads the data from input file.
    DO NOT MODIFY this function.

    Inputs:
        - The input file name

    Output:
        - The input features (nx2 array)
        - The true labels (nx1 array)
    '''
    X, y = [], []
    with open(file_name, "r") as f:
        for line in f:
            num_list = list(map(float, line[:-1].split(",")))
            X.append(num_list[:2])
            y.append([num_list[-1]])

    return np.array(X), np.array(y)

def weak_classifier(X,Y,D):    
    '''
    This function returns the best weak classifier.

    HERE YOU HAVE TO USE DECISION STUMP AS WEAK CLASSIFIER (OR LEARNER) FOR ADABOOST
    
    NOTE: The typical way of training a (1-level) Decision Tree is 
    finding such an attribute that gives the purest split. For this problem you
    are working with 2 attributes (features x={x1,x2} see the plot in the handout).
    The purpose of finding the purest split is to split the dataset into two 
    subsets and we want the labels inside these subsets to be as homogeneous as possible. 
    This procedure can also be seen as building many trees - a tree for each attribute -
    and then selecting the tree that produces the best split.

    Inputs:
        - The input features (nx2 array)
        - The true labels (nx1 array)
        - Current distribution D (nx1 array)
 
    Output:
        - The best weak classifier h
            > If we find the best split based on the error, 
            > then return the best weak classifier, i.e. return:
                  - best feature index
                  - best split value
                  - label when the feature value is equal to or greater than 
                  the best split value
            > as h = [best feature index,best split value, label]
        - The value of alpha_t
        - The predicted labels by the best weak classifier (nx1 array)
    '''
      
    eps_best = 1

    n = len(Y)      
    ypred_best = np.zeros((n,1))
    
    h_feature = 0
    h_splitval = 0
    h_label = 0

    best_error = 10
    for feat_idx in range(X.shape[1]):
        for insta_idx in range(X.shape[0]):
            for label in np.array([-1, 1]):
                
    ####################################################
    #YOUR CODE HERE
    ####################################################
                left = np.arange(0,insta_idx)
                right = np.arange(insta_idx,X.shape[0])
                x_left = X[left,:]
                x_right = X[right,:]
                y_left = Y[left,:]
                y_right = Y[right,:]
                d_left = D[left,:]
                d_right = D[right,:]
                
                left_error = (y_left != label).T @ d_left
                right_error = (y_right != -1 * label).T @ d_right
                error = left_error+right_error
                if error < best_error:
                    best_error = error
                    h_feature = feat_idx
                    h_splitval = insta_idx
                    h_label = label
                    ypred_best = np.vstack((y_left,y_right))
                    alpha = 0.5*np.log((1-best_error)/best_error)
                    print(best_error,)

                
    #################################################### 
    #--------------------------------------------------#
    ####################################################
    
    
    h = [h_feature,h_splitval,h_label]        
    return h, alpha, ypred_best



def update_weight(D, alpha, y, ypred):    
    '''
    This function computes the updated distribution D_{t+1}(i). 
    
    Inputs:
        - Current distribution D (nx1 array)
        - The value of alpha_t
        - The true target values (nx1 array)
        - The predicted target values (nx1 array)
    
    Output:
        - The updated distribution D
    '''
    
    #YOUR CODE HERE:
    for i in len(y):
        D[i] = D[i]*np.exp(-alpha*y[i]*ypred[i])
    D = D/np.sum(D)
    
    return D
    
    
    
def adaboost_pred(X, h, alpha):
    '''
    This function returns the predicted labels for each weak classifier
    according to its importance (given by alpha_t for t = 1...T).
    
    Inputs:
        - The input feature (nx2 array)
        - The array of weak classifiers (Tx3 array)
        - The array of alpha_t for t = 1..T (Tx1 array)
 
    Output:
        - The predicted labels (nx1 array)
    '''
        
    n = len(X)
    t = len(h)
    ypred = np.zeros((n, 1))
    
    for i in range(t):
        #YOUR CODE HERE:
        break

    # Normalize the output according to the labels
    ypred = (ypred >= 0) * 2 - 1  

    return ypred










