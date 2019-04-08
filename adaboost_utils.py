#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt

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

    eps_best = 10
    for feat_idx in range(X.shape[1]):
        for insta_idx in range(X.shape[0]):
            for label in np.array([-1, 1]):
                
    ####################################################
    #YOUR CODE HERE
    ####################################################
                #left = np.arange(0,insta_idx)
                #right = np.arange(insta_idx,X.shape[0])
                left = X[:,feat_idx] <= X[insta_idx,feat_idx]
                right = X[:,feat_idx] > X[insta_idx,feat_idx]
                y_left = Y[left,:]
                y_right = Y[right,:]
                d_left = D[left,:]
                d_right = D[right,:]
                
                left_error = (y_left != label).T @ d_left
                right_error = (y_right != -label).T @ d_right
                error = left_error+right_error
                if error < eps_best:
                    eps_best = error
                    h_feature = feat_idx
                    h_splitval = X[insta_idx,feat_idx]
                    h_label = label
                    #ypred_left = label*np.ones(y_left.shape)
                    #ypred_right = -label*np.ones(y_right.shape)
                    #ypred_best = np.vstack((ypred_left,ypred_right))
                    ypred_best = np.zeros(Y.shape)
                    ypred_best[left,:]=label
                    ypred_best[right,:]=-label

                    alpha = 0.5*np.log((1-eps_best)/eps_best)

                
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
    for i in range(len(y)):
        D[i] = D[i]*np.exp(-alpha*y[i]*ypred[i])
    D = D/np.sum(D)
    
    return D
"""    
def visualize_data(X,y,title=None):
    plt.scatter(X[:,0],X[:,1],c=y.ravel(),cmap='plasma')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title(title)
    plt.show()
"""
   
    
def adaboost_pred(X, h, alpha,title=None):
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
    T = len(h)
    ypred = np.zeros((n, 1))
    
    for t in range(T):
        #YOUR CODE HERE:
        feature,splitval,label = h[t,:]
        left = X[:,int(feature)] <= splitval
        right = X[:,int(feature)] > splitval
        ypred[left] += alpha[t] * label
        ypred[right] += alpha[t] * -label
    


    # Normalize the output according to the labels
    ypred = (ypred >= 0) * 2 - 1  
    #visualize_data(X,ypred,title)
    
    return ypred










