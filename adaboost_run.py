#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from adaboost_utils import read_data, weak_classifier, update_weight, adaboost_pred
import numpy as np


def eval_model(X_test, y_test, hlist, alphalist):
    '''
    This function evaluates the model with test data by measuring the accuracy.
    
    Inputs:
        - The input features for test data (mx2 array)
        - The true labels for test data (mx1 array)
        - The array of weak classifiers (Tx3 array)
        - The array of alpha_t for t = 1..T (Tx1 array)
    
    Output:
        - The predicted labels (nx1 array)
        - The accuracy
    '''

    #YOUR CODE HERE:
    pred = np.zeros((len(X_test),1))
    acc = 0

    return pred, acc


def train(num_iter, X_train, y_train, X_test, y_test):
    '''
    This function trains the model by using AdaBoost algorithm.
    
    Inputs:
        - The number of iterations (T)
        - The input features for training data (nx2 array)
        - The true labels for training data (nx1 array)
        - The input features for test data (mx2 array)
        - The true labels for test data (mx1 array)
    
    Output:
        - The array of weak classifiers (Tx3 array)
        - The array of alpha_t for t = 1..T (Tx1 array)
    '''
    
    hlist = np.zeros((num_iter, 3))
    alphalist = np.zeros((num_iter, 1))

    # To ease your debugging
    train_acc = np.zeros((num_iter, 1))
    test_acc = np.zeros((num_iter, 1))

    ####################################################
    #YOUR CODE HERE
    ####################################################
    N = X_train.shape[0]
    D = 1/N*np.ones((N,1))
    for i in range(num_iter):
        h,alpha,ypred_best = weak_classifier(X_train,y_train,D)
        hlist[ i , : ] = h 
        alphalist[i,:] = alpha
        D = update_weight(D,alpha,y_train,ypred_best)



    #################################################### 
    #--------------------------------------------------#
    ####################################################
    
    return hlist, alphalist
    


def main():
    num_iter = 400
    num_iter = 1

    X_train, y_train = read_data("train_adaboost.csv")
    X_test, y_test = read_data("test_adaboost.csv")

    hlist, alphalist = train(num_iter, X_train, y_train, X_test, y_test)
    final_pred, final_acc = eval_model(X_test, y_test, hlist, alphalist)


if __name__ == "__main__":
    main()
