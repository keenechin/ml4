'''
Install these packages if necessary:
    Seaborn: https://seaborn.pydata.org/installing.html
    Scikit-learn: https://scikit-learn.org/stable/install.html
    Scipy: (sudo pip install scipy) https://pypi.org/project/scipy/
    Matplotlib: https://matplotlib.org/users/installing.html
    
If you have installed Anaconda before, some of the listed packages might be 
already installed in your system
'''


import scipy.io
import numpy as np

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal


def prepare_data_pca():
    '''
    This function prepares and plots the data with the first two principal components. 
    
    Output:
        - The first 2 principal components (PCs) of input data (2115x2 array, X)
        - The labels for input data (2115x1 array, y)
    '''
    
    mat = scipy.io.loadmat('mnist_all.mat') # Load the MNIST data
    
    #######################################################################
    # YOUR CODE HERE:
    # Use test0 and test1 arrays in "mnist_all.mat" to construct X0 
    # (raw input data divided by 255) and the corresponding labels y 
    # (0 for test0, 1 for test1).
    #######################################################################
    test0 = mat['test0']/255
    test1 = mat['test1']/255
    X0 = np.vstack((test0,test1))
    
    #######################################################################

    # Compute PCA:
    # Read: http://tiny.cc/y2m13y
    
    pca = PCA(n_components=2)
    pca.fit(X0)    
    score = pca.transform(X0)
    
    #######################################################################
    # YOUR CODE HERE:
    # Plot the first 2 PCs (you can use sns.scatterplot) and return them 
    #######################################################################
    X = score[:,0]
    y = score[:,1]
    sns.scatterplot(X,y)
    plt.show()
    #######################################################################
    return X, y
    



def gmm(X,mu,sigma,w,iteration=100):
    '''
    This function trains GMM using EM algorithm.
    
    Inputs:
        - The input features after PCA (2115x2 array)
        - The mean of Gaussian mixture components (kx2 array)
        - The covariance matrices of Gaussian mixture components (kx2x2 array)
        - The mixture component weights (kx1 array)
    
    Suggested Outputs:
        - The updated mean of Gaussian mixture components (kx2 array)
        - The updated covariance matrices of Gaussian mixture components (kx2x2 array)
        - The updated mixture component weights (kx1 array)
        - The iteration at which we observe convergence
        - The mean log-lieklihood
    '''
    
    #######################################################################
    # YOUR CODE HERE:
    #######################################################################


    return None

    #######################################################################
    



def expectation_step(X,mu,sigma,w):
    '''
    This function computes the mean log-likelihood.
    
    Inputs:
        - The input features after PCA (2115x2 array)
        - The mean of Gaussian mixture components (kx2 array)
        - The covariance matrices of Gaussian mixture components (kx2x2 array)
        - The mixture component weights (kx1 array)
    
    Suggested Outputs:
        - The mean log-lieklihood
        - The data’s class membership distribution (2115xk array, responsibility)

    Note:
        - You can use scipy.stats: multivariate_normal
    '''
    
    #######################################################################
    # YOUR CODE HERE:
    #######################################################################


    return None
    
    #######################################################################



def maximization_step(X,Ls):
    '''
    This function computes the parameters maximizing MLE given 
    our data’s class membership distribution (responsibility).
    
    Inputs:
        - The input features after PCA (2115x2 array)
        - The data’s class membership distribution (2115xk array, responsibility)
    
    Suggested Outputs:
        - The updated mean of Gaussian mixture components (kx2 array)
        - The updated covariance matrices of Gaussian mixture components (kx2x2 array)
        - The updated mixture component weights (kx1 array)
    '''

    #######################################################################
    # YOUR CODE HERE:
    #######################################################################

    return None

    #######################################################################


def rnd_init(X,k):
    '''
    This function initializes mu randomly by choosing k points 
    from the input samples. It also initialize all covariance matrices to the 
    identity matrices, and all mixture component weights to 1/k.
    
    Inputs:
        - The input features after PCA (2115x2 array)
        - The number of mixture components (k)
    
    Outputs:
        - The initialized mean of Gaussian mixture components (kx2 array)
        - The initialized covariance matrices of Gaussian mixture components (kx2x2 array)
        - The initialized mixture component weights (kx1 array)
    '''
    
    N=X.shape[0]
    d=X.shape[1]
    rnd_idx = np.random.randint(1,N,k)
    mu=X[rnd_idx]
    sigma=np.zeros((k,d,d))
    w=np.ones((1,k))/k
    for i in range(k):
        sigma[i]=np.eye(d)
    
    return mu,sigma,w[0]



def plot_ellipse(mu,sigma):
    '''
    This function is provided for you to plot the ellipses curves.
    
    Inputs:
        - The mean of Gaussian mixture components (kx2 array)
        - The covariance matrices of Gaussian mixture components (kx2x2 array)    
    '''
        
    k = mu.shape[0]
    for i in range(k):
        A = np.linalg.cholesky(sigma[i,:,:])
        theta = np.linspace(0, 2*np.pi, 1000)
        x = np.expand_dims(np.transpose(mu[i,:]),1) + 2.5 * np.dot(A, np.array([np.cos(theta), np.sin(theta)]))
        plt.plot(x[0,:], x[1,:])
  
    return

def compute_log_likelihood():
    return None











