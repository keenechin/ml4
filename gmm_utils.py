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
    X = np.vstack((score[:,0],score[:,1])).T
    y = np.hstack((np.zeros(test0.shape[0]),np.ones(test1.shape[0]))).T
    sns.scatterplot(X[:,0],X[:,1],hue = np.squeeze(y))
    plt.show()
    #######################################################################
    return X, y
    



def gmm(X,mu,sigma,w,iteration=100,stopping_condition=""):
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
    mll_list = []
    last_mll = 0
    convergence_steps = iteration
    for i in range(iteration):
        mll,Ls = expectation_step(X,mu,sigma,w)
        if np.abs(mll-last_mll) < 1e-5:
            convergence_steps = np.min([i,convergence_steps])
            if stopping_condition == 'convergence':
                break
        mu,sigma,w = maximization_step(X,Ls)
        mll_list.append(mll)
        last_mll = mll

    return mu,sigma,w,convergence_steps,mll_list

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
    ############
    ###########################################################
    N,p = X.shape
    k = len(w)
    likelihood = np.zeros((N,k))
    

    #maybe each row of likelhood should sum to 1
    for j in range(k):
        likelihood[:,j] = w[j]*multivariate_normal.pdf(X,mean = mu[j], cov = sigma[j])
    
    l_sum = np.sum(likelihood,axis = 1)
    ll = np.log(l_sum)
    mll = 1/N * np.sum(ll)
    Ls = (likelihood.T/l_sum).T

    return mll, Ls
    
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
    #SHAPES ARE WRONG
    k = Ls.shape[1]
    N,p = X.shape
    den = np.sum(Ls,axis=0) 
    mu = np.nan*np.ones((k,p))
    w = np.nan*np.ones((k,1)) 
    sigma = np.nan*np.ones((k,p,p))
    
    for j in range(k):
        gamma_nk = np.vstack((Ls[:,j],Ls[:,j])).T
        mu_num = np.sum(gamma_nk*X,axis=0)
        mu[j,:] = mu_num/den[j]
        sig_num = ((gamma_nk*(X-mu[j,:])).T@(X-mu[j,:]))
        sigma[j,:,:] = sig_num/den[j]
        w[j,:] = np.sum(Ls[:,j],axis=0)/len(Ls)
    return mu,sigma,w

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

def compute_log_likelihood(X):
    w = np.array([0.5,0.5]).T
    mu = np.array([[-3,0],[5,0]])
    s1 = np.array([[1,0],[0,1]]).reshape(1,2,2)
    sigma = np.concatenate((s1,s1),axis=0)
    mll,Ls = expectation_step(X,mu,sigma,w)
    return mll











