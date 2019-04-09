#%%

from gmm_utils import prepare_data_pca, gmm, rnd_init, plot_ellipse, compute_log_likelihood
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



####################################################
#%% Question 1
# Complete the "prepare_data_pca" function
####################################################

X,y=prepare_data_pca()
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Principal components of gaussians')
sns.scatterplot(X[:,0],X[:,1],hue = np.squeeze(y))
plt.savefig('q4p1.png')
plt.show()


####################################################
#%% Question 2 and 3
####################################################

# YOUR CODE HERE:
w = np.array([0.5,0.5]).T
mu = np.array([[-3,0],[5,0]])
s1 = np.array([[1,0],[0,1]]).reshape(1,2,2)
sigma = np.concatenate((s1,s1),axis=0)

mll = compute_log_likelihood(X)
print('initial mean log-likelihood =  {}'.format(mll)) # Print the mean log-likelihood
mu10,sigma10,w10,iters10,mll_list10 = gmm(X,mu,sigma,w,10)
print('\nConverges after {} steps'.format(iters10))
print('\nAt iteration 10:')
print('mean log-likelihood = {}'.format(mll_list10[-1]))
print('mu = \n{}'.format(mu10)) # Print mu at iteration 10
print('sigma = \n{}'.format(sigma10)) # Print sigma at iteration 10
print('w = \n{}'.format(w10)) # Print w at iteration 10



################################k####################
#%% Question 4
####################################################

# YOUR CODE HERE:
plt.figure()
for i in range(10):
    mu,sigma,w = rnd_init(X,2)
    mu,sigma,w,iters,mll_list = gmm(X,mu,sigma,w,stopping_condition="convergence")
    plt.plot(mll_list)
plt.xlabel('iters')
plt.ylabel('mean log-likelihood')
plt.title('Different random initializations')
plt.savefig('q4p4.png')
plt.show()



####################################################
#%% Question 5
####################################################

for k in range(1,6):
    # YOUR CODE HERE:
    plt.figure()
    mu,sigma,w = rnd_init(X,k)
    mu,sigma,w,iters,mll_list = gmm(X,mu,sigma,w,stopping_condition="convergence")
    plot_ellipse(mu,sigma)
    sns.scatterplot(X[:,0],X[:,1],hue=np.squeeze(y))
    plt.title('GMM solution with k = {}'.format(k))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('q4p5_k{}.png'.format(k))
    plt.show()
 
    


####################################################
#%% Question 6
####################################################

# YOUR CODE HERE:

plt.figure()
mlls = []
for k in range(1,11):
        mll_k = []
        for i in range(10):
                mu,sigma,w = rnd_init(X,k)
                mu,sigma,w,iters,mll_list = gmm(X,mu,sigma,w,stopping_condition="convergence")
                mll_k.append(mll_list[-1])  
        mean_mll = np.mean(np.array(mll_k))
        mlls.append(mean_mll)
plt.plot(mlls)
plt.xlabel('Number of Gaussian Components k')
plt.ylabel('Mean log-likelihood')
plt.savefig('q4p6.png')
plt.show()



#%%
