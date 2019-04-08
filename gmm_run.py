from gmm_utils import prepare_data_pca, gmm, rnd_init, plot_ellipse, compute_log_likelihood
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



####################################################
# Question 1
# Complete the "prepare_data_pca" function
####################################################

X,y=prepare_data_pca()



####################################################
# Question 2 and 3
####################################################

# YOUR CODE HERE:
w = np.array([0.5,0.5]).T
mu = np.array([[-3,0],[5,0]])
s1 = np.array([[1,0],[0,1]]).reshape(1,2,2)
sigma = np.concatenate((s1,s1),axis=0)

data = np.vstack((X,y)).T
mll = compute_log_likelihood(data)
print('mean log-likehood: {}'.format(mll)) # Print the mean log-likelihood
mu10,sigma10,w10 = gmm(data,mu,sigma,w,10)
print(mu10) # Print mu at iteration 10
print(sigma10) # Print sigma at iteration 10
print(w10) # Print w at iteration 10



################################k####################
# Question 4
####################################################

# YOUR CODE HERE:

plt.figure()



####################################################
# Question 5
####################################################

for k in range(1,6):
    # YOUR CODE HERE:

    plt.figure()
    


####################################################
# Question 6
####################################################

# YOUR CODE HERE:

plt.figure()


