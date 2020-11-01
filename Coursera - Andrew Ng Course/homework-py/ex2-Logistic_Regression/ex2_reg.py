#!/usr/bin/env python
# coding: utf-8

# # Exercise 2
# Coursera - Machine Learning <br>
# Andrew Ng

# # Regularized Logistic Regression

# ## Visualizing the Data

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame as dframe


# In[2]:


data = np.loadtxt('ex2data2.txt', delimiter=',')


# In[803]:


X = data[:, :2];
y = data[:, 2]


# In[804]:


y.shape


# In[805]:


pos = np.argwhere(y==1)
neg = np.argwhere(y==0)


# In[806]:


pos.shape, neg.shape


# In[807]:


adm = X[pos, :2].reshape(-1, 2)
not_adm = X[neg, :2].reshape(-1, 2)


# In[809]:


plt.style.use('seaborn')
admit = plt.scatter(adm[:, 0], adm[:, 1], marker='+', color='k', linewidths=2)
not_admit = plt.scatter(not_adm[:, 0], not_adm[:, 1], marker='o', color='y', linewidths=3)
plt.legend((admit, not_admit), ["Microchip1 y = 1", "Microchip2 y = 0"])
plt.show()


# ## Feature Mapping 

# In[721]:


out = np.ones([X1.size, 1])
out.shape


# In[825]:


def mapFeature(X1, X2):
    degree = 6
    out = np.ones([X1.size, 1])
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp = (X1**(i-j)*(X2**j))[:, np.newaxis]
#             out(:, end+1) = (X1.^(i-j)).*(X2.^j);
            out = np.hstack([out, temp])
    
    return out


# In[826]:


X = mapFeature(X[:, 0], X[:, 1])


# In[827]:


# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
X.shape


# In[828]:


# Initialize fitting parameters
initial_theta = np.zeros([X.shape[1], 1])


# In[829]:


# Set regularization parameter lambda to 1
λ = 1


# In[675]:


# Sigmoid function
def sigmoid(z):
    sig = 1/(1+np.exp(-z))
    return sig


# In[676]:


def costFunctionReg(theta, X, y, λ):
    # Initialize some useful values
    m = X.shape[0]; # number of training examples

    # You need to return the following variables correctly 
    J = 0;
    grad = np.zeros([theta.size,1]);
    
    # Sigmoid
    z = X@theta
    g_z = sigmoid(z)
    
    # Cost function
    J = 1/m * np.sum([(-y * np.log(g_z))-(1-y)*(np.log(1-g_z))]) + (λ/(2*m) * np.sum(theta[1:]**2))
    
    # Grad descent
    grad = 1/m * X.T @ (g_z - y);
    grad[1:] = grad[1:] + (λ/m * theta[1:])
#     grad[0] = 1/m * X[:,0] @ (g_z - y);
#     grad[1:] = 1/m * X[:,1:].T @ (g_z - y) + (λ/m * theta[1:])
    
    return J, grad


# In[677]:


cost, grad = costFunctionReg(initial_theta, X, y, λ)


# In[678]:


print('Cost at initial theta (zeros): %f' %cost)
print('Expected cost(approx): 0.693')
print('\n')
print('Gradient at initial theta (zeros) - first five values only: \n', grad[:5])
print('\n')
print('Expected gradients (approx) - first five values only:\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');


# In[679]:


# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones([X.shape[1], 1])
[cost, grad] = costFunctionReg(test_theta, X, y, 10);


# In[680]:


print('Cost at initial theta (zeros): %f', cost);
print('Expected cost (approx): 3.16\n');

print('Gradient at initial theta (zeros) - first five values only:');
print(grad[0:5],'\n');
print('Expected gradients (approx) - first five values only:');
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');


# In[ ]:





# ### Learning parameters using $fminunc$ 

# In[271]:


import scipy.optimize as opt


# In[272]:


# to change the behaviour of numpy's division by zero
np.seterr(divide='ignore', invalid='ignore')


# In[637]:


# Initialize theta and lambda
initial_theta = np.zeros([X.shape[1], 1])
λ=1


# In[638]:


initial_theta.ndim


# In[639]:


# fmin_tnc will return tuple with first element is the optimized theta
result = opt.fmin_tnc(func=costFunctionReg, 
                      x0=initial_theta.flatten(),
                      args=(X, y.flatten(),  λ))


# In[643]:


# The theta returned by fmin_tnc is the first element of result
theta = result[0]
theta


# In[725]:


theta.shape


# ## Plotting Decision Boundary

# In[727]:


'''
This mapFeature function is different than the previous one, was made to be compatible with plotDecisionBoundary function
'''
def mapFeaturePlot(X1, X2):
    degree = 6
    out = np.ones([X1.size])
    for i in range(1, degree+1):
        for j in range(0, i+1):
            temp = (X1**(i-j)*(X2**j))
            out = np.hstack([out, temp])
    
    return out


# In[820]:


def plotDecisionBoundary(theta, X, y):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((u.size, v.size))
    
    for i in range(0, u.size):
        for j in range(0, v.size):
            z[i, j] = mapFeaturePlot(u[i], v[j])@theta
    
    X, Y = np.meshgrid(u, v)
    admit = plt.scatter(adm[:, 0], adm[:, 1], marker='+', color='k', linewidths=2)
    not_admit = plt.scatter(not_adm[:, 0], not_adm[:, 1], marker='o', color='y', linewidths=3)
    plt.legend((admit, not_admit), ["Microchip1 y = 1", "Microchip2 y = 0"])
    plt.show()
    
    return z


# In[821]:


Z = plotDecisionBoundary(theta, X, y)


# In[830]:


X.shape


# In[831]:


theta.shape


# In[832]:


z = X@theta


# In[833]:


z.shape


# In[ ]:





# In[852]:


# Compute accuracy on our training set
def predict(theta, X):
    m = X.shape[0] # Number of training examples
    p = np.zeros([m]);
    
    # Instructions: Complete the following code to make predictions using
    # your learned logistic regression parameters. 
    # You should set p to a vector of 0's and
    
    p = sigmoid(X @ theta)
    
    for i in range(m):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    return p


# In[853]:


p = predict(theta, X)


# In[859]:


acc = (np.sum(y == p)/y.size)*100


# In[861]:


print('Train Accuracy: %.3f' %acc)
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')


# ## Optional (ungraded) exercise

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




