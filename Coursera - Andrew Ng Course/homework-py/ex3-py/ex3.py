#!/usr/bin/env python
# coding: utf-8
# %%

# # Exercise 3 Multi-class Classification 
# For this exercise, you will use logistic regression and neural networks to
# recognize handwritten digits (from 0 to 9).  
# This exercise will show you how the methods you've learned can be used for this
# classification task.

# ## Dataset
# There are 5000 training examples in ex3data1.mat, where each training
# example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is
# represented by a floating point number indicating the grayscale intensity at
# that location. 
# 
# The 20 by 20 grid of pixels is unrolled into a 400-dimensional
# vector. Each of these training examples becomes a single row in our data
# matrix X.  
# 
# To make things more compatible with Octave/MATLAB indexing, where there is no zero index, we have mapped the digit zero to the value ten. Therefore, a `0` digit is labeled as `10`, while the digits `1` to `9` are labeled as `1` to `9` in their natural order.

# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from pandas import DataFrame as dframe


# %%


# Setup the parameters you will use for this part of the exercise
input_layer_size = 400 # 20 x 20 input images of digits
num_labels = 10 # 1-10 labels (0-9)


# %%


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
# Load training data
data = sio.loadmat('ex3data1.mat')


# %%


data.keys()


# %%


# Storing training data to X and y
X = data['X']
y = data['y']
m = len(X)


# %%


X.shape, y.shape


# %%


# Select 100 data randomly to be visualized
_, ax = plt.subplots(10, 10, figsize=(10,10))
for i in range(10):
    for j in range(10):
        pic = X[np.random.randint(m)].reshape([20, 20], order='F')
        ax[i,j].imshow(pic, cmap='gray')          
        ax[i,j].axis('off')


# ## Vectorizing Logistic Regression
# You will be using multiple one-vs-all logistic regression models to build a
# multi-class classifer. Since there are 10 classes, you will need to train 10
# separate logistic regression classifers.
# 
# To make this training efficient, it is
# important to ensure that your code is well vectorized. In this section, you
# will implement a vectorized version of logistic regression that does not employ
# any `for loops`

# %%


# Sigmoid function
def sigmoid(z):
    sig = 1/(1+np.exp(-z))
    return sig


# ### Vectorizing the cost function
# ![vect_costfunct1](./img/vect_costfunct1.png)
# ![vect_costfunct2](./img/vect_costfunct2.png)

# %%


# Later in this code, scipy.optimize.fmin_cg will be used, 
# unlike previous homework, this function requires grad descent function as its parameter
# Thus, regularized cost function and grad descent function will be defined separately as well

def regCostFunction(theta, X, y, λ):
    # Initialize some useful values
    m = X.shape[0]; # number of training examples

    # You need to return the following variables correctly 
    J = 0;
    
    # Sigmoid
    z = X@theta
    g_z = sigmoid(z)
    
    # Cost function
    J = 1/m * np.sum([(-y * np.log(g_z))-(1-y)*(np.log(1-g_z))]) + (λ/(2*m) * np.sum(theta[1:]**2))
    
    return J


# ### Vectorizing the gradient
# ![vect_grad1](./img/vect_grad1.png)
# ![vect_grad2](./img/vect_grad2.png)
# ![vect_grad3](./img/vect_grad3.png)

# %%


def regGradDescent(theta, X, y, λ):
    # Initialize some useful values
    m = X.shape[0]; # number of training examples

    # You need to return the following variables correctly 
    grad = np.zeros([theta.size,1]);
    
    # Sigmoiddd
    z = X@theta
    g_z = sigmoid(z)
    
    # Grad descent
    grad = 1/m * X.T @ (g_z - y);
    grad[1:] = grad[1:] + (λ/m * theta[1:])
    
    return grad


# ### Vectorizing regularized logistic regression
# ![vect_regcostgrad1](./img/vect_regcostgrad.png)

# %%


# LRCOSTFUNCTION Compute cost and gradient for logistic regression with regularization
# This function is copied from previous homework, it is already vectorized

def lrCostFunction(theta, X, y, λ):
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
    
    return J, grad    


# %%


# ============ Part 2a: Vectorize Logistic Regression ============
# Testing lrCostFunction() with regularization
np.random.seed(12346)

theta_t = np.array([-2, -1, 1, 2])
X_t = np.hstack([np.ones([5, 1]), (np.arange(1,16)/10).reshape(5, 3, order='F')])
y_t = (np.array([1, 0, 1, 0, 1]) >= 0.5)*1

lambda_t = 3


# %%


print(y_t)


# %%


print(X_t)


# %%


[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t)


# %%


print('Cost: ', J);
print('Expected cost: 2.534819\n');
print('Gradients:');
print(grad, '\n');
print('Expected gradients:');
print(' 0.146561  -0.548558  0.724722  1.398003');


# ## One-vs-all Classification

# %%


# ============ Part 2b: One-vs-All Training ============
from scipy import optimize


# %%


def oneVsAll(X, y, num_labels, λ):
    '''
    oneVsAll will use fmin_cg to optimize theta values
    Note that the argument passed as X should already has intercept term, shape 5000 x 401
    '''
    # Some useful variables
    (m, n) = X.shape # m = samples, n = features + bias term

    # Need to return following variables correctly
    all_theta = np.zeros([num_labels, n])
        
    # Opmizing regCostFunction by finding optimum theta values
    # Notice that in (y == i+1), '+1' is required to match the values of y which range from 1-10, python counts from 0
    # Don't forget the x0 and y has to be in 1-D
    for i in range(num_labels):
        all_theta[i] = optimize.fmin_cg(f=regCostFunction, fprime=regGradDescent, 
                                           x0=all_theta[i], 
                                           args=(X, ((y == i+1)*1).flatten(), λ), 
                                           maxiter=50)
    
    return all_theta


# %%


theta_opt = oneVsAll(X, y, num_labels=10, λ=0.1)


# %%


dframe(theta_opt).head()


# %%


def predictOneVsAll(all_theta, X):
    '''
    predictOneVsAll will predict the output of each sample and overall accuracy
    This function receives two parameters, optimized theta and X
    Note that the argument passed as X should already has intercept term, shape 5000 x 401

    '''
    # Hypothesis
    h_x = X@all_theta.T
    
    # argmax will return index of maximum value within a indicated axis
    # Likewise, '+1' is added since python indexing starts at 0
    p = np.argmax(h_x, axis=1)+1
    
    acc = (p == y.flatten())*1
    acc = (np.sum(acc)/acc.size)*100
    
    return p, acc


# %%


pred, acc = predictOneVsAll(theta_opt, X)


# %%


# Acquired accuracy may differ from MATLAB
print('Training Set Accuracy: ', acc)


# %%




