#!/usr/bin/env python
# coding: utf-8

# # Exercise 2
# Coursera - Machine Learning <br>
# Andrew Ng

# # Logistic Regression

# ## Visualizing the data

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[101]:


# ==================== Part 1: Plotting ====================
# Set number of significant figures to show
np.set_printoptions(precision=5)


# In[39]:


# As alternative to pd.read_csv, np.loadtxt can be used to upload .txt data
data = np.loadtxt('ex2data1.txt', delimiter = ',')
data


# In[40]:


# Data type of data
data.dtype


# In[41]:


X = data[:, :2]
y = data[:, 2]


# In[42]:


# Find indices of positive and negative example
# (student who got admitted and not admitted)
pos = np.nonzero(y)
neg = np.where(y==0)


# In[43]:


pos


# In[44]:


neg


# In[45]:


# Admitted and Not admitted matrix
admit = data[pos, :2]
not_admit = data[neg, :2]


# In[46]:


# Check shape of admit and not_admit
print("shape of admit: ", admit.shape)
print("shape of not_admit: ", not_admit.shape)


# In[47]:


# Convert shape of admit and not_admit to 2 dimension
admit = admit.reshape(60, 2)
not_admit = not_admit.reshape(40, 2)
print("shape of admit: ", admit.shape)
print("shape of not_admit: ", not_admit.shape)


# In[48]:


# Plotting data
plt.style.use('seaborn')
adm = plt.scatter(admit[:, 0], admit[:, 1], color='g')
not_adm = plt.scatter(not_admit[:, 0], not_admit[:, 1], color='r', marker='x')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.title('1st and 2nd Student Exam Score ')
plt.legend((adm, not_adm), ("Admit", "Not Admitted"))
plt.show()
# Need to add legend here


# ## Implementation

# ### Sigmoid Function
# <img src = "img/sigmoid.jpg">

# In[49]:


# Sigmoid function
def sigmoid(z):
    sig = 1/(1+np.exp(-z))
    return sig


# In[50]:


[m, n] = X.shape
print("m = %d \nn = %d" %(m, n))


# In[51]:


# Add intercept term to X
X = np.hstack([np.ones((m, 1)), X])
X


# In[52]:


# Y stil in 1D array, convert into 2D array
y = y[:, np.newaxis]


# In[53]:


y.shape


# In[54]:


# Theta matrix
initial_theta = np.zeros((n+1, 1))
print(initial_theta)


# ### Cost function and gradient descent
# <img src = "img/cost-funct-sigmoid.jpg">
# <b>And the gradient descent defined as</b>
# <img src = "img/grad-descent-sigmoid.jpg">

# In[55]:


# ============ Part 2: Compute Cost and Gradient ============
# Cost function in logistic regression
def costFunction(theta, X, y):
    z = np.dot(X, theta)
    g_z = sigmoid(z) # Use sigmoid function for hypothesis in logistic regression
    
    #Cost function
    J = 1/m * np.sum([(-y * np.log(g_z))-(1-y)*(np.log(1-g_z))])
    
    # Grad descent
    grad = (X.T).dot((g_z-y)) * 1/m
    
    return J, grad


# In[56]:


[J1, grad1] = costFunction(initial_theta, X, y)
print("Cost function with zero theta: %.3f" %J1)
print("Grad descent with zero theta:\n", grad1)


# In[57]:


grad1.shape


# In[58]:


# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
test_theta


# In[59]:


[J2, grad2] = costFunction(test_theta, X, y)
print("Cost function with zero theta: %.3f" %J1)
print("Grad descent with zero theta:\n", grad1)


# ### Learning parameters using fminunc

# In[102]:


# ============= Part 3: Optimizing using fminunc  =============
import scipy.optimize as opt


# In[61]:


# to change the behaviour of numpy's division by zero
np.seterr(divide='ignore', invalid='ignore')


# In[62]:


# fmin_tnc will return tuple with first element is the optimized theta
result = opt.fmin_tnc(func=costFunction, 
                      x0=initial_theta.flatten(),
                      args=(X, y.flatten())) 


# In[116]:


opt_theta = result[0]
print('Theta found by optimization are:', opt_theta) # Now use this theta to compute cost using previous function


# In[117]:


# Convert opt_theta to 3 x 1 2D array
theta = opt_theta[:, np.newaxis]


# In[118]:


theta.shape


# In[119]:


X.shape


# In[120]:


# Cost function obtained using optimaized theta
[J3, grad3] = costFunction(theta, X, y)


# In[122]:


print('Cost at theta found by fminunc: %.5f\n' %J3);
print('Expected cost (approx): 0.203\n');
print('theta: \n', theta, '\n');
print('Expected theta (approx):');
print(' -25.161\n 0.206\n 0.201\n');


# In[123]:


# Plot Boundary
def plotDecisionBoundary(theta, X, y):
    if X.shape[1] <= 3:
        plot_x = np.array([X[:, 1].min()-2, X[:, 1].max()+2])
        check1 = X[:, 1].min()
        check2 = X[:, 1].max()
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0]) 
    
    # Plotting decision boundary
    plt.style.use('seaborn')
    adm = plt.scatter(admit[:, 0], admit[:, 1], color='g')
    not_adm = plt.scatter(not_admit[:, 0], not_admit[:, 1], color='r', marker='x')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('1st and 2nd Student Exam Score ')
    plt.legend((adm, not_adm), ("Admit", "Not Admitted"))
    plt.plot(plot_x, plot_y, '--')
    plt.show()


# In[124]:


plotDecisionBoundary(opt_theta, X, y)


# ### Evaluating logistic regression 

# After learning the parameters, you'll like to use it to predict the outcomes
# on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and 
# score 85 on exam 2 will be admitted
# 
# Furthermore, you will compute the training and test set accuracies of 
# our model.

# In[125]:


#  ============== Part 4: Predict and Accuracies ==============
def predict(theta, X):
    m = X.shape[0] # Number of training examples
    p = np.zeros([m]);
    
    p = sigmoid(X @ theta)
    
    for i in range(m):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    
    return p


# In[138]:


prob = sigmoid(np.array([1.0, 45.0, 85.0]) @ theta)


# In[140]:


print('For a student with scores 45 and 85, we predict an admission: %.6f' %prob)
print('Expected value: 0.775 +/- 0.002\n')


# In[134]:


# Compute accuracy on our training set
p = predict(theta, X);


# In[135]:


acc = (np.sum(y == p)/y.size)*100


# In[136]:


print('Train Accuracy: %.3f' %acc)
print('Expected accuracy (approx): 89.0\n')

