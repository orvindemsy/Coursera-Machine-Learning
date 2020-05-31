#!/usr/bin/env python
# coding: utf-8

# # Exercise 6: Support Vector Machine

# # 2. Spam Classification 
# In this part of the exercise, you will use SVMs to build your own spam filter. You will be training a classifier to classify whether a given email, x, is spam (y = 1) or non-spam (y = 0).

# ## 2.1 Preprocessing E-mails
# To use an SVM to classify emails into Spam v.s. Non-Spam, you first need to convert each email into a vector of features. In this part, you should produce a word indices vector for a given email.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn import svm
from pandas import DataFrame as dframe


# In[2]:


# ==================== Part 1: Email Preprocessing ====================
# Several functions need to be built to support processEmail, such as vocabList()


# In[3]:


def getVocabList():
    # GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    # cell array of the words
    #     vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt 
    #     and returns a cell array of the words in vocabList.
    
    vocabList = {}
    
    with open('vocab.txt', 'r') as f:
        for line in f:
            key, value = line.split()
            vocabList[int(key)-1] = value
            
    return vocabList


# ### 2.1.1 Vocabulary List
# After preprocessing the emails, we have a list of words for each email. 
# 
# We have chosen only the most frequently occuring words as our set of words considered (the vocabulary list). Since words that occur rarely in the training set are only in a few emails, they might cause the model to overfit our training set. The complete vocabulary list is in the file vocab.txt 
# 
# Given the vocabulary list, we can now map each word in the preprocessed emails into a list of word indices that contains the index of the word in the vocabulary list. For example, in the sample email, the word \anyone" was first normalized to "anyone" and then mapped onto the index 86 in the vocabulary list.

# In[4]:


vocabList = getVocabList()


# In[5]:


len(vocabList)


# In[6]:


def processEmail(email_contents):
    '''
    PROCESSEMAIL preprocesses a the body of an email and
    returns a list of word_indices 
       word_indices = PROCESSEMAIL(email_contents) preprocesses 
       the body of an email and returns a list of indices of the 
       words contained in the email. 
    '''

    
    #  ========================== Preprocess Email ===========================
    # Lower case
    email_contents = email_contents.lower()
    
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space
    email_contents = re.sub('^<[^<>]+>$',' ', email_contents) 
    
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    
    # ========================== Tokenize Email ===========================
    
    # Tokenize words
    tokenized = word_tokenize(email_contents)
    
    # Return indices of words that appear in the content
    word_indices = list()
    
    # Define porter stemmer
    ps = PorterStemmer()
    
    # Stem the words and match it to dictionary vocabList
    for i in range(len(tokenized)):
        stemmed = ps.stem(tokenized[i])
    
        # Remove any non alphanumeric characters
        stemmed = re.sub('[^a-zA-Z0-9]', '', stemmed)

        # Look up the word in the dictionary and add to word_indices if exists
        m = len(vocabList)
    
        for i in range(m):
            if stemmed == vocabList[i]:
                word_indices.append(i)
            
    
    return word_indices


# In[7]:


# First read email sample
with open('emailSample1.txt') as file:
    email_contents = file.read()


# In[8]:


# Extract features
word_indices = processEmail(email_contents)


# In[9]:


# Print stats
print('Word indices:\n')
print(word_indices)


# ## 2.2 Extracting Features from Emails
# You will now implement the feature extraction that converts each email into
# a vector in $\mathbb{R}^n$. For this exercise, you will be using *n = # words* in vocabularylist.
# 
# Specifically, the feature $x_{i} {\in} \{0, 1\}$ for an email corresponds to whether the *i-th* 
# word in the dictionary occurs in the email
# ![spam_extracted_feat](./img/spam_extracted_feat.jpg)

# In[10]:


# ==================== Part 2: Feature Extraction ====================
# This part will convert each email into a vector of feature R^n


# In[11]:


def emailFeatures(word_indices):
    # Number of words in the dictionary
    n = len(vocabList)
    
    # Return the feature vector x correctly
    x = np.zeros([n, 1])
    
    # New feature vector, 1 is substracted to conform with python indexing
    x[word_indices] = 1
    
    return x


# In[12]:


# Extract features
file_contents = email_contents
word_indices = processEmail(file_contents)
features = emailFeatures(word_indices)


# In[13]:


pd.options.display.max_rows=None


# In[14]:


dframe(features)


# In[15]:


# Print stats
print('Length of feature vector: %d' %len(features))
print('Number of non-zero entries: %d' %sum(features>0))


# ## 2.3 Training SVM for Spam Classification

# In[16]:


# =========== Part 3: Train Linear SVM for Spam Classification ========
# This section will train a linear classifier to determine if an email is Spam or Not-Spam


# In[17]:


# Load the spam email train dataset
spamTrain = sio.loadmat('spamTrain.mat')


# In[18]:


type(spamTrain)


# In[19]:


spamTrain.keys()


# In[20]:


X = spamTrain['X']
y = spamTrain['y']


# In[21]:


X.shape, y.shape


# In[22]:


# Set up classifier
clf = svm.SVC(C=0.1, kernel ='linear')
model = clf.fit(X, y.flatten())


# In[23]:


# Predict on train data
p = model.predict(X)


# In[24]:


print('Training Accuracy: %.3f' %(np.mean(p==y.flatten())*100))


# In[25]:


# =================== Part 4: Test Spam Classification ================
# This section will evaluate trained model on a test set. Test set is included in spamTest.mat


# In[26]:


# Load the spam email train dataset
spamTest = sio.loadmat('spamTrain.mat')


# In[27]:


type(spamTest)


# In[28]:


spamTest.keys()


# In[29]:


Xtest = spamTest['X']
ytest = spamTest['y']


# In[30]:


# Predict on test data
p = model.predict(Xtest)


# In[31]:


print('Training Accuracy: %.3f' %(np.mean(p==ytest.flatten())*100))


# ## 2.4 Top Predictors for Spam
# To better understand how the spam classier works, we can inspect the parameters to see which words the classifier thinks are the most predictive of spam.

# In[32]:


# ================= Part 5: Top Predictors of Spam ====================
# This section will inspect the learned weights to get better understanding how this model determine whether or not an email 
# is spam. The highest value of weight will be printed, this weights is the most likely indicator of spam email


# In[33]:


# Create array of weights
weights = model.coef_.flatten()


# In[34]:


# Sort the weights and obtain vocabulary list
idx = np.argsort(weights)[::-1]


# In[35]:


idx


# In[36]:


# Note that due to calculation precision the result might differ from MATLAB result
print('Top predictors of span:\n')
for i in idx[:16]:
    print(vocabList[i], weights[i])


# ## 2.6 Try your own emails

# In[37]:


##### =================== Part 6: Try Your Own Emails =====================
# The trained classifier to test your own email, paste your own email to spamSample2.txt


# In[38]:


# You can change this to any file name to see how it is predicted
filename = 'spamSample2.txt'


# In[39]:


with open(filename) as file: 
    file_contents = file.read()


# In[40]:


# Read and predict
word_indices = processEmail(file_contents)
x = emailFeatures(word_indices)


# In[41]:


# Since x only contains one sample of 1899 feature
# x needs to be transposed so that it is in shape of 1 x 1899
x = x.T
x.shape


# In[42]:


# Predict x
p = model.predict(x)


# In[43]:


print('Processed: %s' %filename)
print('Spam Classification: %d\n' %p)
print('1 indicates spam, 0 indicates not spam)')

