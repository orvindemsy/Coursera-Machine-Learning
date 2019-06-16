function [X_norm, mu, sigma] = featureNormalize(X)
% Source: https://github.com/Borye/machine-learning-coursera-1/blob/master/Week%202%20Assignments/Linear%20Regression%20with%20Multiple%20Variables/mlclass-ex1/featureNormalize.m
% This code is used as a compatison of the results
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% 
%{
X_mean = mean(X,1); %Calculate the mean of each feature
X_mean_feat1 = X_mean(:,1);
X_mean_feat2 = X_mean(:,2);
mu = [X(:,1)- X_mean_feat1, X(:,2)- X_mean_feat2];

sigma = std(X);
X_norm = [mu(:,1)/sigma(:,1), mu(:,2)/sigma(:,2)];  
%}
mu = mean(X_norm);
sigma = std(X_norm);

tf_mu = X_norm - repmat(mu,length(X_norm),1);
tf_std = repmat(sigma,length(X_norm),1);

X_norm = tf_mu ./ tf_std;

% ============================================================

end
