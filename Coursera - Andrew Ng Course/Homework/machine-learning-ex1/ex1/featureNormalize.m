function [X_norm, mu, sigma] = featureNormalize(X)
% FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
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

%This is the second code I create, maybe I need indexing?
n = size(X,2);
for i = 1:n

mu(i) = mean(X(:,i));
sigma(i) = std(X(:,i));

X_norm(:,i) = X_norm(:,i) - mu(i);
X_norm(:,i) = X_norm(:,i)/sigma(i);

end
%SUCCESS
%Yeah for some reason, I think it can only go through if I use iteration


%This is the first code I created but it is not graded by Coursera
%The "Nice Work" statement didn't appear, although I believe it's correct
%{
X_mean = mean(X); %Calculate the mean of each feature
X_mean_feat1 = X_mean(:,1);
X_mean_feat2 = X_mean(:,2);
X_norm = [X(:,1)- X_mean_feat1, X(:,2)- X_mean_feat2];
mu = [X_mean_feat1, X_mean_feat2];

sigma = std(X);
X_norm = [X_norm(:,1)/sigma(:,1), X_norm(:,2)/sigma(:,2)];  
%}

%This is the code I got from Github
%Source https://github.com/tjaskula/Coursera/blob/master/Machine%20Learning/Week%202/machine-learning-ex1/ex1/featureNormalize.m
%{
n = size(X, 2);

for i = 1:n

	avg = mean(X(:, i));
	deviation = std(X(:, i));

	X_norm(:, i) = X_norm(:, i) - avg;
	X_norm(:, i) = X_norm(:, i) / deviation;

	mu(i) = avg;
	sigma(i) = deviation;

end
%}
% ============================================================

end
