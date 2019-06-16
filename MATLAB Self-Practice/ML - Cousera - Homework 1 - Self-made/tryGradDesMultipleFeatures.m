%Created by: Orvin Demsy
%Date: 5 June 2019
%Purpose: understanding grad descent with multiple features and creating
%vectorized gradient descent formula

%Creating 4 features of X with Y as result 
X = magic(5);
X(:,1) = ones(5,1);

%result of Y 
y = X(:, 5);

%Removing 5th column of X, because now 5th column has turned into Y
%Now X is a 5x4 matrix, meaning it has 3 features
X(:,5) = [];

%Defining theta with all-ones 4x1 matrix 
theta_a = zeros(size(X,2),1);

%Calculating hypothesis to produce 5x1 matrix
%h = X * theta_a;   

alpha = 0.01; %Define learning rate
iterations = 1500; %Define how many iterations needed

%So currently we have 5x4 matrix meaning 5 datasets and 3 different
%features, there are 4 columns  but the first one is a dummy features which
%contains ones
%[theta, J_function] = gradDescent_vectorized(X, y, aplha, theta, num_iters)

[theta, J_hist] = gradientDescent(X, y, alpha, theta_a, iterations);
fprintf('By using the not vectorized grad Descent we got %.3f %.3f %.3f %.3f\n', theta);
