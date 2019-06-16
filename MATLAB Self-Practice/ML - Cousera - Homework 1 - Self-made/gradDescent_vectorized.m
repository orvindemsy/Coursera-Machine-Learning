%Author : Orvin Demsy
%Date created : May, 19th 2019
%Calculate gradient descent
%Gradiant descent will return two values which are theta and history of
%cost

function [theta, J_hist] = gradDescent_vectorized(X, y, alpha, theta, num_iters)

m = length(y); %sum of data
J_hist = zeros(num_iters, 1); %initialization of iteration

for (a= 1: num_iters)
delta = 1/ m* ((X'* X* theta) - (X' * y));%remember that B' * A or A' * B 
                                 %is equal to sum of A .*B
theta = theta - alpha.* delta;  

%Compute the cost function and put it in J_hist array
J_hist(a)= costFunction(X, y, theta);
end
J_hist;
end


