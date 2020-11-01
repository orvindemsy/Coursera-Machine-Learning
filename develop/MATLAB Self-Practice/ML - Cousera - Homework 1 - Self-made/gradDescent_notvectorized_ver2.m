%Author : Orvin Demsy
%Date created : May, 19th 2019
%Calculate gradient descent
%Gradiant descent will return two values which are theta and history of
%cost

%In this code I thinkered with the formula, I'm debating whether to write
%theta both in argument and return or only in argument
%It turned out that it's better to write both with same variables because
%in the end the variable updates itself (recursive?)
function [theta_res, J_hist] = gradDescent_notvectorized(X, y, alpha, theta, num_iters)
%n = length(theta); %numbers of features
m = length(y); %numbers of data
J_hist = zeros(num_iters, 1); %initialization of iteration
t = zeros(2,1); %initialize temporary var for theta
theta_res = zeros(2,1);
theta_res = theta;

%This grad descent is applicable to datasets with only 1 features

i = 1; %declare variable for iteration
  while (i <= num_iters)  
    h = X * theta_res;    
    t1 = theta_res(1) - alpha * (1 / m) * sum((h - y) .* X(:,1));
    t2 = theta_res(2) - alpha * (1 / m) * sum((h - y) .* X(:,2));
    theta_res(1) = t1;
    theta_res(2) = t2;
    
    J_hist(i) = costFunction(X, y, theta_res);
    i = i + 1; %increment value of i
  end
end


%{
%This grad descent formula is applicable to datasets with only 1 features
i = 1; %declare variable for iteration
  while (i <= num_iters)  
    for (j= 1: n)
        h = X * theta;    
        t(j) = theta(j) - alpha * (1 / m) * sum((h - y) .* X(:,j));
    end
    for (j= 1: n)
    theta(j) = t(j);
    end
    theta_res = theta;
    J_hist(i) = costFunction(X, y, theta_res);
    i = i + 1; %increment value of i
  end
end
%}



