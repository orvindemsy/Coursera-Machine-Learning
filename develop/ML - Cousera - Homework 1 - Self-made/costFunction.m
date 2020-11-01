%Author : Orvin Demsy
%Date created : May, 18th 2019
%Computing cost function of linear regression
function J = costFunction (X, y, theta)

m = length(y); %Number of data

h = X * theta;

J = 1 / (2 * m) * sum ((h - y).^2);

