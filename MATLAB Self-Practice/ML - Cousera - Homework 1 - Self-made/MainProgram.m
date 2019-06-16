%Uploading the data
data = load('ex1data1.txt');

%Initial Plotting
X = data(:, 1);
y = data(:, 2);

plot(X, y, 'rx', 'Markersize', 10); %Plotting X vs y
ylabel('Profit in $10,000');    %Set the y-axis label
xlabel('Population of City in 10,000s'); %Set the x-axis label

fprintf('The program is paused, press any key to continue\n');
fprintf('\n');
pause; %Pause the program

%Creating design variable vector X, include x0
m = length(data);
X = [ones(m, 1), data(:, 1)];
y = data(:, 2);

%Compute cost function with zero set of thetas
theta_a = zeros(2, 1); %Initialize theta
J1 = costFunction(X, y, theta_a); 

fprintf('=============================\n');
fprintf('Calculating cost function....\n');
fprintf('=============================\n');
fprintf('Your cost function with thetas %d %d is : %.2f\n', theta_a, J1);
fprintf('Expected values (approx) 32.07\n');
fprintf('\n');
fprintf('The program is paused, press any key to continue\n');
pause; %Pause the program
fprintf('\n');

%Compute Cost Function with another set of thetas
theta_b = [-1 ; 2];
J2 = costFunction(X, y, theta_b);

fprintf('=============================\n');
fprintf('Calculating cost function....\n');
fprintf('=============================\n');
fprintf('Your cost function with thetas %d %d is : %.2f\n', theta_b, J2);
fprintf('Expected cost value (approx) 54.24\n');
fprintf('\n');
fprintf('The program is paused, press any key to continue\n');
pause; %Pause the program
fprintf('\n');

%Some gradient descent property
iterations = 1500;
alpha = 0.01;

%Now compute the Gradient Descent

[theta_grad, J_hist] = gradDescent_vectorized(X, y, alpha, theta_a, iterations);
fprintf('=============================\n');
fprintf('Calculating gradient descent....\n');
fprintf('=============================\n');
fprintf('Using non-vectoried gradient descent formula yields:\n');
fprintf('Theta results from the computation is : %.4f %.4f\n', theta_grad);
fprintf('Expected thetas values : -3.6303 and  1.1664\n\n');  

%{
[theta_vect_grad, J_hist_2] = gradDescent_vectorized(X, y, alpha, theta_a, iterations);
fprintf('Using vectorized gradient descent formula yields:\n');
fprintf('The result from the computation is : %.3f %.3f\n', theta_vect_grad);
fprintf('Expected thetas values : -3.6303 and  1.1664\n\n'); 
%}


%Plotting linear regression on top of training sets
hold on;
plot(X(:,2), X* theta_grad, '-');
legend('training set','linear regression');
ylabel('Profit in $10,000');    %Set the y-axis label
xlabel('Population of City in 10,000s'); %Set the x-axis label
hold off;

%Predictions on profit with areas with 3,5 and 7 (x10.000) population
predict1 = [1, 3.5]* theta_grad;
predict2 = [1, 7]* theta_grad;
fprintf('For population of 35000 people the expected profit is : %.2f\n', predict1*10000);
fprintf('For population of 70000 people the expected profit is : %.2f\n\n', predict2*10000);

fprintf('The program is paused, press any key to continue\n');
fprintf('\n');
pause; %Pause the program

%Visualizing J(theta_0, theta_1)
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

%Initialize the J matrix to matrix of zeros
J_vals = zeros(length(theta0_vals), length(theta1_vals));

%Fill each index of J matrix with its corresponding cost value
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(j); theta1_vals(i)];
        J_vals(i,j) = costFunction(X, y, t);
    end
end


% Surface plot
figure(3);
surf(theta0_vals, theta1_vals, J_vals)
title('With J vals not inverted (Wrong)');
xlabel('\theta_0'); ylabel('\theta_1');

J_vals = J_vals';
% Surface plot, with inverted J_vals
figure(4);
surf(theta0_vals, theta1_vals, J_vals)
title('With J vals inverted');
xlabel('\theta_0'); ylabel('\theta_1');


