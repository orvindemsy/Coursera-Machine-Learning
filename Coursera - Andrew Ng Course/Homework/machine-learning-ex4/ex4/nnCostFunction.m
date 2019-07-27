function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X_bias = [ones(size(X,1),1), X]; %X with bias input added, dim 5000*401
a1 = X_bias; %assign variable a1 as X_bias
%%FORWARD PROPAGATION

%HIDDEN LAYER
%z unit in hidden layer
z2 = zeros(hidden_layer_size, size(X, 1));
z2 = Theta1 * a1';  %25*401 X 401*5000 = 25*5000
                                   
%activation unit in hidden layer
a2 = zeros(size(z2));
a2 = sigmoid(z2);  %dimension 25*5000
a2 = [ones(1,size(a2,2)); a2]'; %transposing & adding bias unit to a_2
                                   %dimension 5000*26
%OUTPUT LAYER
%z unit in output layer
z3 = zeros(hidden_layer_size, size(X,1));
z3 = Theta2 * a2';%10*26 X 26*5000 = 10*5000
                                   
%activation unit in output layer, a2 as input
a3 = sigmoid(z3); %dimension 10*5000 
a3 = a3'; %change the dimension to 5000*10

%Expanding y vector to matrix Y with dimension of 5000 x 10, containing a
%single value

I = eye(num_labels);
Y = zeros(m, num_labels);

Y = I(y, :); %dimension 5000*10

Reg = 0; %Initialize variable for regularization
Reg = lambda/(2*m) * ...
    [sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))];

%Cost Function Computation
J = -1/m * sum(sum((log(a3) .* Y) + (log(1 - a3) .* (1-Y)))) + Reg;

%%BACK PROPAGATION
%OUTPUT LAYER
d3 = a3 - Y; %dimension 5000*10

%HIDDEN LAYER
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2)';%5000*10 X 10*25 .X 5000*25

%Delta value initialize
Delta1 = 0;
Delta2 = 0;

Delta1 = Delta1 + d2' * a1; %25*5000 X 5000*401
Delta2 = Delta2 + d3' * a2; %10*5000 X 5000*26

Theta1_grad = 1/m * Delta1; %dim 25*401
Theta2_grad = 1/m * Delta2; %dim 10*26

%%REGULARIZATION
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
