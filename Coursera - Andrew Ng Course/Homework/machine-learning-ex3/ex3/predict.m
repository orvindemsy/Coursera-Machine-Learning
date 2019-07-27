function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%------------------------------------------------------------------
X = [ones(size(X,1),1), X];    %Adding bias input to input layer

%Both if these matrix has dimension of 25 x 5000
z2 = zeros(size(Theta1,1), size(X,1));  %z2 Initial declaration
a2 = zeros(size(Theta1,1), size(X,1));  %a2 Initial declaration

z2 = Theta1 * X';   %Matrix dim 25x401 * 401*5000 = 25*5000
a2 = sigmoid(z2);   %Matrix dim 25*5000
%-------------------------------------------------------------------
a2 = [ones(1, size(a2,2)); a2];  %Adding bias input to input layer
                                %Matrix dim 26x5000

%Both if these matrix has dimension of 10 x 5000
z3 = zeros(size(Theta2,1), size(X,1));  %z2 Initial declaration
a3 = zeros(size(Theta2,1), size(X,1));   %a2 Initial declaration

z3 = Theta2 * a2; %Matrix dim 10x26 * 26x5000 = 10x5000
a3 = sigmoid(z3); %Matrix dim 10x5000
a3 = a3';
%Predict the final result in each example
for c = 1:m
    [max_val, ind] = max(a3(c,:), [], 2);
        p(c) = ind;
end


% =========================================================================


end
