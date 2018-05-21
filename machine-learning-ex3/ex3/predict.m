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

% add the intercept term
real_X = [ones(m, 1) X];
               
% Theta1 is 25X401, real_X is 5000X401, a2 is 5000X25
a2 = sigmoid(real_X * Theta1');

% real_a2 is 5000X26, Theta2 is 10X26
% a3 is 5000X10
real_a2 = [ones(size(a2, 1), 1) a2];
a3 = sigmoid(real_a2 * Theta2');

% pre_p is 5000X10
pre_p = a3;

for i = 1:m
    % find the largest probability with the corresponding index
    [target, index] = max(pre_p(i, :));
    
    % if there there is a one, set it
    if round(target) == 1
        p(i) = index;
    end
end    
% =========================================================================


end
