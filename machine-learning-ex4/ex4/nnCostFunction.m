function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
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


% ====== vectorized_y is mX10 -- this is too slow ======
% vectorized_y = zeros(m, num_labels);
% for i = 1:m
%     vectorized_y(i, y(i)) = 1;
% end    
% ====== vectorized_y is mX10 -- this is too slow ======


% vectorized y
vectorized_y = ([1:num_labels] == y);
% add bias to the X
X_with_bias = [ones(m, 1) X];


% Step 2 - perform the forward propagation:
%    a1 equals the X input matrix with a column of 1's added (bias units)
%    z2 equals the product of a1 and Θ1
%    a2 is the result of passing z2 through g()
%    a2 then has a column of 1st added (bias units)
%    z3 equals the product of a2 and Θ2
%    a3 is the result of passing z3 through g()
%    Cost Function, non-regularized

% X_with_bias is mX401, Theta1 is 25 * 401
z2 = X_with_bias * Theta1';
a2 = sigmoid(z2);

% a2_with_bias is mX26, Theta2 is 10X26, z3 is mX10
% a3 is mX10
a2_with_bias = [ones(m, 1) a2];
a3 = sigmoid(a2_with_bias * Theta2');
% h_theta_x is mX10
h_theta_x = a3;

% vectorized_y is mX10, log(h_theta_x) is mX10
cost = (-vectorized_y .* log(h_theta_x)) - ((1 - vectorized_y) .* ...
                                            log(1 - h_theta_x));
% 
reg_1 = sum(
    sum(
        Theta1(:, 2:input_layer_size + 1) .^ 2
    )
);
reg_2 = sum(
    sum(
        Theta2(:, 2:hidden_layer_size + 1) .^ 2
    )
);

reg = (lambda / (2 * m)) * (reg_1 + reg_2);
J = (1/m) * sum(sum(cost)) + reg;

a1 = X_with_bias;
a2 = a2_with_bias;

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m,
	a1t = a1(t,:);
	a2t = a2(t,:);
	a3t = a3(t,:);
	yt = vectorized_y(t,:);
	d3 = a3t - yt;
	d2 = Theta2' * d3' .* sigmoidGradient([1;Theta1 * a1t']);
	delta1 = delta1 + d2(2:end)*a1t;
	delta2 = delta2 + d3' * a2t;
end

Theta1_grad = (1 / m) * delta1 + (lambda / m) * [zeros(size(Theta1, ...
                                                  1), 1) Theta1(:, 2:end)];
Theta2_grad = (1 / m) * delta2 + (lambda / m) * [zeros(size(Theta2, ...
                                                  1), 1) Theta2(:, 2:end)];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
