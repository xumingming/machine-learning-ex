function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% X is (m by n + 1)
% theta is (n + 1 by 1)
% so h_theta_x is (m by 1)
h_theta_x = sigmoid(X * theta);

% y is  (m by 1)
% y' is (1 by m)
%  h_theta_x is (m by 1)
old_J = sum((-1 * y' * log(h_theta_x)) - ((1 - y)' * log(1 - h_theta_x))) ...
    / m;

theta_without_the_intercept = theta(2:size(X, 2), :);
J = old_J + (lambda / (2 * m)) * sum(theta_without_the_intercept .^ 2);

% X is (m by n + 1) -> X' is (n + 1 by m)
% theta is (n + 1 by 1)
% y is (m by 1)
% old_grad is (n + 1)X1
old_grad = (X' * (sigmoid(X * theta) - y)) / m;

reg_param = [0; ((lambda / m) * theta_without_the_intercept)];
grad = old_grad + reg_param;

% =============================================================

grad = grad(:);

end
