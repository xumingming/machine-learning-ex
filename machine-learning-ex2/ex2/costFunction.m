function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

% X is (m by n + 1)
% theta is (n + 1 by 1)
% so h_theta_x is (m by 1)
h_theta_x = sigmoid(X * theta);

% y is  (m by 1)
% y' is (1 by m)
%  h_theta_x is (m by 1)
J = sum((-1 * y' * log(h_theta_x)) - ((1 - y)' * log(1 - h_theta_x))) ...
    / m;

% X is (m by n + 1) -> X' is (n + 1 by m)
% theta is (n + 1 by 1)
% y is (m by 1)
grad = (X' * (sigmoid(X * theta) - y)) / m;

% =============================================================

end