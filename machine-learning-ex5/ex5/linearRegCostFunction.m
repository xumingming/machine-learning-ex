function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is mX2, theta is 2X1 => h_theta_x is mX1
h_theta_x = X * theta;

normal_J = sum(sum((1 / (2 * m)) * (h_theta_x - y) .^ 2));
reg = sum(sum((lambda / (2 * m)) * (theta(2:end) .^ 2)));
J = normal_J + reg;

% fprintf("Size of theta\n");
% disp(size(theta));

% h_theta_x is mX1, y is mX1, X(:, 1:1) is mX1
grad(1) = (1 / m) * sum((h_theta_x - y) .* X(:, 1:1));

for i = 2:size(theta, 1)
    grad(i) = (1 / m) * sum((h_theta_x - y) .* X(:, i:i)) + (lambda * ...
                                                  theta(i) / m);
end
% =========================================================================

grad = grad(:);

end
