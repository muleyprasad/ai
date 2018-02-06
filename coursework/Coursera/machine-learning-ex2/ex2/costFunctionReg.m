function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

f_of_x = sigmoid(X*theta);

J = (sum((-y.*log(f_of_x)) - ((1-y).*log(1-f_of_x)))/(m)) + sum(theta(2:end).^2)*(lambda/(2*m));

total_features = size(X, 2);

grad(1) = (sum((f_of_x - y).*X(:,1))/m);

for f = 2:total_features
    grad(f) = (sum((f_of_x - y).*X(:,f))/m) + (lambda/m)*theta(f);
end




% =============================================================

end
