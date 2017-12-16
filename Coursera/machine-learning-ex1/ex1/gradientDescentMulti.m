function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
total_features = size(X, 2);

for iter = 1:num_iters
    last_theta = theta;
    for f = 1:total_features
        partial_deri = ((X*last_theta - y)'*X(:, f))/m;
        % partial_deri = 0;
        % for i = 1:m
          % partial_deri = partial_deri + (theta_prev'*X(i, :)'-y(i))*X(i, j);
        % end
        % partial_deri = partial_deri/m;

        theta(f) = last_theta(f) - (alpha*partial_deri);
    end 
    J_history(iter) = computeCost(X, y, theta);

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
