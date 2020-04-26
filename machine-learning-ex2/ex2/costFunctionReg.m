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


%lambda = 0;
estimate = sigmoid(X*theta);
theta_reg = theta(2:size(theta));
degree = length(theta);

J=(1/m)*(-y'*log(estimate)-(1-y)'*log(1-estimate)+(lambda/2)*sum(theta_reg'*theta_reg));


grad(1) = (1/m)*(X(:,1))'*(estimate-y); 



for i=2 : degree
	grad(i) = (1/m)*(X(:,i))'*(estimate-y)+(lambda/m)*theta(i); 
	%grad(i) = (1/m)*(X(:,i))'*(estimate-y); 
end 





% =============================================================

end
