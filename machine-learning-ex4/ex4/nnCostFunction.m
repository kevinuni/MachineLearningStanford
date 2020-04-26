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




% X: 5000 x 400
% Theta1: 25 x 401

X = [ones(m, 1) X];

Z2= X * Theta1'; % Z2: 5000 x 25
a2 = sigmoid(Z2); 

a2 = [ones(m, 1) a2]; % a2: 5000 x 26

% Theta2: 10 x 26
Z3 = a2 * Theta2'; % Z3: 5000 x 10
 
a3 = sigmoid(Z3); % a3: 5000 x 10

	% note that h(x) = a3 (output layer)
for k=1:num_labels
	% y: 5000 x 1
	J = J + (1/m)*(-(y==k)'*log(a3(:,k))-(1-(y==k))'*log(1-a3(:,k)));
end

%calculate regularization
reg1 = Theta1(:,2:input_layer_size+1).^2;
reg2 = Theta2(:,2:hidden_layer_size+1).^2;
J = J + (lambda/(2*m))*(sum(reg1(:)) + sum(reg2(:)));



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
				 

Delta1 = zeros(hidden_layer_size, input_layer_size+1); % 25x401
Delta2 = zeros(num_labels, hidden_layer_size+1); % 10x26

% X: 5000 x 401
%Theta1_grad: 25x401
%Theta2_grad: 10x26
for t=1:m
	% feedforward propagation
	%perform forward propagation to compute al for l in 2,3...L
	a1 = X(t,:)'; %a1 : 401x1
	Z2 = Theta1*a1; %Z2 : 25x1
	a2 = sigmoid(Z2); %a2 : 25x1	
	a2 = [1; a2]; %a2 : 26x1
	Z3 = Theta2*a2; % Z3: 10x1
	a3 = sigmoid(Z3); %a3 : 10x1
	
	%backpropagation
	
	d3 = zeros(num_labels,1);
	for k=1:num_labels
		d3(k) = a3(k)-(y(t)==k); %d3 : 10x1
	end
	% d3 : 10x1
	
	d2 = Theta2'*d3; %d2: 26x1	
	d2 = d2(2:end); %d2: 25x1
	d2 = d2.*sigmoidGradient(Z2); % d2: 25x1
	
	Delta1 = Delta1 + d2*a1'; % 25x401
	Delta2 = Delta2 + d3*a2'; % 10x26
	
end

D1 = (1/m)*Delta1;
D2 = (1/m)*Delta2;


Theta1_grad = D1;
Theta2_grad = D2;


% calculate Theta1_grad, Theta2_grad





	


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



D1 = (1/m)*(Delta1 + lambda*[zeros(hidden_layer_size,1) Theta1(:,2:end)]); %25x401
D2 = (1/m)*(Delta2 + lambda*[zeros(num_labels,1) Theta2(:,2:end)]); %10x26



Theta1_grad = D1;
Theta2_grad = D2;




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
