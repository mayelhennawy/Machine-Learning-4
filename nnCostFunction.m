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

X = [ones(m , 1)  X];

a1 = X;

z2 = a1*Theta1';

a2 = sigmoid(z2);

a2 = [ones(m , 1)  a2];

a3 = sigmoid(a2*Theta2');

ry = eye(num_labels);

ry=ry(y,:);

cost = ry.*log(a3) + (1 - ry).*log(1 - a3);

J = -sum(sum(cost,2)) / m;

reg = sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(: , 2:end).^2));

J = J + lambda/(2*m)*reg;

%=================================================================
delta3 = a3 - ry;

delta2 = (delta3*Theta2);
delta2=delta2(:,2:end) .* sigmoidGradient(z2);



Delta1 = delta2'*a1;

Delta2 = delta3'*a2;



Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;

Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
