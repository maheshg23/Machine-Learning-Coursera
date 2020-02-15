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
% X = 5000 X 401
%y = 5000 X 1
%Theta1 = 25 X 401
%Theta2 = 10 X 26

a1 = [ones(m, 1) X];
a2=sigmoid(a1*Theta1');
a2 = [ones(m, 1) a2];
a3=sigmoid(a2*Theta2');


d = zeros(num_labels,num_labels);
for j=1:num_labels
	d(j,j)=1;
end
Y= zeros(m,num_labels);

for h=1:m
	for j=1:num_labels
		if(y(h) == j)
			Y(h,:)=d(:,j);
		endif
	end
end

for i=1:m
	for k=1:num_labels
		J = J + ((-1/m)*((log(a3(i,k)) * Y(i,k)) + ((log(1-(a3(i,k)))) * (1 - Y(i,k)))));
	end
end


Z=0;
for j=1:hidden_layer_size
	for k=2:input_layer_size+1
		Z = Z+ Theta1(j,k)^2;
	end
end

for j=1:num_labels
	for k=2:hidden_layer_size+1
		Z = Z+ Theta2(j,k)^2;
	end
end

J = J + ((lambda/(2*m))*Z);

D1=zeros(size(Theta1));
D2 =zeros(size(Theta2));
for i=1:m
	A1 = [1 X(i,:)];
	A2=sigmoid(A1*Theta1');
	A2 = [1 A2];
	A3=sigmoid(A2*Theta2');
	for k=1:num_labels
		d3(k)=(A3(k) - (Y(i,k)));
	end
	
	%for j=1:hidden_layer_size
		d2 = (d3*Theta2)(2:end) .* sigmoidGradient(A1*Theta1');
		%d2=d2(2:end);
	%end
	D1 = D1+(d2'*A1);
	D2 = D2+(d3'*A2);
	
end

D1(:,2:end) = D1(:,2:end) + (lambda)*Theta1(:,2:end);
D2(:,2:end) = D2(:,2:end) + (lambda)*Theta2(:,2:end);

Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
 