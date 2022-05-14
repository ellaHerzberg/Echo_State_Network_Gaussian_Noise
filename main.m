%% EX_3 - ESN
clear;
close all;
clc;


% Define network size
in_size = 1;
out_size = 100;
res_size = 500;
input_len = 10000;
train_len = input_len - out_size;

% Learning input_scalestants
zero_w_percent = 0.4;   % 0-weights precentage
alpha = 0.99;           % leaking rate
beta = 1e-10;           % regularization coefficient
input_scale = 0.0005;   % how strongly input affects

% Set the input and target matrix
input = randn(input_len, in_size); % gaussian noise
Y0 = buffer(input, out_size, out_size - 1);
Y0 = Y0(:, out_size+1:end);
Y0 = flip(Y0);
input = input(out_size+1:end);


% generate the ESN reservoir
W_in = randn(res_size,in_size) * input_scale;
K = randn(res_size,res_size);

% Set some weights as zero, and normalize network:
for row = 1:res_size
   K(row, :) = [K(row, 1:res_size*(1-zero_w_percent)) zeros(1, res_size * zero_w_percent)];
   K(row, randperm(size(K, 2))) = K(row, :);
end

rho_W = real(eigs(K,1,'largestreal'));
K = K * (1/rho_W);

% Allocated memory for the matrix state
X = zeros(1+in_size+res_size, input_len - out_size);
R = zeros(res_size, input_len - out_size);


% Run the reservoir with the data and collect X
R(:,1) = (1-alpha)*R(:,1) + alpha*tanh(W_in*input(1)+K*R(:,1));
for t = 1:train_len
    curr_r = R(:,t);
    X(:,t) = [1;input(t);curr_r];
    if t <= train_len-1
        R(:,t+1) = (1-alpha)*curr_r + alpha*tanh(W_in*input(t+1)+K*curr_r);
    end
end

% train the output
W_out = Y0*X' * (X*X' + beta*eye(1+in_size+res_size))^(-1);

% run the trained ESN in a predictive mode.
Y = zeros(out_size,train_len);
u = input(1);
for t = 1:train_len
	curr_x = [1;input(t);R(:,t)];
	y = tanh(W_out * curr_x);
	Y(:,t) = y;
	u = input(t);
end

% Calc R square error
mean_y0 = mean(Y0,2);
SSres = sum((Y-Y0)'.^2);
SStot = sum((Y0-mean_y0)'.^2);

for s = 1:out_size
    error(s) = 1-(SSres(:,s)/SStot(:,s));
end

% Plots
plot(1:out_size, error,'color','g');
title('R square for each output neuron');
xlabel("Memory step ");
ylabel("Correlation coefficient");

% Disp Memory Capacity
memory = sum(error);
fprintf('Memory Capacity:');
disp(memory);
 