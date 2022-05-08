clear clc

% init constants

input = randn(10000,1); % gaussian noise

trainLen = 2000;
testLen = 2000;
initLen = 100;


% generate the ESN reservoir
inSize = 1;
outSize = 100;
resSize = 500;
alpha = 0.3; % leaking rate

rand('seed', 42);
W_in = (rand(resSize,1+inSize)-0.5).* 1;
W = rand(resSize,resSize)-0.5;
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
% W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
disp 'Computing gaussian noise...';
opt.disp = 0;
rho_W = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rho_W);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = input(initLen+2:trainLen+1)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = input(t);
	x = (1-alpha)*x + alpha*tanh( W_in*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output
reg = 1e-8;  % regularization coefficient
X_T = X';
W_out = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% Wout = Yt*pinv(X);

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize,testLen);
u = input(trainLen+1);
for t = 1:testLen 
	x = (1-alpha)*x + alpha*tanh( W_in*[1;u] + W*x );
	y = W_out*[1;u;x];
	Y(:,t) = y;
	% generative mode:
	u = y;
	% this would be a predictive mode:
	%u = data(trainLen+t+1);
end

errorLen = 500;
mse = sum((input(trainLen+2:trainLen+errorLen+1)'-Y(1,1:errorLen)).^2)./errorLen;
disp( ['MSE = ', num2str( mse )] );

% plot some signals
figure(1);
plot( input(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
hold on;
plot( Y', 'b' );
hold off;
axis tight;
title('Target and generated signals y(n) starting at n=0');
legend('Target signal', 'Free-running predicted signal');

figure(2);
plot( X(1:20,1:200)' );
title('Some reservoir activations x(n)');

figure(3);
bar( W_out' )
title('Output weights W^{out}');

