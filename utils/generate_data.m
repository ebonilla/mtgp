function [x, Y, xtrain, ytrain, ind_kf_train, ind_kx_train, nx] = ...
                                    generate_data(covfunc_x, D, M)
% [x, Y, xtrain, ytrain, ind_kx_train, ind_kf_train, nx] = ...
%                                    generate_data(covfunc_x, D, M)
% Generates data for testing MTGP model
%
% INPUT:
% - covfunc_x : Input covariance function
% - D         : Input dimensionality
% - M         : Number of tassks
%
% OUTPUT:    
% - x             : all inputs
% - Y             : all target values
% - xtrain        : Training inputs
% - ytrain        : Training target values
% - ind_kf_train  : Vector containing the indexes of the task to which
%                each observation y corresponds to
% - ind_kx_train  : Vector containing the indexes of the data-points in x
%                which each observation y corresponds to
% - nx            : Number of observations per each task input
%
% Edwin V. Bonilla


rho      = 0.8;    % Correlations between problems
sigma_2n = 0.0001; % Noise variance


%% Input Point
if (D==1)
    range = linspace(-1,1, 100)'; % Alternative
    x = range;
elseif (D==2)    
   range = linspace(-1,1, 20)'; % Alternative
  [X1,X2] = meshgrid(range, range);
  x = [X1(:), X2(:)];
end
[N D] = size(x);    % Total number of samples per task
n = N*M;


%% Task covariance
Kf = rho*ones(M);
idx_diag = sub2ind(size(Kf), 1:M, 1:M);
Kf(idx_diag) = 1;

%% Input covariance
ltheta_x =  eval(feval(covfunc_x{:}));
theta_x = log(ones(ltheta_x,1));      

%% Noise variances (re-parametrized)
theta_sigma = log(sqrt(sigma_2n*ones(M,1))); 

%% Full covariance
Sigma2 = diag(exp(2*theta_sigma));
Kx = feval(covfunc_x{:}, theta_x, x);
C = kron(Kf,Kx) + kron(Sigma2,eye(N));
L = chol(C)';    

%% Observations
y      = L*randn(n,1); % Noisy observations
v      = repmat((1:M),N,1); 
ind_kf = v(:);  % indices to task
v      = repmat((1:N)',1,M);
ind_kx = v(:);           % indices to input space data-points
Y      = reshape(y,N,M); % Matrix of observations across all tasks

%% Selecting data-points for training
ntrain       = floor(0.1*n);
v            = randperm(n); 
idx_train    = v(1:ntrain);
nx           = ones(ntrain,1); % observations on each task-input point
ytrain       = y(idx_train);
xtrain       = x; 
ind_kx_train = ind_kx(idx_train);
ind_kf_train = ind_kf(idx_train);





return;



