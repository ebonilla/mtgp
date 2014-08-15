function [ logtheta_all deriv_range] = init_mtgp_default(xtrain, covfunc_x, M, irank  )
% Initializes parameters of mtgp by default
% You should pay careful attention to modifying this as it may be a 
% bad initialization for your problem!
% 
% INPUT:
% - xtrain: Input training data
% - covfunc_x: Input covariance function
% - M: Number of tasks
% - irank: Rank required for Kf
%
% OUTPUT:
% - logtheta_all: Vector of all hyper-paramters
% - deriv_range: Indices of hyper-parameters to optimize for
%
% Edwin V. BOnilla

nlf           = irank*(2*M - irank +1)/2;    % Number of parameters for Lf

theta_lf0     = init_Kf(M,irank);
theta_kx0     =  init_kx(xtrain, covfunc_x);
theta_sigma0  = init_sigma(M);
logtheta_all  = [theta_lf0; theta_kx0; theta_sigma0];

% assumes that we don't want to optimize the signal variance of kx
deriv_range = 1 : length(logtheta_all);
deriv_range(nlf+length(theta_kx0)) = [];




return;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_kx0 = init_kx(xtrain, covfunc_x)
D         = size(xtrain,2);
L         = eval(feval(covfunc_x{:}));
theta_kx0 = log(ones(L,1)); 
 
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_lf0 = init_Kf(M,irank)
Kf0 = eye(M);                 % Init to diagonal matrix (No task correlations)
Lf0 = chol(Kf0)';
theta_lf0 = lowtri2vec_inchol(Lf0,M,irank);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function theta_sigma0 = init_sigma(M) % noise variances
theta_sigma0 =  (1e-7)*rand(M,1);  
 
return;



