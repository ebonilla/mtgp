function [logtheta_all nl] = learn_mtgp( logtheta_all, deriv_range, data )
%LEARN_MTGP Learns hyperparameters of mtgp model using minimize
%
% INPUT:
% - logtheta_all : All initial hyper-parameter values 
% - deriv_range  : Indices of hyper-parameters to optimize for
% - data         : cell data in the order 
%                  [covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train]
%
% OUTPUT:
% - logtheta_all : all learned hyperparameters
% - nl           : Final negative marginal likelihood
%
% Edwin V. Bonilla

%% This can be changed: See minimize function for details
niter     = 1000; % setting for minimize function: number of function evaluations


% ************* Learning Hyperparameters here *******************************
logtheta0 = logtheta_all(deriv_range); % selects hyper-parameters to optimize
[covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train] = deal(data{:});
[logtheta nl] = minimize(logtheta0,'nmargl_mtgp',niter, logtheta_all, ...
			 covfunc_x, xtrain, ytrain,...
			 M, irank, nx, ind_kf_train, ind_kx_train, deriv_range);

%% Update whole vector of parameters with learned ones
logtheta_all(deriv_range) = logtheta;



return;



