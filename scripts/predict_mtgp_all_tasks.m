function [ Ypred, Vpred ] = predict_mtgp_all_tasks(logtheta_all, data, xtest )
%PREDICT_MTGP_ALL_TASKS Makes predictions at all points xtest for all tasks
%
% INPUT:
% - logtheta_all : all hyperparameters
% - data         : cell data in the order 
%                  [covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train]
% - xtest        : Test points 
%
% OUTPUT
% - YPred : (Ntest x M) Matrix of Mean MTGP Predictions 
% - Vpred : (Ntest x M) Matrix of MTGP Variances
%           Where M is the number of tasks and Ntest: number of test points            
%
%   Edwin V. Bonilla

[covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train] = deal(data{:});


Ntest = size(xtest,1);
[alpha, Kf, L, Kxstar, Kss] = alpha_mtgp(logtheta_all, covfunc_x, xtrain, ytrain,...
				    M, irank, nx, ind_kf_train, ...
				    ind_kx_train, xtest);
all_Kxstar = Kxstar(ind_kx_train,:);
Ypred = zeros(Ntest,M);
Vpred = zeros(Ntest,M);
for task = 1 : M
  Kf_task = Kf(ind_kf_train,task);
  Kstar          = repmat(Kf_task,1,Ntest).*all_Kxstar;
  Ypred(:,task)  = Kstar'*alpha;
  v              = L\Kstar;
  Vpred(:,task)  = Kf(task,task)*Kss - sum(v.*v)';
end



return;


