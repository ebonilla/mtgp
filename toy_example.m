% toy_example.m
% A toy example demonstrating how to use the mtgp package  for M=3 tasks 
%
% function toy_example()
% 
% 1. Generates sample from true MTGP model 
% 2. Assigns cell data for learning and prediction
% 3. Learns hyperparameters with minimize function
% 4. Makes predictions at all points on all tasks
% 5. Plots Data and predictions
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
clear all; clc;
rand('state',18);
randn('state',20);
 

PLOT_DATA       = 1;
covfunc_x       = {'covSEard'};
M               = 3;    % Number of tasks
D               = 2;    % Dimensionality of input spacce
irank           = M;    % rank for Kf (1, ... M). irank=M -> Full rank


%% 1. Generating samples from true Model
[x, Y, xtrain, ytrain, ind_kf_train, ind_kx_train , nx] = generate_data(covfunc_x, D, M);

%% 2. Assigns cell data for learning and prediction
data  = {covfunc_x, xtrain, ytrain, M, irank, nx, ind_kf_train, ind_kx_train};

%% 3. Hyper-parameter learning
 [logtheta_all deriv_range] = init_mtgp_default(xtrain, covfunc_x, M, irank);
[logtheta_all nl]           = learn_mtgp(logtheta_all, deriv_range, data);


%% 4. Making predictions at all points on all tasks
[ Ypred, Vpred ] = predict_mtgp_all_tasks(logtheta_all, data, x );


%% 5. Plotting the predictions
if (PLOT_DATA)
    plot_predictions(x,  Y, D, Ypred, Vpred, ind_kf_train, ind_kx_train);
    figure, plot(nl); title('Negative Marginal log-likelihood');
end








