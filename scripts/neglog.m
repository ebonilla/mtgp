function out1 = neglog(logtheta, logtheta_all, covfunc_x, x, y,...
				      m, irank, nx, ind_kf, ind_kx, deriv_range)
% Negative (maginal) log-likelihood of  multi-task Gaussian Process 
%
% Useful for testing the gradients or for using an optimization method 
% different to minimize.m
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last udapte: 23/01/2011


% *** General settings here ****
config = get_mtgp_config();
MIN_NOISE = config.MIN_NOISE;
% ******************************


if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

[N, D] = size(x); % N: number of data-points per task
n = length(y); 

ltheta = length(logtheta_all);         % total number of parameters
logtheta_all(deriv_range) = logtheta;
ltheta_x = eval(feval(covfunc_x{:}));     % number of parameters for input covariance

nlf = irank*(2*m - irank +1)/2;        % number of parameters for Lf
vlf = logtheta_all(1:nlf);             % parameters for Lf

% NEW
theta_lf = vlf; %theta_lf = tanh(vlf); 
Lf = vec2lowtri_inchol(theta_lf,m,irank);

theta_x = logtheta_all(nlf+1:nlf+ltheta_x);                     % cov_x parameters
sigma2n = exp(2*logtheta_all(nlf+ltheta_x+1:end));              % Noise parameters
Sigma2n = diag(sigma2n);                                        % Noise Matrix
Var_nx = diag(1./nx);

Kx = feval(covfunc_x{:}, theta_x, x);
Kf = Lf*Lf';
K = Kf(ind_kf,ind_kf).*Kx(ind_kx,ind_kx);
K = K + ( Sigma2n(ind_kf,ind_kf) .*Var_nx ); 
Sigma_noise = MIN_NOISE*eye(n);
K = K + Sigma_noise;
 
L = chol(K)';                        % cholesky factorization of the covariance
alpha = solve_chol(L',y);

% negative log-likelihood
out1 = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);

