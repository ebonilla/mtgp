function out2 = grad_neglog(logtheta, logtheta_all, covfunc_x, x, y,...
				      m, irank, nx, ind_kf, ind_kx, deriv_range)
% Gradient of the negative (marginal) log-likelihood of  multi-task Gaussian Process 
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
theta_lf = vlf; 
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


out2 = zeros(size(logtheta));        % set the size of the derivative vector
W = L'\(L\eye(n))-alpha*alpha';      % precompute for convenience

count = 1;
for zz = 1 : length(deriv_range) 
  z = deriv_range(zz);
  
  if ( z <= nlf )                          % Gradient wrt  Kf
    [o p] = pos2ind_tri_inchol(z,m,irank); % determines row and column
    J = zeros(m,m); J(o,p) = 1;
    Val = J*Lf' + Lf*J';
    dK = Val(ind_kf,ind_kf).*Kx(ind_kx,ind_kx);
    
  elseif ( z <= (nlf+ltheta_x) )           % Gradient wrt parameters of Kx
    z_x =  z - nlf;
    dKx = feval(covfunc_x{:},theta_x, x, z_x);      
    dK = Kf(ind_kf,ind_kf).*dKx(ind_kx,ind_kx);
    
  elseif ( z >= (nlf+ltheta_x+1) )         % Gradient wrt Noise variances
    Val = zeros(m,m);
    kk = z - nlf - ltheta_x;
    Val(kk,kk) = 2*Sigma2n(kk,kk);
    dK = Val(ind_kf,ind_kf).*Var_nx;
    
  end % endif z
  
  out2(count) =  sum(sum(W.*dK,2),1)/2;
  count = count + 1;
end % end for derivarives








