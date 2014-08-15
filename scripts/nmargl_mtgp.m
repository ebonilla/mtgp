function [nl, gradnl] = nmargl_mtgp(logtheta, logtheta_all, covfunc_x, x, y,...
				      m, irank, nx, ind_kf, ind_kx, deriv_range)
% Marginal likelihood and its gradients for multi-task Gaussian Processes
% 
% [nl, gradnl] = nmargl_mtgp(logtheta, logtheta_all, covfunc_x, x, y,...
%                	      m, irank, nx, ind_kf, ind_kx, deriv_range)
%
% To be used in conjunction with Carl Rasmussen's minimize function
% and the gpml package http://www.gaussianprocess.org/gpml/code/matlab/doc/
%
% nl = nmargl_mtgp(logtheta, ...) Returns the marginal negative log-likelihood
% [nl gradnl] =  nmargl_mtgp(logtheta, ...) Returns the gradients wrt logtheta
%
% logtheta    : Column vector of initial values of parameters to be optimized
% logtheta_all: Vector of all parameters: [theta_lf; theta_x; sigma_l]
%                - theta_lf: the parameter vector of the
%                   cholesky decomposition of k_f
%                - theta_x: the parameters of K^x
%                - sigma_l: The log of the noise std deviations for each task
% covfunc_x   : Name of covariance function on input space x
% x           : Unique input points on all tasks 
% y           : Vector of target values
% m           : The number of tasks
% irank       : The rank of K^f 
% nx          : number of times each element of y has been observed 
%                usually nx(i)=1 unless the corresponding y is an average
% ind_kx      : Vector containing the indexes of the data-points in x
%                which each observation y corresponds to
% ind_kf      : Vector containing the indexes of the task to which
%                each observation y corresponds to
% deriv_range : The indices of the parameters in logtheta_all
%                to which each element in logtheta corresponds to
%

% Author: Edwin V. Bonilla
% Last update: 23/01/2011


% *** General settings here ****
config = get_mtgp_config();
MIN_NOISE = config.MIN_NOISE;
% ******************************

if ischar(covfunc_x), covfunc_x = cellstr(covfunc_x); end % convert to cell if needed

D = size(x,2); %  Dimensionality to be used by call to covfunc_x
n = length(y); 

logtheta_all(deriv_range) = logtheta;
ltheta_x = eval(feval(covfunc_x{:}));     % number of parameters for input covariance

nlf = irank*(2*m - irank +1)/2;        % number of parameters for Lf
vlf = logtheta_all(1:nlf);             % parameters for Lf

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

% negative log-likelihood
nl = 0.5*y'*alpha + sum(log(diag(L))) + 0.5*n*log(2*pi);
 

if (nargout == 2)                      % If requested, its partial derivatives
  gradnl = zeros(size(logtheta));        % set the size of the derivative vector
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
    
    gradnl(count) =  sum(sum(W.*dK,2),1)/2;
    count = count + 1;
  end % end for derivarives
  
end % end if nargout ==2





 

