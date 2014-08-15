function plot_confidence_interval(xstar,mu,se,t)
% plots a function with confidence intervals
% mu: vector of mean values of the function
% se: vector of standard errors
% t: value of +/- se to achieve some % confidence
% t=1.96 by default
% Edwin V. Bonilla

% making all vectors column vectors
xstar = xstar(:);
mu = mu(:);
se = se(:);

if (nargin == 3)
  t = 1.96;
end
f = [mu+t*se;flipdim(mu-t*se,1)];
fill([xstar; flipdim(xstar,1)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on; plot(xstar,mu,'k-','LineWidth',2);

return;
    