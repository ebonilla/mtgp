function L = vec2lowtri_inchol(v,D,irank)
% converts the column  vector v into an incomplete lower triangular matrix
% of dimensions DxD.
%
% D = (-1 + sqrt(8*length(v)+1))/2;
%
% irank: intrinsic rank
%
% D = (-1 + sqrt(8*length(v)+1))/2;
% fills D row-wisely
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last udapte: 23/01/2011

L = zeros(D);
low = 1;
for i = 1 : D
  up = min(irank,i);
  L(i,1:up) = v(low:low+up-1);
  low = low + up;
end



