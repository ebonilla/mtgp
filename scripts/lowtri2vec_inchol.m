function v = lowtri2vec_inchol(L,D,irank)
% converst a lower triangular matrix into a column vector representation
%
% reads T row-wisely
% irank: intrinsic rank
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last udapte: 23/01/2011
v = [];
for i = 1 : D
  v = [v;L(i,1:min(i,irank))'];
end
