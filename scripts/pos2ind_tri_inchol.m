function [i j] = pos2ind_tri_inchol(pos,D,irank)
% translates a position of a vector that reads a triangual matrix
% row-wisely into  an index (i,j) of the matrix
%
% irank: the instrinc rank
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last udapte: 23/01/2011

i = 0;
s(1) = 0;
j = 1;
while (s < pos)
  i = i + 1;
  s(i+1) = s(i) + min(i,irank);
end

if (i > 0)
  j = pos - s(i);
end

