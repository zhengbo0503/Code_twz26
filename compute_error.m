function [f, r, oU, oV] = compute_error(A, U, S, V)
%COMPUTE_ERROR - Computing error for our testing scripts 
%
%   Usage:
%       err = COMPUTE_ERROR(A, U, S, V)
%
%   Purpose:
%       Compute several errors for the computed singular value
%       decomposition. The algorithm will first compute the reference
%    	singular values in quadruple precision and then compute the
%		following errors:
%		1. Relative forward error
%			max_k {|sigma_k - sigma_ref_k|/sigma_ref_k}
%		2. Backward error:
%			||A-U*S*V'||_F
%		3. Orthogonality errors
%			||U'*U - I||_inf and ||V'*V - I||_inf
%
%   Arguments: 
%		(1) G - Real, double matrix
%	       The input matrix to the SVD algorithm.
%		
%       (2,3,4) U, S, V - Real, double matrices
%   	    The computed singular value decomposition
%
%   Output: 
%		(1) f - Real, double scalar
%			Maximum relative forward error
%		(2) r - Real, double scalar
%			Backward error
%	    (3) oU - Real, double scalar
%			Orthogonality error of the left singular vectors
%	    (4) oV - Real, double scalar
%			Orthogonality error of the right singular vectors
%		
%   Author: 
%       Zhengbo Zhou, Manchester, UK, Dec 2025
%

r = norm(A - U*S*V','fro')/norm(A,'fro');
oU = norm(U'*U - eye(size(U,2)), inf);
oV = norm(V'*V - eye(size(V,2)), inf);

% Forward error needs special treatment
Aref = mp(A,34);
[~,Sref,~] = svd(Aref, 'econ');
sref = sort(diag(Sref),'descend');
sref = reshape(sref, length(sref), 1);

s = sort(diag(S), 'descend');
s = reshape(s, length(s), 1);

f = max(abs(sref - s)./abs(sref));
end
