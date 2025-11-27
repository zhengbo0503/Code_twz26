function nrm = scond(A, type, nrmtype)
%SCOND - Diagonal scaled condition number
%	nrm = scond(A,type) computes the condition number of
%	diagonally scaled A.
%	If A is symmetric, then it operates two-sided scaling such that
%	the diagonals of the scaled A are all one, and maintaining symmetry.
%	If A is no symmetric, then it operates column scaling, such that
%	the columns of the scaled A are all have unit 2 norm.
%
%	Arguments:
%		A       : A symmetric or general matrix.
%		type    : 'diagonal' or 'column', 'D' or 'C'. By default, it is
%			determined by whether A is symmetric ('D') or not ('C').
%		nrmtype : by default, is 2. One can pass any type as long as
%			it is recognized by MATLAB `cond`.
%
%	Outputs:
%		nrm  : the scaled condition number. 		

% Check twosided or onesided
if nargin == 1
    % type and nrmtype are not defined
    if (issymmetric(A))
        type = 'D';
        nrmtype = 2;
    else
        type = 'C';
        nrmtype = 2;
    end
elseif nargin == 2
    % nrmtype is not defined
    nrmtype = 2;
end

switch type
  case {'C', 'Column', 'column'}
    D1 = eye(size(A,1));
    D2 = zeros(size(A,2));
    for i = 1:size(A,2)
        D2(i,i) = norm(A(:,i))^(-1);
    end
  case {'D', 'Diagonal', 'diagonal'}
    D1 = diag(diag(A).^(-1/2));
    D2 = D1;
  otherwise
    error("Invalid scaling type, either 'D' or 'C'.")
end

nrm = cond(D1*A*D2, nrmtype); 

end
