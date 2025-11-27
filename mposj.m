function [U,S,V,nos,scalecond] = mposj(G, nop)
%MPOSJ - Mixed precision one-sided Jacobi algorithm 
%
%   Usage:
%       [U,S,V] = mposj(G)
%       [U,S,V,nos] = mposj(G)
%       [U,S,V,nos,scalecond] = mposj(G, nop)
%
%   Purpose: 
%       MPOSJ computes a SVD of a general matrix based on LAPACK routine
%       DGESVJ, the QR-preconditioned Jacobi SVD algorithm. 
%       MPOSJ first computes a preconditioner at single precision, applies
%       the preconditioner at double or quadruple precision, and finally
%       performs DGESVJ on the preconditioned matrix. 
%       For ill-conditioned matrix, MPOSJ can compute singular values with
%       smaller relative forward error compared to MATLAB function svd.
%
%   Arguments: 
%       (1) G - Real, double matrix.
%       (2) nop - Integer, and by default 3.
%           If nop = 3, MPOSJ will use quadruple precision to apply the
%           preconditioner; If nop = 2, MPOSJ will use double precision
%           instead. 
%   
%   Outputs:
%       (1) U,S,V - Real, double matrix.
%           Suppose G is mxn, then U is mxn whose columns are numerically
%           orthonormal, S is a nxn diagonal matrix whose diagonal entries
%           are singular values of G, and V is a nxn numerically orthogonal
%           matrix. 
%       (2) nos - Integer.
%           Number of sweep used by DGESVJ.
%       (3) scalecond - Real, double.
%           Scaled condition number of the preconditioned matrix. Useful
%           for some posterior analysis. 
%
%   Author:
%       Zhengbo Zhou, Manchester, UK, 2025 Oct.
%

[m,n] = size(G);
if m < n   
    [Vt,St,Ut,nos,scalecond] = mposj(G');
    U = Ut; S = St; V = Vt;
    return
end

if nargin == 1
    nop = 3; % Use MP3SVDJacobi by default.
end
if (nop ~= 2) && (nop ~= 3)
    error("The number of precision (the second argument) should be 2 or 3.");
end
idty = eye(n);
doqr = (m >= (11*n)/6); % Consistent with LAPACK choice. 

% Compute the preconditioner
[~,~,Vs] = svd(single(G),'econ');
[Vd,~] = qr(double(Vs));

% Apply the preconditioner
if nop == 3
    Gt = double(mp(G)*mp(Vd));
else
    Gt = G*Vd; 
end

% Output scaled condition number for posterior analysis. 
if nargout == 5
    scalecond = scond(Gt); 
end

% Apply the one-sided Jacobi
if doqr % Apply QR factorization before doing Jacobi SVD
    [Qgt,Rt] = qr(Gt,'econ');
    optlwork = max(6,m+n);
    work = zeros(optlwork,1);
    [U,S,V,~,work,info] = dgesvj_mex(Rt,'U','U','V',n,idty,optlwork,work);
    nos = work(4);
    U = Qgt*U;
else % Plain Jacobi SVD
    optlwork = max(6,m+n);
    work = zeros(optlwork,1);
    [U,S,V,~,work,info] = dgesvj_mex(Gt,'G','U','V',n,idty,optlwork,work);
    nos = work(4);
end

if info < 0
    error("DGESVJ has invalid inputs.\n");
elseif info > 0
    warning("DGESVJ does not converged in 30 iterations.\n")
end

V = Vd*V;
end
