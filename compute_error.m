function [f, r, oU, oV, errV, relgap] = compute_error(A, U, S, V)

errV = zeros(size(A,2),1); 
r = norm(A - U*S*V','fro')/norm(A,'fro');
oU = norm(U'*U - eye(size(U,2)), inf);
oV = norm(V'*V - eye(size(V,2)), inf);

% Forward error needs special treatment
Aref = mp(A,64);
[~,Sref,Vref] = svd(Aref, 'econ');
sref = sort(diag(Sref),'descend');
sref = reshape(sref, length(sref), 1);
s = sort(diag(S), 'descend');
for (i = 1:length(s))
    if (s(i) < eps('double')/2)
        s = s(1:i-1); 
        break;
    end
end
s = reshape(s, length(s), 1);
f = max(abs(sref - s)./abs(sref));
for i = 1:size(A,2)
    errV(i) = norm(Vref(:,i)-V(:,i)*(V(:,i)'*Vref(:,i)),2); 
end

if nargout == 6
    % relgap = sval_gap(s, sref); 
    relgap = sval_gap(sref); 
end

end
