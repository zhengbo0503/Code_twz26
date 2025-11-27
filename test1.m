clc; clear; rng(1);

% Arguments
M = ones(20,1)*1000;
N = round(logspace(1,3,20)); 

for i = 1:length(M)
    % Set parameters
    m = M(i);
    n = N(i);
    
    % Setup matrices, preconditioners
    A = gallery('randsvd', [m,n], 1e10, 3); % Random matrix
	[~,~,Vs] = svd(single(A));
	Vd = double(Vs);
	[Vt,~] = qr(Vd,'econ'); 
    
    % Way 1 -- preconditioning then QR
    % Preprocessing
    At1 = A*Vt; 
    [Qt1,Rt1] = qr(At1,'econ');
    % Jacobi
    % A,joba,jobu,jobv,mv,V0,lwork
    % [Aout,S,V,sva,work,info]
    [UR1,S1,VR1,sva1,work1,info1] = ...
        dgesvj_mex(Rt1, 'U', 'U', 'V', size(Rt1,1), [], size(Rt1,1)+size(Rt1,2), []);
    if info1 ~= 0
        warn("Jacobi on way 1 not converged\n");
        return
    end
    nos1(i) = work1(4);
    % Postprocessing
    U1 = Qt1*UR1;
    V1 = Vt*VR1;
    % Compute error
    [f1(i), r1(i), oU1(i), oV1(i)] = compute_error(A, U1, S1, V1);


    % Way 2 -- QR then preconditioning
    % preprocessing
    [Q2,R2] = qr(A,'econ');
    Rt2 = R2*Vt;
    % Jacobi
    [UR2,S2,VR2,sva2,work2,info2] = ...
        dgesvj_mex(Rt2, 'G', 'U', 'V', size(Rt2,1), [], size(Rt2,1)+size(Rt2,2), []);
    if info2 ~= 0
        warn("Jacobi on way 2 not converged\n");
        return
    end
    nos2(i) = work2(4); 
    % Postprocessing
    U2 = Q2*UR2;
    V2 = Vt*VR2;
    % Compute error
    [f2(i), r2(i), oU2(i), oV2(i)] = compute_error(A, U2, S2, V2);

    % Printout
    fprintf("Iteration %d of %d\n", i, length(M)); 

end

%% Plot

figure(1)
semilogy(N, f1,'x'); hold on;
loglog(N, f2,'o');
legend('precondition -- QR', 'QR -- precondition', location='northwest')
title('fwd error');
xlabel('No. of cols')
axis square

figure(2)
loglog(N, r1,'x'); hold on;
loglog(N, r2,'o');
legend('precondition -- QR', 'QR -- precondition', location='northwest')
title('residual error');
xlabel('No. of cols')
axis square

figure(3)
loglog(N, max(oU1, oV1),'x'); hold on;
loglog(N, max(oU2, oV2),'o');
legend('precondition -- QR', 'QR -- precondition', location='northwest')
title('orth error');
xlabel('No. of cols')
axis square

figure(4)
semilogx(N, nos1,'x'); hold on;
semilogx(N, nos2,'o');
legend('precondition -- QR', 'QR -- precondition', location='northwest')
title('no. of sweeps');
xlabel('No. of cols')
axis square
