%TEST5 - mposj_ssd with (single, single, double)
%	compared with SGESVJ and SGEJSV

clc; close all; clear; rng(1);

m = 3000*ones(10,1); 
n = round(logspace(2, log10(3000), 10));
epsln = eps('single')/2;
time_mposj = zeros(length(m),1);
time_sgesvj = zeros(length(m),1);
time_sgejsv = zeros(length(m),1);
time_matlab = zeros(length(m),1);

for i = 1:length(m)

    mm = m(i);
    nn = n(i);
    A = gallery('randsvd', [mm,nn], 1e8, 3, 'single');

    % Our algorithm 
    tic_mposj_tmp1 = tic;
    [~,~,~,~,~] = mposj_ssd(A);
    tic_mposj_tmp1 = toc(tic_mposj_tmp1);
    tic_mposj_tmp2 = tic;
    [U1,S1,V1,nos,scnd] = mposj_ssd(A);
    tic_mposj_tmp2 = toc(tic_mposj_tmp2);
    time_mposj(i) = (tic_mposj_tmp1 + tic_mposj_tmp2)/2;
    [f1(i),~,~,~] = compute_error_temp(A, U1, S1, V1);
    
    % SGESVJ (plain Jacobi)
    tic_sgesvj_tmp1 = tic;
    [~,~,~,~,~,~] = sgesvj_mex(A,'G','U','V',nn,eye(nn,'single'),max(6,mm+nn));
    tic_sgesvj_tmp1 = toc(tic_sgesvj_tmp1);
    tic_sgesvj_tmp2 = tic;
    [U2,S2,V2,sva2,work2,info2] = sgesvj_mex(A,'G','U','V',nn,eye(nn,'single'),max(6,mm+nn));
    tic_sgesvj_tmp2 = toc(tic_sgesvj_tmp2);
    if info2 ~= 0
        fprintf("Error: SGESVJ does not converge.\n");
        break;
    end
    time_sgesvj(i) = (tic_sgesvj_tmp1 + tic_sgesvj_tmp2)/2;
    [f2(i),~,~,~] = compute_error_temp(A, U2, S2, V2);

    % DGEJSV (preconditioned Jacobi)
    tic_sgejsv_tmp1 = tic;
    sgejsv_mex(A,'C','U','V','R','N','N');
    tic_sgejsv_tmp1 = toc(tic_sgejsv_tmp1);
    tic_sgejsv_tmp2 = tic;
    [U3,S3,V3,sva3,work3,iwork3,info3] = sgejsv_mex(A,'C','U','V','R','N','N');
    tic_sgejsv_tmp2 = toc(tic_sgejsv_tmp2);
    if info2 ~= 0
        fprintf("Error: SGEJSV does not converge.\n");
        break;
    end
    time_sgejsv(i) = (tic_sgejsv_tmp1 + tic_sgejsv_tmp2)/2;
    [f3(i),~,~,~] = compute_error_temp(A, U3, S3, V3);


    bound2(i) = scnd * sqrt(mm * nn) * epsln;

    fprintf("Finished %d of %d \n", i, length(n));

end

savedata = 1;
if savedata == 1
    save("./data/timing_ssd.mat")
end
%% 
close all;
C1 = "#1171BE";
C2 = "#DD5400";
C3 = "#EDB120";
C4 = "#3BAA32";

figure(1)

loglog(n,f1,'LineStyle','none','Marker','*','Color',C1);
hold on;
loglog(n,f2,'LineStyle','none','Marker','pentagram','Color',C2);
loglog(n,f3,'LineStyle','none','Marker','square','Color',C3);
loglog(n,bound2,'LineStyle',':','Marker','none','Color','k');

legend('MP3JacobiSVD', 'SGESVJ', 'SGEJSV', 'Bound');
set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1);
xlabel('$n$', 'FontSize', 10);
ylabel('$\mathrm{max}_k {\varepsilon}^{(k)}_{fwd}$', 'FontSize', 10);
ylim([1e-7, 10]);

figure(2)
loglog(n, time_mposj,'LineStyle','none','Marker','*','Color',C1); hold on;
loglog(n, time_sgesvj,'LineStyle','none','Marker','pentagram','Color',C2);
loglog(n, time_sgejsv,'LineStyle','none','Marker','square','Color',C3);
legend('MP3JacobiSVD', 'SGESVJ', 'SGEJSV')

set(findall(gcf, 'Type', 'Line'), 'LineWidth', 1);
xlabel('Number of columns', 'FontSize', 10); 
ylabel('Runtime (sec)', 'FontSize', 10); 

%%
function [f, r, oU, oV, errV, relgap] = compute_error_temp(A, U, S, V)

errV = zeros(size(A,2),1); 
r = norm(A - U*S*V','fro')/norm(A,'fro');
oU = norm(U'*U - eye(size(U,2)), inf);
oV = norm(V'*V - eye(size(V,2)), inf);

% Forward error needs special treatment
Aref = double(A);
[~,Sref,~] = svd(Aref, 'econ');
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

if nargout == 6
    % relgap = sval_gap(s, sref); 
    relgap = sval_gap(sref); 
end

end
