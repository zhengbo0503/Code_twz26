%TEST3 -- forward accuracy test, changing matrix size
%	This is a test script for accessing the forward accuracy of four different algorithms
%	that relates to the one-sided Jacobi algorithm
%	 - mixed-precision preconditioned Jacobi algorithm (mposj) : Our algorithm,
%	 - DGESVJ : LAPACK routine for the one-sided Jacobi algorithm, and
%	 - DGEJSV : LAPACK routine for the QR-preconditioned one-sided Jacobi algorithm.
%	 - svd : MATLAB function for computing the SVD.

clc; close all; clear; rng(1);
savedata = 1;
printout = 0;

n = 400;
m = round(logspace(log10(500), log10(3000), 20));
epsln = eps('double')/2;
kappa = 1e8;
f1 = zeros(length(m), 5);
f2 = zeros(length(m), 5);
f3 = zeros(length(m), 5);
f4 = zeros(length(m), 5);
bound1 = zeros(length(m),5);
bound2 = zeros(length(m),5);

for mode = 1:5

    for i = 1:length(m)

        mm = m(i);
        nn = n;
        A = gallery('randsvd', [mm,nn], kappa, mode);

        [U1,S1,V1,nos1,scnd] = mposj(A);
        [f1(i,mode),~,~,~] = compute_error(A, U1, S1, V1);

        [U2,S2,V2,sva2,work2,info2] = dgesvj_mex(A,'G','U','V',nn,eye(nn),max(6,mm+nn));
        if info2 ~= 0
            fprintf("Error: DGESVJ does not converge.\n");
            break;
        end
        [f2(i,mode),~,~,~] = compute_error(A, U2, S2, V2);

        [U3,S3,V3,sva3,work3,iwork3,info3] = dgejsv_mex(A,'C','U','V','R','N','N');
        if info3 ~= 0
            fprintf("Error: DGEJSV does not converge.\n");
            break;
        end
        [f3(i,mode),~,~,~] = compute_error(A, U3, S3, V3);

        [U4,S4,V4] = svd(A,'econ');
        [f4(i,mode),~,~,~] = compute_error(A, U4, S4, V4);

        bound1(i,mode) = scond(A) * sqrt(mm * nn)* epsln;
        bound2(i,mode) = scnd * sqrt(mm * nn) * epsln;

        fprintf("Finished MODE = %d, %d of %d\n", mode, i, length(m));

    end
end

if savedata == 1
    save("./data/fwderr_diff_size.mat")
end


%% 
close all; 
figure(99);
set(gcf, 'Units','inches', 'Position',[1 1 11 17.5]);  % [left bottom width height]

okabeito = [
    0.000 0.000 0.000  % black
    0.902 0.624 0.000  % orange
    0.337 0.706 0.914  % sky blue
    0.000 0.620 0.451  % bluish green
    0.941 0.894 0.259  % yellow
    0.000 0.447 0.698  % blue
    0.835 0.369 0.000  % vermillion
    0.800 0.475 0.655  % reddish purple
];

for mode = 1:5
    subplot(3,2,mode);
    ax = gca; 
    ax.ColorOrder = okabeito;
    
    %%%
    loglog(m,f1(:,mode),'LineStyle','-','Marker','*');
    hold on;
    loglog(m,f2(:,mode),'LineStyle','-','Marker','x');
    loglog(m,f3(:,mode),'LineStyle','-','Marker','square');
    loglog(m,f4(:,mode),'LineStyle','-','Marker','diamond');
    loglog(m,bound1(:,mode),'LineStyle','--','Marker','^','Color','k');
    loglog(m,bound2(:,mode),'LineStyle','-.','Marker','o','Color','k');
    %%%

    axis square
    xticks([500, 1000, 1750, 3000]);
    xlabel('Number of row');
    set(findall(gcf, 'Type', 'Line'), 'LineWidth', 2);

    % Set the title 
    alphabet = ['a','b','c','d','e'];
    t = sprintf('(%s) MODE = %d', alphabet(mode), mode); 
    title(t, 'FontWeight', 'normal'); 

    % Only get ylabel if the subplot is on the left
    if mod(mode,2) == 1
        ylabel('$\mathrm{max}_k {\varepsilon}^{(k)}_{fwd}$', ...
            'Interpreter', 'latex');
    end

    % Set legend for the fifth one
    if mode == 5
        L = legend('MP3JacobiSVD', ...
            '\texttt{DGESVJ}', ...
            '\texttt{DGEJSV}', ...
            'MATLAB \texttt{svd}', ...
            '$(mn)^{1/2}u\kappa_2^S(A)$', ...
            '$(mn)^{1/2}u\kappa_2^S(\tilde{A})$', ...
            'Interpreter','latex');
        
        ax(6) = subplot(3,2,6);
        set(ax(6),'Visible','off'); box(ax(6),'off');

        set([L ax(6)],'Units','normalized');
        lp = L.Position;       
        tp = ax(6).Position;    
        lp(1) = tp(1) + (tp(3)-lp(3))/2;   
        lp(2) = tp(2) + (tp(4)-lp(4))/2;   
        L.Position = lp;
    end
end

%%
if printout == 1
    myprint('fwderr_diff_size_cond_1e8', 1);
end