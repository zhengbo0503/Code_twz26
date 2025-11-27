%TEST2 -- timing test
%	This is a test script for accessing the speed of four different algorithms
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
t_mp = zeros(length(m),1);
t_j = zeros(length(m),1);
t_qrj = zeros(length(m),1);
t_matlab = zeros(length(m),1);

for i = 1:length(m)

    mm = m(i);
    nn = n;
    A = gallery('randsvd', [mm,nn], 1e6);

    % [U,S,V,nos,time] =
    f1 = @() mposj(A);
    f1(); 
    tmp1 = timeit(f1, 3);
    tmp2 = timeit(f1, 3);
    t_mp(i) = (tmp1 + tmp2)/2;
    
    % [Uj1,Sj1,Vj1,svaj1,workj1,infoj1] =
    f2 = @() dgesvj_mex(A,'G','U','V',nn,eye(nn),max(6,mm+nn));
    f2(); 
    tmp1 = timeit(f2, 6);
    tmp2 = timeit(f2, 6);
    t_j(i) = (tmp1+tmp2)/2;

    % [Uj2,Sj2,Vj2,svaj2,workj2,iworkj2,infoj2] =
    f3 = @() dgejsv_mex(A,'C','U','V','R','N','N');
    f3(); 
    tmp1 = timeit(f3, 7);
    tmp2 = timeit(f3, 7);
    t_qrj(i) = (tmp1+tmp2)/2;

    % [Uj3,Sj3,Vj3] =
    f4 = @() svd(A,'econ');
    f4(); 
    tmp1 = timeit(f4, 3);
    tmp2 = timeit(f4, 3);
    t_matlab(i) = (tmp1+tmp2)/2;

    fprintf("Finished %d of %d \n", i, length(m));
    
end

if savedata == 1
    save('data/timing.mat');
end

%%
close all; 
set(gcf, 'Units','inches', 'Position',[1 1 7 7]);  % [left bottom width height]

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

semilogx(m, t_mp,'LineStyle','-','Marker','*'); hold on;
semilogx(m, t_j,'LineStyle','-','Marker','x');
semilogx(m, t_qrj,'LineStyle','-','Marker','square');
semilogx(m, t_matlab,'LineStyle','-','Marker','diamond');

axis square
set(findall(gcf, 'Type', 'Line'), 'LineWidth', 2);
xlabel('Number of row'); 
ylabel('Runtime'); 
xticks([500,1000,1750,3000]); 

legend('MP3JacobiSVD', ...
    '\texttt{DGESVJ}', ...
    '\texttt{DGEJSV}', ...
    'MATLAB \texttt{svd}', ...
    'Interpreter','latex', 'Location', 'northwest');

%%
if printout == 1
    myprint('timing');
end