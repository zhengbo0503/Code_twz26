function myprint(filename,prnt)
if nargin < 2, prnt = 1; end
if prnt == 1
    set(gcf,'Color','w');
    export_fig(['../figs/' filename '.pdf']);
end