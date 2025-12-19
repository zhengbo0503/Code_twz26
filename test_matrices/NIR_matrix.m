function A = NIR_matrix()
% NIR of Corn Samples 
% This data set consists of 80 samples of corn measured on instrument mp5 
    warning('off', 'MATLAB:unknownObjectNowStruct');
    G = load("test_matrices/corn.mat");
    A = G.mp5spec.data;
    A = A'; 
end