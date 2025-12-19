function A = blur_matrix()
    [A,~,~] = blur(10, 5, 1.4);
    A = full(A); 
end