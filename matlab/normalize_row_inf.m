function B = normalize_row_inf(A, alpha)
% normalize each row of A such that the norms are equal to alph
[n,d] = size(A);
A_norm = sum(abs(A), 2);
% A_norm(A_norm<=alpha) = alpha; 
% A_norm = repmat(A_norm,1,d);
% B = alph*A./A_norm;
B = spdiags(alpha./max(alpha,A_norm), 0, n, n)*A;
end