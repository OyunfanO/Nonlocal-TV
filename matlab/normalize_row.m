function B = normalize_row(A, alph)
% normalize each row of A such that the norms are equal to alph
n = size(A, 1);
A_norm = sqrt(sum(A.^2, 2));
B = spdiags(alph./max(alph,A_norm), 0, n, n)*A;
end