function Q = div_w(P, W)
% div_w computes the non-local divergence of a vector field P, whose
% columns are n-dimensional sparse vectors
% W is the entry-wise square root of the Affinity matrix
% Q1 = W*P';
% Q2 = W*P;
% Q = - diag(Q1) + diag(Q2);
Q = W.*P;
Q = sum(Q - Q',2);