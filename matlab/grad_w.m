function DPhi = grad_w(Phi, W)
% grad_w computes the non-local gradient of a function Phi defined on the
% vertices of a graph
% W is the entry-wise square root of the affinity matrix of the graph

% [I, J, K] = find(W);
% N = size(I,1);
% n = size(Phi, 1);
% ID = zeros(N, 1);
% JD = zeros(N, 1);
% KD = zeros(N, 1);
% for ii=1:N
%     x = I(ii);
%     y = J(ii);
%     w = K(ii);
%     ID(ii) = y;
%     JD(ii) = x;
%     KD(ii) = w * (Phi(y) - Phi(x));
% end
% DPhi = sparse(ID, JD, KD, n, n);

n=size(Phi,1);
diagPhi = spdiags(Phi, 0, n, n);
p1=W*diagPhi;
p2=diagPhi*W;
DPhi=p1-p2;