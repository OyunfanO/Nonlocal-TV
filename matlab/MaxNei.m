path = "image/0.mat";
mat = load(path);
W = mat.W;
W = max(W,W');
k = 0;
[h,w] = size(W);
for ii=1:h
    %non = nonzeros(W(ii,:));
    non = nnz(W(ii,:));
    if k<non
        k = non;
    end
end