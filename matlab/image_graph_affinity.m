function W = image_graph_affinity(I)
% IMAGE_GRAPH_AFFINITY tries to compute the affinity matrix for the image I
% I is an m x n image, W is the output which is a mn x mn sparse matrix
[nrow, ncol, d] = size(I);
u = reshape(I, [], d);
rowI = zeros((nrow-2)*(ncol-2)*5, 1);
colI = zeros((nrow-2)*(ncol-2)*5, 1);
val = zeros((nrow-2)*(ncol-2)*5, 1);
idx2=0;
last=idx2;
sigma = 13/255;
for jj=2:(ncol-1)
    for ii=2:(nrow-1)%by coloumn order
        idx = ii + (jj-1)*nrow;%idx is the index in original image
        last = idx2;
        idx2 = ii-1 + (jj-2)*(nrow-2);%idx2 is the index in cropped image
        rowI(5*(idx2-1)+1:5*idx2) = idx;
        colI(5*(idx2-1)+1:5*idx2) = [idx-1; idx+1; idx; idx-nrow; idx+nrow];
        val(5*(idx2-1)+1:5*idx2) = exp(-sum((u([idx-1 idx+1 idx idx-nrow idx+nrow],:) ...
            -u(idx*ones(1,5),:)).^2,2));
    end
end

W = sparse(rowI, colI, val, nrow*ncol, nrow*ncol);
W = max(W, W');