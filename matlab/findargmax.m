function argmax  = findargmax(A)
    [vmax,idx] = max(A,[],3);
    argmax = zeros(size(A));
    argmax(:,:,1) = idx==1;
    argmax(:,:,2) = idx==2;
    argmax(:,:,3) = idx==3;
end