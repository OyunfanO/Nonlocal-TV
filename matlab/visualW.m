function rgb = visualW(A)
    [nrow,ncol,d] = size(A);
    [value,idx] = max(A,[],3);
    Sky = [0,0,0];
    Building = [128,128,128];
    Pole = [255,255,255];

    r = zeros(nrow,ncol);
    g = zeros(nrow,ncol);
    b = zeros(nrow,ncol);
    
    label_colours = [Sky; Building; Pole];
    for l=1:3
        r(idx==l) = label_colours(l,1);
        g(idx==l) = label_colours(l,2);
        b(idx==l) = label_colours(l,3);
    end

    rgb = zeros(nrow,ncol,3);
    rgb(:,:,1) = r/255.0;
    rgb(:,:,2) = g/255.0;
    rgb(:,:,3) = b/255.0;
end