function rgb = visualP(A)
    [nrow,ncol,d] = size(A);
    [value,idx] = max(A,[],3);
    Sky = [128,128,128];
    Building = [128,0,0];
    Pole = [192,192,128];
    Road = [128,64,128];
    Pavement = [60,40,222];
    Tree = [128,128,0];
    SignSymbol = [192,128,128];
    Fence = [64,64,128];
    Car = [64,0,128];
    Pedestrian = [64,64,0];
    Bicyclist = [0,128,192];
    Unlabelled = [0,0,0];

    r = zeros(nrow,ncol);
    g = zeros(nrow,ncol);
    b = zeros(nrow,ncol);
    
    label_colours = [Sky; Building; Pole; Road; Pavement; Tree; SignSymbol; Fence; Car; Pedestrian; Bicyclist; Unlabelled];
    for l=1:12
        r(idx==l) = label_colours(l,1);
        g(idx==l) = label_colours(l,2);
        b(idx==l) = label_colours(l,3);
    end

    rgb = zeros(nrow,ncol,3);
    rgb(:,:,1) = r/255.0;
    rgb(:,:,2) = g/255.0;
    rgb(:,:,3) = b/255.0;
end