function W = image_graph_affinity3(I)
    % IMAGE_GRAPH_AFFINITY tries to compute the affinity matrix for the image I
    % I is an m x n image, W is the output which is a mn x mn sparse matrix

    %default alpha=40, beta=13/255, gamma=3, w1=1,w2=1
    %appearance
    w1 = 1;
    alpha = 40; %sxy
    beta = 10/255; %srgb
    %smoothness
    w2 = 0.8;
    gamma = 3; %sxy
    
    %keep top-k nearest pixel weights, others set to 0
    topk = 20;

    global nrow ncol d;
    [nrow, ncol, d] = size(I);
    W = sparse(nrow*ncol,nrow*ncol);
    
    global radius;
    radius = 2*alpha;
    f_temp = padarray(I,[radius,radius],0,'both');
    
    aCR = 2*alpha; %only consider pixels less than aCR, others set to 0
    aPR = 2*alpha;
    sPR = 3*gamma;
    APW = PositionWeights(alpha,aPR);%output size: [2*aPR+1,2*aPR+1]
%     SPW = PositionWeights(gamma,sPR);
    SPW = PositionWeights(gamma,aPR);
    mSPW = zeros(size(SPW));
    mSPW(aPR+1-sPR:aPR+1+sPR,aPR+1-sPR:aPR+1+sPR) = 1;
    SPW = SPW.*mSPW;
%     for jj=1:ncol
%         for ii=1:nrow
    %parfor kk=1:ncol*nrow
     parfor kk=1:ncol*nrow
        ii = mod(kk,nrow);
        jj = ceil(kk/nrow);
        if ii==0
            ii=nrow;
            %disp(jj)
        end
        %str = sprintf("kk:%d,ii:%d, jj:%d",kk,ii,jj);
        %disp(str)
            %appearance color weights
            sx = max(ii-aCR,1);
            ex = min(ii+aCR,nrow);
            sy = max(jj-aCR,1);
            ey = min(jj+aCR,ncol);
            aCW = ColorWeights(I,beta,ii,jj,sx,sy,ex,ey);
            %appearance position weights
            x1 = aPR+1-(ii-sx);%aPR+1 centre, ii-sx left side length
            x2 = aPR+1+ex-ii;
            y1 = aPR+1-(jj-sy);
            y2 = aPR+1+ey-jj;
            aPW = APW(x1:x2,y1:y2);
            %smoothness position weights
            sPW = SPW(x1:x2,y1:y2);
            
            vval = (w1*aPW.*aCW + w2*sPW)/(w1+w2);             
            %vval = vval.*(vval>0.05);
            %[topval,topidx] = findkmax(vval(:),topk);
            [topval,topidx] = maxk(vval(:),topk);
            reserve = zeros(size(vval));
            reserve(topidx) = 1;        
%             %add four adjacent neighbor
%             [x,y,val] = find(vval>=1);
%             [vrow,vcol] = size(vval);
%             if(x-1>=1)
%                 reserve(x-1,y) = 0.08;
%             end
%             if(y-1>=1)
%                 reserve(x,y-1) = 0.08;
%             end
%             if(x+1<=vrow)
%                 reserve(x+1,y) = 0.08;
%             end
%             if(y+1<=vcol)
%                 reserve(x,y+1) = 0.08;
%             end
            
            vval = vval.*reserve;
            val = sparse(nrow,ncol);
            val(sx:ex,sy:ey) = vval;
            %W(ii+(jj-1)*nrow,:) = sparse(val(:));
            W(kk,:) = val(:);
    end
    W = W';
%     end
   % W = max(W, W');
end

function PW = PositionWeights(sigma,r)
    global radius;
    if r>radius
        sprintf("r should be less than radius");
        return
    end
    [x,y] = meshgrid(-r:r);
    PW = exp(-(x.^2+y.^2)/(2*sigma^2));
end

function CW = ColorWeights(I,sigma,idx,idy,sx,sy,ex,ey) %current pixel I(idx,idy), compute region I[sx:ex,sy:ey]
    temp = I(idx,idy,:);
    fr_temp = I(sx:ex,sy:ey,1);
    fg_temp = I(sx:ex,sy:ey,2);
    fb_temp = I(sx:ex,sy:ey,3);
    dr = fr_temp-temp(1);
    dg = fg_temp-temp(2);
    db = fb_temp-temp(3);
    CW = exp(-(dr.^2+dg.^2+db.^2)/(2*sigma^2));
end

% function CW = ColorWeights(padI,sigma,r,idx,idy)%idx,idy is current pixel location in padI
%     temp = padI(idx,idy,:);
%     fr_temp = padI(idx-r:idx+r,idy-r:idy+r,1);
%     fg_temp = padI(idx-r:idx+r,idy-r:idy+r,2);
%     fb_temp = padI(idx-r:idx+r,idy-r:idy+r,3);
%     dr = fr_temp-temp(1);
%     dg = fg_temp-temp(2);
%     db = fb_temp-temp(3);
%     w2 = exp(-(dr.^2+dg.^2+db.^2)/(2*sigma^2));
%     global nrow ncol d radius;
%     if r>radius
%         sprintf("r should be less than radius");
%         return
%     end
%     CW = zeros(nrow+2*radius,ncol+2*radius);
%     CW(idx-r:idx+r,idy-r:idy+r) = w2;
% end