%imagepath = 'npy/10.png';
%imagepath = 'CamVid/0001TP_009180.png';
imagepath = 'CamVid/0001TP_008580.png';
%imagepath = 'CamVid/Seq05VD_f00000.png';
%imagepath = 'CamVid/Seq05VD_f05070.png';
%imagepath = 'CamVid/Seq05VD_f05100.png';
npystr = strrep(imagepath,'png','npy');
%npystr = strrep(npystr,'image','NL_b3_20k1');
npy = readNPY(npystr);
gtstr = strrep(imagepath,'.png','_gt.png');
%gtstr = strrep(gtstr,'image','gt');
gtstr = strrep(gtstr,'npy','gt');
gt = imread(gtstr);

O_k = permute(npy,[2,3,1]);

II = imread(imagepath);
I = double(II);
I = I/max(I(:));

global n_clusters nrow ncol
[nrow, ncol, d] = size(I);
[nrow_o,ncol_o,d_o] = size(O_k);
n_clusters= d_o;
% 
% delete(gcp('nocreate'));
% parpool(4);

tic;
W_type = 'n4';
comix = 0.8;
lam = 2;
tau = 0.1;
iter = 500;
W = image_graph_affinity2(double(I));
if strcmp(W_type,'n4')
    %W_mat = load(strrep(imagepath,'.png','_n4.mat'));
%     W_mat = load('CamVid/n4.mat');
%     W = W_mat.W;
    W = image_graph_affinity2(double(I));
elseif strcmp(W_type,'CRF')
    %W = image_graph_affinity3(double(I));
    
     W_mat = load(strrep(imagepath,'.png','_k100.mat'));
     W = W_mat.W;
% %     parfor ii=1:nrow*ncol
% %         [val,index] = maxk(W(ii,:),10);
% %         Wrow = zeros(1,nrow*ncol);
% %         Wrow(index) = full(val);
% %         W(ii,:) = sparse(Wrow);
% %     end
%     W_mat = load('8_NL100.mat');
     %W = W_mat.W;
     W = max(W,W');
elseif strcmp(W_type,'mix')
    %W1_mat = load(strrep(imagepath,'.png','_n4.mat'));
    W1_mat = load('CamVid/n4.mat');
    W_n4 = W1_mat.W;
    %W_CRF = image_graph_affinity3(double(I));
     W2_mat = load(strrep(imagepath,'.png','_k100.mat'));
     W_CRF = W2_mat.W;
%     parfor ii=1:nrow*ncol
%         [val,index] = maxk(W_CRF(ii,:),10);
%         Wrow = zeros(1,nrow*ncol);
%         Wrow(index) = full(val);
%         W_CRF(ii,:) = sparse(Wrow);
%     end
    %W_CRF = max(W_CRF,W_CRF');
    n4idx = W_n4>0;
    W = (W_CRF-W_CRF.*n4idx) + comix*W_n4;   
end
toc;
%W = sign(W);
%npy1 = squeeze(npy(1,:,:));


O = reshape(O_k,[],n_clusters);

A = softmax(O,2);

tic;
[ro,ep,ed] = alternative_update(O,A,W,lam,tau,iter);
toc;

rA = reshape(A,nrow,ncol,d_o);
ro = reshape(ro,nrow,ncol,d_o);

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
label_colours = [Sky; Building; Pole; Road; Pavement; Tree; SignSymbol; Fence; Car; Pedestrian; Bicyclist; Unlabelled];

[vmax,idx] = max(rA,[],3);
argmaxrA = zeros(nrow,ncol,3);
argmax1 = zeros(nrow,ncol);
argmax2 = zeros(nrow,ncol);
argmax3 = zeros(nrow,ncol);
for ii=1:12
    argmax1(idx==ii) = label_colours(ii,1);
    argmax2(idx==ii) = label_colours(ii,2);
    argmax3(idx==ii) = label_colours(ii,3);
    argmaxrA(:,:,1) = argmax1;
    argmaxrA(:,:,2) = argmax2;
    argmaxrA(:,:,3) = argmax3;
    argmaxrA = argmaxrA/255;
end
argmaxro = zeros(nrow,ncol,3);
[vmax,idx] = max(ro,[],3);
argmax1 = zeros(nrow,ncol);
argmax2 = zeros(nrow,ncol);
argmax3 = zeros(nrow,ncol);
for ii=1:12
    argmax1(idx==ii) = label_colours(ii,1);
    argmax2(idx==ii) = label_colours(ii,2);
    argmax3(idx==ii) = label_colours(ii,3);
    argmaxro(:,:,1) = argmax1;
    argmaxro(:,:,2) = argmax2;
    argmaxro(:,:,3) = argmax3;
    argmaxro = argmaxro/255;
end

figure1=figure('Position', [0, 0, ncol*2, ncol*2]);
set(gca,'FontSize',18);

%figure;
subplot(2,3,1);
imshow(II);
title('original image');
axis tight;

subplot(2,3,2);
imshow(gt);
title('Ground truth');
axis tight;

% subplot(2,4,3);
% imshow(rA);
% title('softmax(O)');
% axis tight;

% figure;
 subplot(2,3,3);
% brA = findargmax(rA);
imshow(argmaxrA);
title('Binarized softmax(O)');
axis tight;

% figure;
subplot(2,3,4);
plot(ep(1:iter),'LineWidth', 2);
title('primal energy');
%set(gca,'DataAspectRatio',[1 1 1]);
pbaspect([1 1 1]);

subplot(2,3,5);
plot(ed(1:iter),'r--','LineWidth', 2);
title('dual energy');
pbaspect([1 1 1]);
%axis equal;
%set(gca,'DataAspectRatio',[1 1 1]);

% figure;
% subplot(2,4,7);
% imshow(ro);
% title('Non-local softmax(O)');
% axis tight;

% figure;
subplot(2,3,6);
% bro = findargmax(ro);
imshow(argmaxro);
str = sprintf('tau=%.4f,lam=%.4f,W:%s,norm2', tau,lam,W_type);
title(str)
axis tight;

% figure;
% subplot(2,3,4);
% plot(ep(10:iter),'LineWidth', 2);
% title('primal energy');
% axis tight;
%hold on; 
% figure;
% subplot(6,2,4);
% plot(ed(10:iter),'r--','LineWidth', 2);
%legend('primal energy','dual energy');


%subplot(2,1,2);


function [ro, ep, ed] = alternative_update(O,A,W,lam,tau,iter)

    global n_clusters nrow ncol
    Phi = A; % primal variable
    % dual variable
    Q = cell(n_clusters,1);
    DPhi = cell(n_clusters,1);
    for jj=1:n_clusters
         Q{jj} = sparse(nrow*ncol,nrow*ncol);
    end
    % auxillary variable
    div_Q = zeros(size(A));
    Phi_old = Phi;
    
    na = isnan(Phi);
    if sum(na(:))>0
        pause('on');
    end

    ep = zeros(iter);
    ed = zeros(iter);
    %%%%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ii=1:iter
        % update dual variable by gradient acsend
        temp = zeros(1,n_clusters);
        parfor jj=1:n_clusters
            DPhi{jj} = grad_w(Phi(:,jj),W);
            Q{jj} = normalize_row(Q{jj} - tau*DPhi{jj}, lam);
            %Q{jj} = Q{jj} - lam*tau*DPhi{jj};
            %Q{jj} = normalize_row(Q{jj}, 1);
%             ep(ii) = ep(ii) + lam*sum(sqrt(sum(DPhi{jj}.^2, 2)));
%             ap = abs(DPhi{jj});
            temp(jj) = sum(lam.*sqrt(sum(DPhi{jj}.^2, 2)));
            %ep(ii) = ep(ii) + lam*sum(ap(:));
        end 
        ep(ii) = sum(temp(:));
        %sprintf('iter=%d,ming1=%.4f,maxg1=%.4f,ming2=%.4f,maxg2=%.4f,ming3=%.4f,maxg3=%.4f', ii,full(min(DPhi{1}(:))),full(max(DPhi{1}(:))),full(min(DPhi{2}(:))),full(max(DPhi{2}(:))),full(min(DPhi{3}(:))),full(max(DPhi{3}(:))))
        
       AO = -Phi.*O;
       AlogA = Phi.*log(Phi);
       sumAO = sum(AO(:));
       sumAlogA =  sum(AlogA(:));
       ep(ii) = ep(ii) + sumAO + sumAlogA;
        
        % update primal variable by gradient descend
        parfor jj=1:n_clusters
            div_Q(:, jj) = div_w(Q{jj},W);
        end
        Phi = softmax(O-lam*div_Q,2);
       
        %sprintf('iter=%d,mineat=%.4f,maxeta=%.4f,mindiv=%.4f,maxdiv=%.4f,minO=%.4f,maxO=%.4f', ii,full(min(Q{1}(:))),full(max(Q{1}(:))),min(div_Q(:)),max(div_Q(:)),min(O(:)),max(O(:)))

%        AO = -Phi.*O;
%        AlogA = Phi.*log(Phi);
       gAeta = sparse(size(DPhi{1}));
%      gAeta = -lam*DPhi{jj}.*Q{jj};
       gAeta = Phi.*div_Q;
       ed(ii) = lam*sum(gAeta(:)) + sumAO + sumAlogA;
       %Phi = theta*Phi_old + (1-theta)*Phi;
       Phi_old = Phi;
    end
    ro = Phi;
end


