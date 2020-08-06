imagepath = 'npy/10.png';
II = imread(imagepath);
I = double(II);
I = I/max(I(:));
npypath = strrep(imagepath,'png','npy');
npy = readNPY(npypath);
O_k = permute(npy,[2,3,1]);

global n_clusters nrow ncol
[nrow, ncol, d] = size(I);
[nrow_o,ncol_o,d_o] = size(O_k);
n_clusters= d_o;

lam = 2;
tau = 0.01;
iter = 100;
W_type = 'mix';
comix = 0.5;

%path = 'CamVid/';
path = 'npy/';
files = dir(path);
[L,t] = size(files);

delete(gcp('nocreate'));
%parpool(4);

for ii=1:L
    if  isempty(strfind(files(ii).name,'.png'))
        continue
    end
    imagepath = strcat(path,files(ii).name);
    npypath = strrep(imagepath,'png','npy');
    npy = readNPY(npypath);
    gtstr = strrep(imagepath,'.png','_gt.png');
    gtstr = strrep(gtstr,'npy','gt');
    gt = imread(gtstr);

    O_k = permute(npy,[2,3,1]);

    II = imread(imagepath);
    I = double(II);
    I = I/max(I(:));    

    if strcmp(W_type,'n4')
        W_mat = load('n4.mat');
        W = W_mat.W;
    elseif strcmp(W_type,'CRF')
         W_mat = load(strrep(imagepath,'.png','_k100.mat'));
         W = W_mat.W;
    elseif strcmp(W_type,'mix')
        W1_mat = load('n4.mat');
        W2_mat = load(strrep(imagepath,'.png','_k100.mat'));
        W_n4 = W1_mat.W;
        W_CRF = W2_mat.W;
        n4idx = W_n4>0;
        W = comix*(W_CRF-W_CRF.*n4idx) + W_n4;   
    end
    W = max(W',W);
    toc;
    
%     tic;
%     W = image_graph_affinity3(double(I));
%     W = max(W,W');
%     Wpath = strrep(imagepath,'.png','_k100.mat');
%     save(Wpath,'W');
%     toc;
    
    O = reshape(O_k,[],n_clusters);
    A = softmax(O,2);

    tic;
    [ro,ep,ed] = alternative_update(O,A,W,lam,tau,iter);
    toc;

    rA = reshape(A,nrow,ncol,d_o);
    ro = reshape(ro,nrow,ncol,d_o);
    
    [vmax,idx] = max(ro,[],3);
    argmax = zeros(nrow,ncol);
    argmax(idx==1) = 0;
    argmax(idx==2) = 127;
    argmax(idx==3) = 255;
    argmax = argmax/255;
    outpath = strrep(imagepath,'.png','_NLTV.png');
    outpath = strrep(outpath,'npy','NLTV');
    imwrite(argmax,outpath);

%     figure1=figure('Position', [10, 10, nrow*4+20, ncol*2+20]);
%     set(gca,'FontSize',18);
% 
%     subplot(2,4,1);
%     imshow(II);
%     title('original image');
%     axis tight;
% 
%     subplot(2,4,2);
%     imshow(gt);
%     title('Ground truth');
%     axis tight;
% 
%     subplot(2,4,3);
%     imshow(rA);
%     title('softmax(O)');
%     axis tight;
%     
%     subplot(2,4,4);
%     brA = findargmax(rA);
%     imshow(brA);
%     title('Binarized softmax(O)');
%     axis tight;
% 
%     subplot(2,4,5);
%     plot(ep(1:iter),'LineWidth', 2);
%     title('primal energy');
%     pbaspect([1 1 1]);
% 
%     subplot(2,4,6);
%     plot(ed(1:iter),'r--','LineWidth', 2);
%     title('dual energy');
%     pbaspect([1 1 1]);
% 
%     subplot(2,4,7);
%     imshow(ro);
%     title('Non-local softmax(O)');
%     axis tight;
% 
%     subplot(2,4,8);
%     bro = findargmax(ro);
%     imshow(bro);
%     str = sprintf('tau=%.2f,lam=%.2f,W:%s,norm2', tau,lam,W_type);
%     title(str)
%     axis tight;
end


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

    ep = zeros(iter);
    ed = zeros(iter);
    %%%%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ii=1:iter
        % update dual variable by gradient acsend
        temp = zeros(1,n_clusters);
        for jj=1:n_clusters
            DPhi{jj} = grad_w(Phi(:,jj),W);
            Q{jj} = normalize_row(Q{jj} - tau*DPhi{jj}, lam);
            %Q{jj} = Q{jj} - lam*tau*DPhi{jj};
            %Q{jj} = normalize_row(Q{jj}, 1);
%             ep(ii) = ep(ii) + lam*sum(sqrt(sum(DPhi{jj}.^2, 2)));
%             temp(jj) = sum(lam.*sqrt(sum(DPhi{jj}.^2, 2)));
        end 
%         ep(ii) = sum(temp(:));
        %sprintf('iter=%d,ming1=%.4f,maxg1=%.4f,ming2=%.4f,maxg2=%.4f,ming3=%.4f,maxg3=%.4f', ii,full(min(DPhi{1}(:))),full(max(DPhi{1}(:))),full(min(DPhi{2}(:))),full(max(DPhi{2}(:))),full(min(DPhi{3}(:))),full(max(DPhi{3}(:))))
        
%        AO = -Phi.*O;
%        AlogA = Phi.*log(Phi);
%        sumAO = sum(AO(:));
%        sumAlogA =  sum(AlogA(:));
%        ep(ii) = ep(ii) + sumAO + sumAlogA;
        
        % update primal variable by gradient descend
        for jj=1:n_clusters
            div_Q(:, jj) = div_w(Q{jj},W);
        end
        Phi = softmax(O-lam*div_Q,2);
%         isn = isnan(Phi(:));
%         sum(isn);
        %sprintf('iter=%d,mineat=%.4f,maxeta=%.4f,mindiv=%.4f,maxdiv=%.4f,minO=%.4f,maxO=%.4f', ii,full(min(Q{1}(:))),full(max(Q{1}(:))),min(div_Q(:)),max(div_Q(:)),min(O(:)),max(O(:)))

%        AO = -Phi.*O;
%        AlogA = Phi.*log(Phi);
%        gAeta = sparse(size(DPhi{1}));
% %      gAeta = -lam*DPhi{jj}.*Q{jj};
%        gAeta = Phi.*div_Q;
%        ed(ii) = lam*sum(gAeta(:)) + sumAO + sumAlogA;
       %Phi = theta*Phi_old + (1-theta)*Phi;
       Phi_old = Phi;
    end
    ro = Phi;
end


