%imagepath = 'CamVid/Seq05VD_f05100.png';
%imagepath = 'CamVid_example/0001TP_009180.png'
imagepath = 'AUnet/5.png';
npy = readNPY(strrep(imagepath,'png','npy'));
O_k = permute(npy,[2,3,1]);

II = imread(imagepath);
% h = fspecial('gaussian',7,13);
% II = imfilter(II,h,'symmetric');
I = double(II);
I = I/max(I(:));
gt = imread(strrep(imagepath,'.png','_gt.png')); 

global n_clusters nrow ncol
[nrow, ncol, d] = size(I);
[nrow_o,ncol_o,d_o] = size(O_k);
n_clusters= d_o;

tic;
W_type = 'CRF20';
%W_type = 'wn4';
% delete(gcp('nocreate'));
% parpool(4);
W = image_graph_affinity3(double(I));
% if strcmp(W_type,'n4')
%     W_mat = load(strrep(imagepath,'.png','_n4.mat'));
% elseif strcmp(W_type,'CRF')
%     W_mat = load(strrep(imagepath,'.png','_k100.mat'));
% end
%W = W_mat.W;
%W = max(W,W');
toc;
%W = sign(W);
%npy1 = squeeze(npy(1,:,:));

E = [0.1,0.5,1,3];
lam = 6;
global e
iter = 400;
interval = 0.25*iter;

for ii=2:2
    
    e = E(ii);
    tau = 0.03*e;
    O = reshape(O_k,[],n_clusters);
    A = softmax(O/e,2);
    
    tic;
    alternative_update(O,A,W,lam,tau,iter,imagepath,II,gt,W_type,interval);
    toc;
end

%function [ro, ep, ed] = alternative_update(O,A,W,lam,tau,iter,imagepath,II,gt,W_type,interval)
function alternative_update(O,A,W,lam,tau,iter,imagepath,II,gt,W_type,interval)    

    global n_clusters nrow ncol e
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

    ep = zeros(1,iter);
    ed = zeros(1,iter);
    %%%%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure1=figure('Position', [10, 10, nrow*2+10, ncol*2+10]);
    set(gca,'FontSize',18);
    
    for ii=1:iter
        % update dual variable by gradient acsend
        for jj=1:n_clusters
            DPhi{jj} = grad_w(Phi(:,jj),W);
%             Q{jj} = normalize_row_inf(Q{jj} - tau*DPhi{jj}, lam);
%             ap = abs(DPhi{jj});
%             ep(ii) = ep(ii) + lam*sum(ap(:));
            Q{jj} = normalize_row(Q{jj} - tau*DPhi{jj}, lam);
            ep(ii) = ep(ii) + lam*sum(sqrt(sum(DPhi{jj}.^2, 2)));
        end       
        
        % update primal variable by gradient descend
        for jj=1:n_clusters
            div_Q(:, jj) = div_w(Q{jj},W);
        end
        
       %Phi = softmax((O-lam*div_Q)/e,2);
       Phi = softmax((O-div_Q)/e,2);
       
       AO = -Phi.*O;
       AlogA = e*Phi.*log(Phi+1e-6);
       ep(ii) = ep(ii)+sum(AO(:)) + sum(AlogA(:));
       ed(ii) = sum(AO(:)) + sum(AlogA(:)) + sum(sum(Phi.*div_Q));
       
        %Phi = theta*Phi_old + (1-theta)*Phi;
        Phi_old = Phi;
       
        if mod(ii,interval)==0 
            rA = reshape(A,nrow,ncol,n_clusters);
            ro = reshape(Phi,nrow,ncol,n_clusters);
            visA = visualW(rA);
            viso = visualW(ro);

            %figure;
            subplot(2,2,1);
            imshow(II);
            title('original image');
            axis tight;

            subplot(2,2,2);
            imshow(gt);
            title('gt');
            axis tight;

            % figure;
            subplot(2,2,3);
            imshow(visA);
            title('Binarized softmax(O)');
            axis tight;

            % figure;
            subplot(2,2,4);
            imshow(viso);
            %str = sprintf('tau=%.3f,e=%.2f,lam=%.2f,W:%s,norm2', tau,e,lam,W_type);
            str = sprintf('tau=%.3f,e=%.2f,lam=%.2f,W-%s,iter=%d', tau,e,lam,W_type,ii);
            title(str);
            axis tight;

            s = split(imagepath,'.');
            filename = strcat(s{1},'_',str,'.',s{2});
            saveas(gcf,filename);
            
            imagename = strcat(s{1},'_NLUnet_',string(ii),'.',s{2});
            imwrite(viso,convertStringsToChars(imagename));
        end
    end
    figure;
    plot(ep(10:iter),'LineWidth', 2);
    axis tight;
    hold on; 

    %figure;
    plot(ed(10:iter),'r--','LineWidth', 2);
    title('primal-dual energy');
    %title('dual energy');
    legend('primal energy','dual energy');
    filename = strcat(s{1},'_',str,'_pd.',s{2});
    saveas(gcf,filename);
    ro = Phi;
end


