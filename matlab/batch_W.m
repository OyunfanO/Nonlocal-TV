path = 'CamVid/';
%path = 'image/';
files = dir(path);
[L,t] = size(files);

W_type = 'n4';

% for ii=1:L
%     %if  ~isempty(strfind(files(ii).name,'_gt.png'))
%     if  ~isempty(strfind(files(ii).name,'.png'))
%         imagepath = strcat(path,files(ii).name);
        %imagepath = strrep(imagepath,'_gt.','.');
        imagepath = 'CamVid/Seq05VD_f05100.png';
        I = imread(imagepath);
        I = double(I);
        I = I/max(I(:));
        [nrow, ncol, d] = size(I);
        tic;
        W = image_graph_affinity3(double(I));
        if strcmp(W_type,'n4')
            save(strrep(imagepath,'.png','_n4.mat'), 'W' );
        elseif strcmp(W_type,'CRF')
            save(strrep(imagepath,'.png','_k10.mat'), 'W' );
        end
        toc;
%     end
% end