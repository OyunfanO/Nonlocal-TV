  function mu = softmax(eta,dim)
    % Softmax function
    % mu(i,c) = exp(eta(i,c))/sum_c' exp(eta(i,c'))

    % This file is from matlabtools.googlecode.com
    [nrow,k] = size(eta);
    %M = max(eta(:));
    M = max(eta,[],dim);
    mM = repmat(M,1,k);
    eta = eta - mM;
    tmp = exp(eta);
    nom = sum(tmp, dim);
    dnom = repmat(nom,1,k);
%     dnom = zeros(size(eta));
%     dnom(:,:,1) = nom;
%     dnom(:,:,2) = nom;
%     dnom(:,:,3) = nom;
    mu = bsxfun(@rdivide, tmp, dnom);

end