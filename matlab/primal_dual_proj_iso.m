function [Phi, J_Phi, J_gap, iter] = primal_dual_proj_iso(opt, Cs, Ct)
%  primal dual with projection onto simplex algorithm for segmentation
%  Phi = argmin int_\Omega Cs*(1-Phi) + Ct*Phi + alpha*||grad(Phi)||, s.t. Phi is in
%  unit simplex

%%%%%%%%%%%%%%%%%%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
max_iter = opt.max_iter;
P = opt.P;
F = Ct - Cs;
alpha = opt.alpha;
n = size(Cs,1);
n_clusters = opt.n_clusters;
beta_l = opt.beta_l;
gamma_l = opt.gamma_l;
W = opt.W;
intCs = sum(sum(Cs)); % integral of Cs over the domain
theta = -0.5; % coefficient for weighted average of two consecutive Phi's
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%% variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% energy functional
J_Phi = zeros(max_iter, 1);
% duality gap
J_gap = zeros(max_iter, 1);
% primal variable
% Phi = Cs;
Phi = 1/n_clusters*ones(n,n_clusters); % primal variable
% dual variable
Q = cell(n_clusters,1);
DPhi = cell(n_clusters,1);
for jj=1:n_clusters
    Q{jj} = normalize_row(grad_w(Phi, W), alpha);
end
% auxillary variable
QF = zeros(size(F));
Phi_old = Phi;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%% main loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:max_iter
    % update dual variable by gradient acsend
    for jj=1:n_clusters
        DPhi{jj} = grad_w(Phi(:,jj),W);
        Q{jj} = normalize_row(Q{jj} - beta_l(ii)*DPhi{jj}, alpha);
    end
    % update primal variable by gradient descend
    for jj=1:n_clusters
        QF(:, jj) = div_w(Q{jj},W) + F(:,jj);
        Phi(:,jj) = Phi(:,jj) - gamma_l(ii)*QF(:,jj);
    end
    Phi = projsplx_mult(Phi);
    % energy
    J_Phi(ii) = sum(dot(Cs, 1-Phi) + dot(Ct, Phi));
    for jj=1:n_clusters
        %size(alpha)
        %size(sum(sqrt(sum(DPhi{jj}.^2, 2))))
        %size(sqrt(sum(DPhi{jj}.^2, 2)))
        J_Phi(ii) = J_Phi(ii) + sum(alpha.*sqrt(sum(DPhi{jj}.^2, 2)));
    end
    % duality gap
    J_gap(ii) = J_Phi(ii) - sum(min(QF,[],2)) - intCs;
%     J_gap(ii) = J_Phi(ii) - sum(dot(QF, Phi)) - intCs;
    if abs(J_gap(ii)/J_Phi(ii))< 1e-5
        break;
    end

%     if ii>1 
%         if abs(J_Phi(ii,1)-J_Phi(ii-1,1))/abs(J_Phi(ii-1,1)) < 1e-8
%             break;
%         end
%     end
    
    %Phi = theta*Phi_old + (1-theta)*Phi;
    Phi_old = Phi;
end
J_Phi = J_Phi(1:ii);
J_gap = J_gap(1:ii);
iter = ii;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%