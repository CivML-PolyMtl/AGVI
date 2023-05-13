%% Online Inference of Full Q matrix (5x5) using AGVI%%
% Section 5.2 in the article "Approximate Gaussian Variance Inference for
% State-Space Models"
%% Created by Bhargob Deka and James-A. Goulet, 2023
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% A matrix
A_LL = 1;
%% Q matrix
T=1000;                 % Time-series length
N=50;
n_x        = 5;
n_w        = n_x;
n          = n_x*2;
n_w2hat    = n_x*(n_x+1)/2;
y          = zeros(T,1);        %  Initialization of the vector of observations
% sV         = 1e-02;
R_T        = 0.1.*eye(n_x);
A_T        = diag(1*ones(1,n_x));

sig        = [-0.3 -0.2 -0.1 0.25 0.35 0.4 0.45 0.5 0.55 0.6];
corr       = [1 sig(1) sig(2) sig(3) sig(4);...
              sig(1) 3 sig(5) sig(6) sig(7);...
              sig(2) sig(5) 4 sig(8) sig(9);...
              sig(3) sig(6) sig(8) 0.8 sig(10);...
              sig(4) sig(7) sig(9) sig(10) 2];
Q          =  eye(5)*corr;
QT         = Q.';
m1         = tril(true(size(QT)),-1);
sW         = diag(QT)';
sW_cov     = Q(m1);
eig_Q      = eig(Q);

C_LL        = 1;
C_T         = diag((C_LL.*ones(1,n_x)));
alpha       = 1.5:0.1:2;
beta        = 0.5:0.1:1;
mean_ANIS   = zeros(1,length(alpha));
for init=1:length(alpha)
seedlist    = [67 10 49 7 4];
    for p = 1:5
        rand_seed=seedlist(p);%67,10,49,7,34, seedlist(p)
        RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));
        YT          = zeros(n_x,T);
        x_true      = zeros(n_x,T);
        x_true(:,1) = zeros(n_x,1);
        w           = chol(Q)'*randn(n_x,T);
        v           = chol(R_T)'*randn(n_x,T);
        for t = 2:T
            x_true(:,t)  = A_T*x_true(:,t-1) + w(:,t);
            YT(:,t)      = C_T*x_true(:,t)   + v(:,t);
        end
        YT = filloutliers(YT,'center','mean','ThresholdFactor', 2);
        %     dat = load(['AGVI_M_CS1_Data0' num2str(p) '.mat']);
        %     YT  = dat.YT;
        %% Initial value in Cholesky-space E[L], var[L]
        total       = n_x*(n_x+1)/2;
        mL_old      = [alpha(init)*ones(n_x,1);beta(init)*ones(total-n_x,1)]; % cannot start at zero have to fix
        SL_old      = diag([0.5*ones(n_x,1);0.5*ones(total-n_x,1)]);
        %% State Estimation
        % initialization of L
        E_Lw        = zeros(total,T);
        P_Lw        = zeros(total,total,T);
        EX          = zeros(n_w+n_w2hat,T);
        E_Lw(:,1)   = mL_old;
        P_Lw(:,:,1) = SL_old;
        EX(:,1)     = [zeros(1,n_x) mL_old']';    %[E[\bm{X}]  E[\bm{W2hat}]]
        PX(:,:,1)   = diag([1*ones(1,n_x), diag(SL_old)']);    %[P[\bm{X}]  P[\bm{W2hat}]]   % high value of variance is important
    
        ind_covij   = multiagvi.index_covij(n_x);
        NIS_AGVI    = zeros(1,T);
        %% Indices
        % Creating the cells that will hold the covariance terms
        % cov(w_iw_j,w_kw_l) and the indices for each row
        n_w2             = n_x*(n_x+1)/2;
        ind_wijkl        = multiagvi.indcov_wijkl(n_x);
        %% Getting the mean indices
        ind_mu           = multiagvi.ind_mean(n_x,ind_wijkl);
        %% Getting the covariance indices from the \Sigma_W matrix
        ind_cov          = multiagvi.ind_covariance(n_w2,n_w,ind_wijkl);
        %% Computing the indices for the covariances for prior \Sigma_W^2:
        ind_cov_prior = multiagvi.indcov_priorWp(ind_wijkl,n_w2,n_x);
        start = tic;
        for t=2:T
            %     if t==101
            %         stop=1;
            %     end
            Ep      = [A_T*EX(1:n_x,t-1); zeros(n_x,1)];           % mu_t|t-1
            mL_old  = E_Lw(:,t-1); %old
            SL_old  = P_Lw(:,:,t-1);%old
            [mL, SL]= multiagvi.convertstructure(mL_old,SL_old,n_x); %converted to new structure
            %% Matrices for mL and SL
            mL_mat  = triu(ones(n_x));
            mL_mat(logical(mL_mat)) = mL;
            SL_mat  = triu(ones(n_x));
            SL_mat(logical(SL_mat)) = diag(SL);
            %     SL_mat = (SL_mat+SL_mat')/2;
            %     m      = logical(triu(mL_mat));
            %% Initialization for Product L^T*L
            [E_P,P_P,Cov_L_P] = multiagvi.LtoP(n_x,mL_mat,SL_mat,SL,ind_covij);
            P_P = (P_P+P_P')/2;
            %% check
            [E_P_old,P_P_old,Cov_L_P_old] = multiagvi.convertback(E_P,P_P,Cov_L_P,n_x);
            P_P_old = (P_P_old+P_P_old')/2;
            % Covariance matrix construction for Prediction Step : Sigma_t|t-1
            Q_W  = triu(ones(n_x));
            Q_W(logical(Q_W)) = E_P;
            Q_W = triu(Q_W)+triu(Q_W,1)';
            Q_W  = (Q_W + Q_W)'/2;
            C   = [eye(n_x) zeros(n_x)];
            %%  1st update step:
            [EX1,PX1,NIS] = multiagvi.KFPredict(A_T,C,Q_W,R_T,YT(:,t),PX(1:n_x,1:n_x,t-1),Ep);
            PX1=(PX1+PX1')/2;
            NIS_AGVI(:,t)   = NIS;
            EX(1:n_w,t)       = EX1(1:n_w);    % n = n_x*2 i.e., no of x + no of w
            PX(1:n_w,1:n_w,t) = PX1(1:n_w,1:n_w);
            %% Collecting W|y
            EX_wy   = EX1(end-n_x+1:end,1);
            PX_wy   = PX1(end-n_x+1:end,end-n_x+1:end,1);
            m       = triu(true(size(PX_wy)),1);  % Finding the upper triangular elements
            cwiwjy  = PX_wy(m)';                  % covariance elements between W's after update
            P_wy    = diag(PX_wy); % variance of updated W's
            i = 1; j = 1; k = 1;
            while i <= n_x-1
                PX_wy(i,j+1) = cwiwjy(k);
                j = j+1;
                k = k+1;
                if j == n_x
                    i = i+1;
                    j = i;
                end
            end
            PX_wy  = triu(PX_wy)+triu(PX_wy,1)';
            %%% 2nd Update step:
            %% Computing W^2|y
            [m_wii_y,s_wii_y,m_wiwj_y, s_wiwj_y] = multiagvi.meanvar_w2pos(EX_wy,PX_wy,cwiwjy, P_wy, n_w2hat,n_x);
    
            %% Computing the covariance matrix \Sigma_W^2|y:
            cov_wijkl = multiagvi.cov_w2pos(PX_wy,EX_wy,n_w2,ind_cov,ind_mu);
            %     cov_wpy   = cell2mat(reshape(cov_wijkl,size(cov_wijkl,2),1));
            %% Adding the variances and covariances of W^p to form \Sigma_W^p
            PX_wpy    = multiagvi.var_wpy(cov_wijkl,s_wii_y,s_wiwj_y);
            PX_wpy    = (PX_wpy+PX_wpy')/2;
            %% Creating E[W^p]
            EX_wpy        = [m_wii_y' m_wiwj_y];
    
            %% Computing prior mean and covariance matrix of Wp
            m_wsqhat    = E_P_old;
            s_wsqhat    = P_P_old;
            PX_wp       = multiagvi.covwp(P_P_old,m_wsqhat,n_x,n_w2,n_w);% old format
    
            %% Computing the prior covariance matrix \Sigma_W^p
            cov_prior_wijkl = multiagvi.priorcov_wijkl(m_wsqhat,ind_cov_prior,n_w2);
            %% Adding the variances and covariances for Prior \Sigma_W^p
            cov_wp              = cell2mat(reshape(cov_prior_wijkl,size(cov_prior_wijkl,2),1));
            s_wp                = zeros(size(PX_wp,1));
            s_wp(1:end-1,2:end) = cov_wp;
            PX_wp               = PX_wp + s_wp;
            PX_wp               = triu(PX_wp)+triu(PX_wp,1)'; % adding the lower triangular matrix
            PX_wp               = (PX_wp+PX_wp')/2;
            %% Creating Prior E[W^p]
            E_wp        = m_wsqhat;
            %% Smoother Equations
            [ES,PS] = multiagvi.agviSmoother(E_wp,s_wsqhat,PX_wp,EX_wpy,PX_wpy,E_P_old,P_P_old);
            EX(end-n_w2+1:end,t)                 = ES;
            PX(end-n_w2+1:end,end-n_w2+1:end,t)  = PS;
    
            E_Pw_y  = ES;
            P_Pw_y  = PS;
            P_Pw_y  = (P_Pw_y+P_Pw_y')/2;
            if any(eig(P_Pw_y)<0)
                P_PW_y = multiagvi.makePSD(P_Pw_y);
            end
            %% Converting from Pw to Lw
            [EL_pos,PL_pos] = multiagvi.PtoL(Cov_L_P_old,E_P_old,P_P_old,E_Lw(:,t-1),P_Lw(:,:,t-1),E_Pw_y,P_Pw_y);
            PL_pos      = (PL_pos+PL_pos')/2;
            if any(eig(PL_pos)<0)
                P_PW_y = multiagvi.makePSD(PL_pos);
            end
            E_Lw(:,t)   = EL_pos;
            P_Lw(:,:,t) = PL_pos;
        end
    
        runtime = toc(start)
        % E_Lw(:,end)
        Avg_NIS_time    = NIS_AGVI;
        N_ANIS(p)       = length(Avg_NIS_time(Avg_NIS_time > 12.833 | Avg_NIS_time < 0.831));
%         disp(N_ANIS)
        ANIS            = mean(Avg_NIS_time);
        % figure;
        % plot(1:T-1,Avg_NIS_time(2:end),'k');hold on; plot(1:T-1,5.02*ones(1,T-1),'r');hold on; plot(1:T-1,0*ones(1,T-1),'g')
        % title('NIS');
        % ylabel('test statistic');
    end
mean_ANIS(init) = sum(N_ANIS)/5;
end
disp(mean_ANIS)
mean(mean_ANIS)
