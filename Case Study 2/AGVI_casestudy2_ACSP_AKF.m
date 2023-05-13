%% Online Inference of Full Q matrix (5x5) using AGVI%%
% Section 5.2 in the article "Approximate Gaussian Variance Inference for
% State-Space Models"
%% Created by Bhargob Deka and James-A. Goulet, 2023
clear;clc;
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock

%% Q matrix
T=1000;                 %Time-serie length
N=50;
n_x        = 5;
n_w        = n_x;
n          = n_x*2;
n_w2hat    = n_x*(n_x+1)/2;
y          = zeros(T,1);        %  Initialization of the vector of observations


%% Q matrix construction
% Choosing correlation terms for the Q matrix as defined in Section 5.2
sig        = [-0.3 -0.2 -0.1 0.25 0.35 0.4 0.45 0.5 0.55 0.6];
corr       = [1 sig(1) sig(2) sig(3) sig(4);...
              sig(1) 3 sig(5) sig(6) sig(7);...
              sig(2) sig(5) 4 sig(8) sig(9);...
              sig(3) sig(6) sig(8) 0.8 sig(10);...
              sig(4) sig(7) sig(9) sig(10) 2];
Q          =  eye(5)*corr; % Process error covariance matrix
% Q          =  Q*Q';
RT         =  0.1*eye(n_x); % Observation error covariance matrix
QT         =  Q.';
m1         =  tril(true(size(QT)),-1);
sW         =  diag(QT)';
sW_cov     =  Q(m1);
eig_Q      =  eig(Q);


w           = chol(Q)'*randn(n_x,T);
v           = chol(RT)'*randn(n_x,T);
C_LL        = 1;
C_T         = diag((C_LL.*ones(1,n_x)));
A_T         = eye(n_x); % rand(n_x), diag(rand(n_x,1))
% A_T = A_T*A_T';
%% Data
no_of_datasets = 5;
runtimes = zeros(1,5);
for p = 1:no_of_datasets
    filename = sprintf('Datasets_CaseStudy2_ACSP/Data%d.mat',p);
    load(filename);
    %% Low-Rank
    LR=0;
    %% Initial value in Cholesky-space E[L], var[L]
    total       = n_x*(n_x+1)/2;
    
    % Initializing the random terms in L^W -- Equation 24 in the article
    mL_old      = [2*ones(n_x,1);0.8*ones(total-n_x,1)]; % cannot start at zero have to fix
    SL_old      = diag([0.5*ones(n_x,1);0.5*ones(total-n_x,1)]); %0.1 -> n=10, 0.5 -> n=30
   
    %% State Estimation
    % initialization of L
    E_Lw        = zeros(total,T);
    P_Lw        = zeros(total,total,T);
    EX          = zeros(n_w+n_w2hat,T);
    E_Lw(:,1)   = mL_old;
    P_Lw(:,:,1) = SL_old;
    % Total Hidden state initialization -- Equation 23
    EX(:,1)     = [zeros(1,n_x) mL_old']';    %[E[\bm{X}]  E[\bm{W2hat}]]
    PX(:,:,1)   = diag([1*ones(1,n_x), diag(SL_old)']);    %[P[\bm{X}]  P[\bm{W2hat}]]   % high value of variance is important
    
    ind_covij   = multiagvi.index_covij(n_x);
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
        %     if t==500
        %         stop=1;
        %     end
        Ep      = [A_T*EX(1:n_x,t-1); zeros(n_x,1)];           % mu_t|t-1
        mL_old  = E_Lw(:,t-1); %old
        SL_old  = P_Lw(:,:,t-1);%old
        if any(eig(SL_old)<0)
            SL_old  = multiagvi.makePSD(SL_old);
        end
        [mL, SL]= multiagvi.convertstructure(mL_old,SL_old,n_x); %converted to new structure
        %% Matrices for mL and SL
        mL_mat  = triu(ones(n_x));
        mL_mat(logical(mL_mat)) = mL;
        SL_mat  = triu(ones(n_x));

        %% Initialization for Product L^T*L
        [E_P,P_P,Cov_L_P] = multiagvi.LtoP(n_x,mL_mat,SL_mat,SL,ind_covij);
        P_P = (P_P+P_P')/2;
        if any(eig(P_P)<0)
            P_P  = multiagvi.makePSD(P_P);
        end
        %% check
        [E_P_old,P_P_old,Cov_L_P_old] = multiagvi.convertback(E_P,P_P,Cov_L_P,n_x);
        P_P_old = (P_P_old+P_P_old')/2;
        if any(eig(P_P_old)<0)
            P_P_old  = multiagvi.makePSD(P_P_old);
        end
        % Covariance matrix construction for Prediction Step : Sigma_t|t-1
        Q_W  = triu(ones(n_x));
        Q_W(logical(Q_W)) = E_P;
        Q_W = triu(Q_W)+triu(Q_W,1)';
        Q_W  = (Q_W + Q_W)'/2;
        C   = [eye(n_x) zeros(n_x)];
        %%  1st update step:
        [EX1,PX1] = multiagvi.KFPredict(A_T,C,Q_W,RT,YT(:,t),PX(1:n_x,1:n_x,t-1),Ep);
        PX1=(PX1+PX1')/2;
        % Make PSD
        if any(eig(PX1)<0)
            PX1  = multiagvi.makePSD(PX1);
        end
        % Check if the covariance matrix is rank full
        %     if rank(PX1) == size(PX1, 1)
        %         disp('The covariance matrix is rank full');
        %     else
        %         disp('The covariance matrix is rank deficient');
        %     end
        % low-rank and/or sparse matrix
        if LR==1
            explained_variance=0.99;
            PX1 = multiagvi.lowRankApprox(PX1, explained_variance);
        end
        EX(1:n_w,t)       = EX1(1:n_w);    % n = n_x*2 i.e., no of x + no of w
        PX(1:n_w,1:n_w,t) = PX1(1:n_w,1:n_w);
        %% Collecting W|y
        if any(isnan(YT(:,t)))
            E_Lw(:,t)   = E_Lw(:,t-1);
            P_Lw(:,:,t) = P_Lw(:,:,t-1);
        else
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
            if any(eig(PX_wpy)<0)  % eigen
                PX_wpy  = multiagvi.makePSD(PX_wpy);
            end
            if issparse(PX_wpy)
                disp('update 2: PX_wpy is sparse')
            end
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
            if any(eig(PX_wp)<0)  % eigen
                PX_wp  = multiagvi.makePSD(PX_wp);
            end
            if issparse(PX_wp)
                disp('update 2: PX_wp is sparse')
            end
            %% Creating Prior E[W^p]
            E_wp        = m_wsqhat;
            %% Smoother Equations
            [ES,PS] = multiagvi.agviSmoother(E_wp,s_wsqhat,PX_wp,EX_wpy,PX_wpy,E_P_old,P_P_old);
            EX(end-n_w2+1:end,t)                 = ES;
            PX(end-n_w2+1:end,end-n_w2+1:end,t)  = PS;

            E_Pw_y  = ES;
            P_Pw_y  = PS;
            P_Pw_y  = (P_Pw_y+P_Pw_y')/2;
            if any(eig(P_Pw_y )<0)  % eigen
                P_Pw_y   = multiagvi.makePSD(P_Pw_y );
            end
            %     if rank(P_Pw_y) == size(P_Pw_y, 1)
            %         disp('The covariance matrix is rank full');
            %     else
            %         disp('The covariance matrix is rank deficient');
            %     end
            %     if LR==1
            %         explained_variance=0.99;
            %         P_Pw_y  = multiagvi.lowRankApprox(P_Pw_y, explained_variance);
            %     end
            %     if any(eig(P_Pw_y)<0)
            %         P_PW_y = multiagvi.makePSD(P_Pw_y);
            %     end
            %% Converting from Pw to Lw
            [EL_pos,PL_pos] = multiagvi.PtoL(Cov_L_P_old,E_P_old,P_P_old,E_Lw(:,t-1),P_Lw(:,:,t-1),E_Pw_y,P_Pw_y);
            PL_pos      = (PL_pos+PL_pos')/2;

            E_Lw(:,t)   = EL_pos;
            P_Lw(:,:,t) = PL_pos;
        end
    end
    runtimes(p) = toc(start);
    
    E_W = zeros(total,length(EX));
    V_W = zeros(total,length(EX));
    for t = 1:length(EX)
        mL_old     = E_Lw(:,t);
        SL_old     = P_Lw(:,:,t);
        [mL, SL]   = multiagvi.convertstructure(mL_old,SL_old,n_x); %converted to new structure
        if any(eig(SL)<0)  % eigen
            SL  = multiagvi.makePSDV2(SL);
        end
        %% Matrices for mL and SL
        mL_mat = triu(ones(n_x));
        mL_mat(logical(mL_mat)) = mL;
        SL_mat = triu(ones(n_x));
        SL_mat(logical(SL_mat)) = diag(SL);
        
    %     m      = logical(triu(mL_mat));
        %% Initialization for Product L^T*L
        [E_P_new,P_P_new]         = multiagvi.LtoP(n_x,mL_mat,SL_mat,SL,ind_covij);
        P_P_new  = (P_P_new + P_P_new')/2;
        if any(eig(P_P_new)<0)  % eigen
            P_P_new  = multiagvi.makePSD(P_P_new);
        end
        [E_W(:,t),~]     = multiagvi.convertback(E_P_new,P_P_new,[],n_x);
        
    end
    var_cov = E_W(:,end);
    %% Estimated Q matrix
    Q_mat                = diag(var_cov(1:n_x));
    L_mat                = tril(ones(n_x),-1);
    Q_mat(logical(L_mat))= var_cov(n_x+1:end);
    est_Q = Q_mat';
    est_Q(logical(L_mat))= var_cov(n_x+1:end);
%     save(['Results_Q_casestudy2_AGVI_ACSP/Q_AGVI_Dataset' num2str(p) '.mat'],'est_Q')
end
disp(['Average runtime is :' num2str(mean(runtimes))])

