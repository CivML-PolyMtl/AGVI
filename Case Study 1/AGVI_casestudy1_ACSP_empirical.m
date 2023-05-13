%% Online Inference of Univariate Process Error Variance for the Linear Time-Varying Model Using AGVI %%
% Empirical Consistency Test 
% Section 5.1.3 in the article "Approximate Gaussian Variance Inference for State-Space Models"
%% Authors: Bhargob Deka and James-A. Goulet, 2023 %%
%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock

%% Parameters
format short 
T          = 1000;                 % Time-serie length
n_x        = 1;                    % no. of hidden states
n_w        = n_x;                  % no. of process error terms
N          = 1000;                 % Total no. of runs
n_w2hat    = n_x*(n_x+1)/2;        % total variance and covariance terms   
y          = zeros(T,1);           % initialization of the vector of observations
%% Initialization
EX_S = zeros(1,T,N);
PX_S = zeros(1,T,N);
index_onestd   = cell(T,1);
index_twostd   = cell(T,1);
index_thirdstd = cell(T,1);
ratio_onestd   = zeros(1,T);
ratio_twostd   = zeros(1,T);
ratio_thirdstd = zeros(1,T);
sW             = zeros(1,N);

for n = 1:N
    % Truncated Gaussian
    sw2=0.1;
    mw2=0.2;
    % sigma initialised from prior knowledge
     w2hat_sample = normrnd(mw2,sw2);
     while w2hat_sample<0
         w2hat_sample = normrnd(mw2,sw2);
     end
     sW(n) = sqrt(w2hat_sample);
     if w2hat_sample < 0
         check;
     end
     %% Q matrix
    Q          = sW(n)^2;
    %% R matrix
    QR_ratio   = 1;                    % Q/R = (\sigma_AR)^2/(\sigma_V)^2
    R          = Q/QR_ratio;
    sV         = sqrt(R);
    %% Data
    YT          = zeros(n_x,T);
    x_true      = zeros(n_x,T);
    x_true(:,1) = 0;
    w           = chol(Q)'*randn(n_x,T);
    v           = chol(R)'*randn(n_x,T);
    for t = 2:T
        A_t          = (0.8-0.1*sin(7*pi*t/T));
        C_t          = (1-0.99*sin(100*pi*t/T));
        x_true(:,t)  = A_t*x_true(:,t-1) + w(:,t);
        YT(:,t)      = C_t*x_true(:,t)   + v(:,t);
    end
    %% State estimation
    EX        = zeros(3,T);                               %   X  = [X^LL  X^AR  W  hat(W^2)]
    EX(:,1)   = [0 nan mw2]';
    PX(:,:,1) = diag([100,nan,sw2^2]);
    
    
    %% Prediction
    for t=2:T
        A        = (0.8-0.1*sin(7*pi*t/T));
        Ep       = [A * EX(1,t-1); 0];              % mu_t|t-1
    
        m_w2hat  = EX(3,t-1);
        s_w_sq   = m_w2hat;
    
        Sp       = [A.^2 * PX(1,1,t-1)+s_w_sq    s_w_sq  ;        %  V(X)_t|t-1
                                s_w_sq           s_w_sq  ];
    
        C        = [(1-0.99*sin(100*pi*t/T)) 0];
        SY       = C*Sp*C'+R;
        SYX      = Sp*C';
        my       = C*Ep;
        K        = SYX/SY;
        %% Ist Update step:
        EX1  = Ep+SYX/SY*(YT(:,t)-my);
    
        PX1  = Sp-SYX/SY*SYX';
    
        EX(1:2,t)     = EX1;
        PX(1:2,1:2,t) = PX1;
        v             = YT(:,t)-my;
        NIS_NPI(n,t)  = v * inv(SY) * v';
        Nees_NPI(n,t) = (x_true(1,t)-EX(1,t))'*inv(PX(1,1,t))*(x_true(1,t)-EX(1,t));
    
        %% 2nd Update step:
        s_w2_sq  = 2*(EX(3,t-1))^2;
        m_w2     = EX(2,t)^2+PX(2,2,t);
        s_w2     = 2*PX(2,2,t)^2+4*PX(2,2,t)*EX(2,t)^2;
    
        my1      = EX(3,t-1);
        SYX1     = PX(3,3,t-1);
    
        %% Smoother Equations
        E_W2_pos      = m_w2;
        E_W2_prior    = my1;
        C_W2_W2hat    = SYX1;
        P_W2_prior    = 3*PX(3,3,t-1) + s_w2_sq;
        P_W2_pos      = s_w2;
        J             = C_W2_W2hat/P_W2_prior;
        EX(end,t)     = EX(3,t-1)  + J*(E_W2_pos - E_W2_prior);
        PX(end,end,t) = PX(3,3,t-1) + J^2*P_W2_pos - C_W2_W2hat^2/P_W2_prior;
        
    end
    EX_S(:,:,n) = EX(end,:);
    PX_S(:,:,n) = squeeze(PX(end,end,:))';
    U1(n,:) = squeeze(EX_S(:,:,n) + sqrt(PX_S(:,:,n)));
    L1(n,:) = squeeze(EX_S(:,:,n) - sqrt(PX_S(:,:,n)));
    U2(n,:) = squeeze(EX_S(:,:,n) + 2*sqrt(PX_S(:,:,n)));
    L2(n,:) = squeeze(EX_S(:,:,n) - 2*sqrt(PX_S(:,:,n)));
    U3(n,:) = squeeze(EX_S(:,:,n) + 3*sqrt(PX_S(:,:,n)));
    L3(n,:) = squeeze(EX_S(:,:,n) - 3*sqrt(PX_S(:,:,n)));
    Tr(n,:) = sW(n)^2.*ones(1,1000);

end
for i = 1:T
     index_onestd{i,1} = find(Tr(:,i) >= L1(:,i) & Tr(:,i) <= U1(:,i));
     ratio_onestd(i)   = 100*size(index_onestd{i},1)/N;
     index_twostd{i,1} = find(Tr(:,i) >= L2(:,i) & Tr(:,i) <= U2(:,i));
     ratio_twostd(i)   = 100*size(index_twostd{i},1)/N;
     index_thirdstd{i,1} = find(Tr(:,i) >= L3(:,i) & Tr(:,i) <= U3(:,i));
     ratio_thirdstd(i)   = 100*size(index_thirdstd{i},1)/N;
 end
 figure;
 plot(ratio_onestd,'g');hold on;plot(ratio_twostd,'r');hold on;plot(ratio_thirdstd,'b');hold on
 plot(95*ones(1,T),'--k');hold on; plot(68*ones(1,T),'--k');
 xlabel('t');
 ylabel('C.I.');
 legend('one std','two std','third std');
 ylim([25,100]);