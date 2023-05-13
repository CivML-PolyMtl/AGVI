%% Online Inference of Univariate Process Error Variance for the Linear Time-Varying Model Using AGVI %%
% Impact of Q/R ratio
% Section 5.1.4 in the article "Approximate Gaussian Variance Inference for State-Space Models"
%% Authors: Bhargob Deka and James-A. Goulet, 2023 %%
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock

%% Parameters
T          = 1000;                 % Time-serie length
n_x        = 1;                    % no. of hidden states
n_w        = n_x;                  % no. of process error terms
N          = 50;
n_w2hat    = n_x*(n_x+1)/2;        % total variance and covariance terms   
y          = zeros(T,1);           % initialization of the vector of observations
% Truncated Gaussian
sw2=10;
mw2=20;
% sigma initialised from prior knowledge
w2hat_sample = normrnd(mw2,sw2);
while w2hat_sample<0
    w2hat_sample = normrnd(mw2,sw2);
end
sW = sqrt(w2hat_sample);
if w2hat_sample < 0
  check;
end
Qr    = [0.1:0.05:1 1.5:0.5:10 100];
sigma_w   = zeros(1,length(Qr));
s_sigma_w = zeros(1,length(Qr));
j = 1;
while j <= length(Qr)
    Q                = sW^2;
    R                = Q/Qr(j);
    %% ** important to set the seed here ** %%
    rand_seed=2;
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed)); 
    %% Data
    YT          = zeros(n_x,T);
    x_true      = zeros(n_x,T);
    x_true(:,1) = 0;
    w           = chol(Q)'*randn(n_x,T);
    v           = chol(R)'*randn(n_x,T);
    for t = 2:T
        A_t          = (0.8-0.1*sin(7*pi*t/T)); %(0.8-0.1*sin(7*pi*t/T))
        C_t          = (1-0.99*sin(100*pi*t/T)); %(1-0.99*sin(100*pi*t/T))
        x_true(:,t)  = A_t*x_true(:,t-1) + w(:,t);
        YT(:,t)      = C_t*x_true(:,t)   + v(:,t);
    end
    
    %% State estimation
    EX        = zeros(3,T);                               %   X  = [X^LL  X^AR  W  hat(W^2)]
    EX(:,1)   = [0 nan mw2]';
    PX(:,:,1) = diag([100,nan,sw2^2]);
    
    %% Prediction
    for t=2:T
        A        = (0.8-0.1*sin(7*pi*t/T));%(0.8-0.1*sin(7*pi*t/T));
        Ep       = [A * EX(1,t-1); 0];              % mu_t|t-1
    
        m_w2hat  = EX(3,t-1);
        s_w_sq   = m_w2hat;
    
        Sp       = [A.^2 * PX(1,1,t-1)+s_w_sq    s_w_sq  ;        %  V(X)_t|t-1
                                s_w_sq           s_w_sq  ];
    
        C        = [(1-0.99*sin(100*pi*t/T)) 0];%
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
    % Collecting final variance value
    estim_sigma_w = EX(end,T);
    P_sigma_w     = PX(end,end,T);
    sigma_w(:,j)  = estim_sigma_w;
    s_sigma_w(:,j)= sqrt(P_sigma_w);
    j = j+1;
end
%% Plotting Q/R ratio
figure;
patch([Qr,fliplr(Qr)],[sigma_w(1,:)+s_sigma_w(1,:),fliplr(sigma_w(1,:)-s_sigma_w(1,:))],'g','FaceAlpha',0.2,'EdgeColor','none');hold on;
plot(Qr,sigma_w(1,:),'k');hold on; plot(Qr,repmat(sW^2,[1,length(Qr)]),'-.r','Linewidth',1.5);
xlabel('Q/R ratio');
ylabel('sigma_W');
ylim([10,30])
set(gca, 'XScale', 'log');