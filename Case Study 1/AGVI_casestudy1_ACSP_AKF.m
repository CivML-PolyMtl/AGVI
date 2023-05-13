%% Online Inference of Univariate Process Error Variance for the Linear Time-Varying Model Using AGVI %%
% Saving Results for AGVI for comparing with the AKF methods
% Section 5.1.5 in the article "Approximate Gaussian Variance Inference for State-Space Models"
%% Authors: Bhargob Deka and James-A. Goulet, 2023 %%
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock

%% Parameters
T          = 1000;                 % Time-serie length
n_x        = 1;                    % no. of hidden states
n_w        = n_x;                  % no. of process error terms

n_w2hat     = n_x*(n_x+1)/2;        % total variance and covariance terms   
y           = zeros(T,1);           % initialization of the vector of observations
sW_list     = [0.42 1.35 18.75];
estim_Q     = zeros(3,5);
Pr_w2hat    = [0.2 2 20;0.1 1 10];
avg_runtime = zeros(1,length(sW_list));
for i = 1:length(sW_list)
    %% Q matrix
    sW_AR      = sqrt(sW_list(i)); %0.42, 1.35, 18.75
    Q_true     = sW_AR^2;
    mw2        = Pr_w2hat(1,i); 
    sw2        = Pr_w2hat(2,i); 
    %% R matrix
    QR_ratio   = 1;                    % Q/R = (\sigma_AR)^2/(\sigma_V)^2
    R          = Q_true/QR_ratio;
    sV         = sqrt(R);
    %% Data
    no_of_datasets = 5;
    runtimes = zeros(1,5);
    for j = 1:no_of_datasets
        filename = sprintf('Datasets_CaseStudy1_ACSP/Data%d_sigmatrue%d.mat',j,i);
        load(filename);
        % save('Datasets_CaseStudy1_ACSP/Data1_sigmatrue2.mat');
        %% State estimation
        EX  =zeros(3,T);                               %   X  = [X^LL  X^AR  W  hat(W^2)]
        EX(:,1)   = [0 nan mw2]';
        PX(:,:,1) = diag([100,nan,sw2^2]);
        
        runtime_s = tic;
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
        %     NIS_NPI(n,t)  = v * inv(SY) * v';
        %     Nees_NPI(n,t) = (x_true(1,t)-EX(1,t))'*inv(PX(1,1,t))*(x_true(1,t)-EX(1,t));
        
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
            %% NEES for each t
            %Nees_v2hat_NPI(n,t) = (sW_AR(n)^2-EX(end,t))'*inv(PX(end,end,t))*(sW_AR(n)^2-EX(end,t));
            %         Nees_W2_NPI(n,t)    = (W2_exp(n,t)-E_W2_pos)'*inv(P_W2_pos)*(W2_exp(n,t)-E_W2_pos);
        
        end
        estim_Q(i,j)   = EX(3,t);
        runtimes(j) = toc(runtime_s);
    end
    avg_runtime(i) = mean(runtimes);
%     disp(['average runtime (in s): ' num2str(avg_runtime)]);
end
disp(['average runtime for all three cases (in s): ' num2str(mean(avg_runtime))]);
% save('Results_casestudy1_ACSP/AGVI_Q.mat','estim_Q')