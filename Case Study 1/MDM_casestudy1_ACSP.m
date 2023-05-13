%% Online Inference of Q Matrix using the Measurement Difference Method (MDM)%%
%% --------- Original Code by Jindřich Duník, Oliver Kost, Ondřej Straka (2019) -----------------------
%% Created by Bhargob Deka and James-A. Goulet, 2023
%%
clear;clc
% rand_seed=4;
% RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
%% System parameters
nx  = 1;
T   = 1000+1;
F = zeros(nx,T); % state
H = zeros(nx,T); % measurement
for k = 2:T
    F(k) = (0.8-0.1*sin(7*pi*k/T));
    H(k) = (1-0.99*sin(100*pi*k/T));
end 
F(1)=[]; H(1)=[];

sW_list = [0.42 1.35 18.75];
estim_Q = zeros(3,5);
avg_runtime = zeros(1,length(sW_list));
for i = 1:length(sW_list)
    %% Q matrix
    sW      = sqrt(sW_list(i));%0.42, 1.35, 18.75
    Q_true     = sW^2;
    %% R matrix
    QR_ratio   = 1;                    % Q/R = (\sigma_AR)^2/(\sigma_V)^2
    R_true     = Q_true/QR_ratio;
    
    sys.R = R_true;
    sys.Q = Q_true;
    sys.F = F;
    sys.H = H;
    sys.nx = 1; % state
    sys.nz = 1; % measurement
    % - number of data
    sys.N = 1e3;
    % initial estimate
    param.xp        = zeros(nx,1); % initial state estimate
    param.lags      = 1; % # of time lags
    param.K         = diag(0.8*ones(1,nx)); % stable linear filter gain
    no_of_datasets  = 5;
    runtimes        = zeros(1,5);
    for j = 1:no_of_datasets
        filename     = sprintf('Datasets_CaseStudy1_ACSP/Data%d_sigmatrue%d.mat',j,i);
        dat          = load(filename);
        z            = dat.YT;
        runtime_s    = tic;
        est          = MDM_LTV(sys,z);
        estim_Q(i,j) = est.Qhat;
        runtimes(j)  = toc(runtime_s);
    end
    avg_runtime(i) = mean(runtimes);
%     disp(['average runtime (in s): ' num2str(avg_runtime)]);
end
disp(['average runtime for all three cases (in s): ' num2str(mean(avg_runtime))]);
%save('Results_casestudy1_ACSP/MDM_Q.mat','estim_Q')
