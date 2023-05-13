%% Online Inference of Q Matrix using Slide Window Variational Adaptive Kalman Filter (VBAKF)%%
%% ---------Original Code by Huang et al. (2019) -----------------------
%% Created by Bhargob Deka and James-A. Goulet, 2023
%%
clear;
close all;clc;
format short;
nx  = 1;
nz  = 1;
T   = 1000+1;
F_T = zeros(nx,T); % state
H_T = zeros(nx,T); % measurement
for k = 2:T
    F_T(k) = (0.8-0.1*sin(7*pi*k/T));
    H_T(k) = (1-0.99*sin(100*pi*k/T));
end 
F_T(1)=[]; H_T(1)=[];
sW_list     = [0.42 1.35 18.75];
estim_Q     = zeros(3,5);
avg_runtime = zeros(1,length(sW_list));
for i = 1:length(sW_list)
%% Q matrix
sW         = sqrt(sW_list(i));%0.42, 1.35, 18.75
Q_true     = sW^2;
%% R matrix
QR_ratio   = 1;                    % Q/R = (\sigma_W)^2/(\sigma_V)^2
R_true     = Q_true/QR_ratio;
Q          = Q_true;
R          = R_true;
Q1         = Q;
R1         = R;
%% Selections of filtering parameters as per the original code
ts    = 1000;
nxp   = 100;
rou   = 1-exp(-4);
beta  = 10;
n     = 50;                 
tao_P = 5;  
tao_R = 5; 
Q_s   = 10;
alfa  = 1;

%% Loading data
no_of_datasets = 5;
runtimes = zeros(1,5);
    for j = 1:no_of_datasets
    filename = sprintf('Datasets_CaseStudy1_ACSP/Data%d_sigmatrue%d.mat',j,i);
    load(filename);
        %% Monte-Carlo Runs
        runtime_s = tic;
        for expt=1:nxp
            
            fprintf('MC Run in Process=%d\n',expt); 
            
            %%%%%Set the system initial value%%%%%%%%%%%
            x=zeros(nx,1);                     %%%True initial state value 
            P=diag(100*ones(nx,1));              %%%Initial estimation error covariance matrix 
            Skk=utchol(P);                         %%%Square-root of initial estimation error covariance matrix
            
            %%%%Nominal measurement noise covariance matrix (R)
            R0=beta*eye(nz);
            
            %%%%Initial state estimate of standard KF    (Kalman filter)
            xf=x+Skk*randn(nx,1);               
            Pf=P;
            
            %%%%Initial state estimate of Kalman filter with true noise covariance matrices    (Optimal Kalman filter)
            xtf=xf;
            Ptf=Pf;
        
            %% Initial state estimate of adaptive methods
            xiv=xf;
            Piv=Pf;
        
            %%%%
            xA=[];
            ZA=[];
            
            
            %%%%Nominal state noise covariance matrix (Q)
            Q0=alfa*eye(nx);
            
            %%%%Initial state estimate of VBAKF-PR
            xapriv=xiv;
            Papriv=Piv;
            uapriv=tao_R;
            Uapriv=tao_R*R0;
            
            %%%%Initial state estimate of proposed SWVAKF
            new_xapriv=xiv;
            new_Papriv=Piv;
            new_xapriv_A=[];
            new_Papriv_A=[];
            new_yapriv=0;
            new_Yapriv=zeros(nx);
            new_uapriv=0;
            new_Uapriv=zeros(nz);
            new_Qapriv=Q0;
            new_Rapriv=R0;
            new_Lapriv=5;
            zA=[];
    
            for t=1:ts
                
                %%%%True noise covariance matrices
                Q=Q1;
                R=R1;
                
                %%%%Extract data
    %             x = x_true(:,t);
                z = YT(:,t);            
                F = F_T(:,t);
                H = H_T(t);
                %%%%Save measurement data
                if t<=(new_Lapriv+1)
                    zA=[zA z];
                else
                    zA=[zA(:,2:end) z];
                end
                
                %%%%Run VBAKF-PR and proposed SWVAKF
    %             [xapriv,Papriv,uapriv,Uapriv,Ppapriv,Rapriv,Qapriv]=aprivbkf(xapriv,Papriv,uapriv,Uapriv,F,H,z,Q0,R,N,tao_P,rou);
                
                [new_xapriv,new_Papriv,new_xapriv_A,new_Papriv_A,new_yapriv,new_Yapriv,new_uapriv,new_Uapriv,new_Qapriv,R,new_Ppapriv]=...
                new_aprivbkf(new_xapriv,new_Papriv,new_xapriv_A,new_Papriv_A,new_yapriv,new_Yapriv,new_uapriv,new_Uapriv,F,H,zA,new_Qapriv,R,rou,new_Lapriv,t);
    
                
                
            end
                
            
            
        end
        estim_Q(i,j) = new_Qapriv;
        runtimes(j) = toc(runtime_s);
    %     save(['Q_SWVBAKF_results/Q_SWVBAKF_Dataset' num2str(j) '.mat'],'new_Qapriv')
    end
    avg_runtime(i) = mean(runtimes);
%     disp(['average runtime (in s): ' num2str(avg_runtime)]);
end
disp(['average runtime for all three cases (in s): ' num2str(mean(avg_runtime))]);
% save('Results_casestudy1_ACSP/VBAKF_Q.mat','estim_Q')