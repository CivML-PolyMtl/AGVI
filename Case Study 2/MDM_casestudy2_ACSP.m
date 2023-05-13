%% Online Inference of a 5x5 Q Matrix using the Measurement Difference Method (MDM)%%
%% --------- Original Code by Jindřich Duník, Oliver Kost, Ondřej Straka (2017) -----------------------
%% Created by Bhargob Deka and James-A. Goulet, 2023
%%
clear;
close all;clc;
format short;
%%% New Parameters
ts  =1000;
%%%%%Model parameters
nxp = 100;
n_x = 5;
nz  = 5;
T   = 1; 
A          = diag(1*(ones(1,n_x)));
%% Q matrix construction
% Q          = diag(rand(n_x,1));%diag(0.5*ones(1,n_x)),diag(rand(n_x,1))
sig        = [-0.3 -0.2 -0.1 0.25 0.35 0.4 0.45 0.5 0.55 0.6];
corr       = [1 sig(1) sig(2) sig(3) sig(4);...
              sig(1) 3 sig(5) sig(6) sig(7);...
              sig(2) sig(5) 4 sig(8) sig(9);...
              sig(3) sig(6) sig(8) 0.8 sig(10);...
              sig(4) sig(7) sig(9) sig(10) 2];
Q          =  eye(5)*corr;
% Q          =  Q*Q';
R          =  0.1*eye(n_x);
H          = eye(n_x);
sys.R      = R;
sys.Q      = Q;
sys.F      = A;
sys.H      = H;
sys.nx     = 5; % state
sys.nz     = 5; % measurement
sys.N      = 1e3;
%% initial estimate
% param.xp = [0;0]; % initial state estimate
param.lags = 1; % # of time lags
% param.K = [0.8 0;0 0.8]; % stable linear filter gain
%% 
no_of_datasets  = 5;
runtimes        = zeros(1,no_of_datasets);
for j = 1:no_of_datasets
    filename     = sprintf('Datasets_CaseStudy2_ACSP/Data%d.mat',j);
    dat          = load(filename);
    z            = dat.YT;
    runtime_s    = tic;
    est          = MDCM(sys,z,param);
    Q_MDM        = est.Q;
    %save(['Results_Q_casestudy2_MDM_ACSP/Q_MDM_Dataset' num2str(j) '.mat'],'Q_MDM')
    runtimes(j)  = toc(runtime_s);
end
disp(['Average runtime is :' num2str(mean(runtimes))])