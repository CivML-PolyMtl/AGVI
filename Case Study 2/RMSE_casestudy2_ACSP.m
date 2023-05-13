%% RMSE Comparison for AGVI with SWVAKF and MDM
%% Created by Bhargob Deka and James-A. Goulet, 2023 %
%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
format short
n_x         = 5;                    % no. of hidden states
n_w         = n_x;                  % no. of process error terms
n_w2hat     = n_x*(n_x+1)/2;        % total variance and covariance terms   
sig         = [-0.3 -0.2 -0.1 0.25 0.35 0.4 0.45 0.5 0.55 0.6];
corr        = [1 sig(1) sig(2) sig(3) sig(4);...
              sig(1) 3 sig(5) sig(6) sig(7);...
              sig(2) sig(5) 4 sig(8) sig(9);...
              sig(3) sig(6) sig(8) 0.8 sig(10);...
              sig(4) sig(7) sig(9) sig(10) 2];
Q           =  eye(5)*corr;
% Q          =  Q*Q';
RT          =  0.1*eye(n_x);
QT          =  Q;
ind         =  tril(true(size(QT)));
True_var    = QT(ind);
methods = {'AGVI','SWVBAKF','MDM'};
M       = length(methods);
Q       = cell(M,1);

Cov     = zeros(n_w2hat,M);
% RMSE    = zeros(n_w2hat,M);
N = 1;
for j = 1:N % for each synthetic datasets
    for i = 1:length(methods)
        Q{i} = load(['Results_Q_casestudy2_' methods{i} '_ACSP/Q_' methods{i} '_Dataset' num2str(j) '.mat']);
        if strcmp(methods{i},'AGVI')
            Q_est = Q{i,1}.est_Q;
        elseif strcmp(methods{i},'SWVBAKF')
            Q_est = Q{i,1}.new_Qapriv;
        elseif strcmp(methods{i},'MDM')
            Q_est = Q{i,1}.Q_MDM;
        end
        ind = tril(true(size(Q_est)));
        Cov(:,i)  = Q_est(ind);
        
    end
    CovT = Cov'; % one row represent all variances for one method
    % Cov for each method
    CovT_AGVI(j,:)    = CovT(1,:);
    CovT_SWVBAKF(j,:) = CovT(2,:);
    CovT_MDM(j,:)     = CovT(3,:);
end
for k = 1:length(True_var)
RMSE_AGVI(1,k)      = sqrt(mean((True_var(k) - CovT_AGVI(:,k)).^2));
RMSE_MDM(1,k)       = sqrt(mean((True_var(k) - CovT_MDM(:,k)).^2));
RMSE_SWVBAKF(1,k)   = sqrt(mean((True_var(k) - CovT_SWVBAKF(:,k)).^2));
end
RMSE = [RMSE_AGVI;RMSE_SWVBAKF;RMSE_MDM];
[~,N] = min(RMSE);

label={'var1';'cov12';'cov13';'cov14';'cov15';'var2';'cov23';'cov24';'cov25';'var3';'cov34';'cov35';'cov45';'var5'}';
disp(label)
disp(N)




