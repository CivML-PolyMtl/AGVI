%% Online Inference of Univariate Process Error Variance for the Linear Time-Varying Model Using AGVI%%
% Section 5.1.5 in the article "Approximate Gaussian Variance Inference for State-Space Models"
% -- Comparing RMSE values obtained using AGVI, SWVAKF, and MDM
%% Authors: Bhargob Deka and James-A. Goulet, 2023 %%

%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock
format short
True_var    = [0.42;1.35;18.75];
AGVI  = load("Results_casestudy1_ACSP/AGVI_Q.mat");
MDM   = load("Results_casestudy1_ACSP/MDM_Q.mat");
VBAKF = load("Results_casestudy1_ACSP/VBAKF_Q.mat");
%% Q %%
Q_AGVI  = AGVI.estim_Q';
Q_MDM   = MDM.estim_Q';
Q_VBAKF = VBAKF.estim_Q';
RMSE = zeros(3,3);
for k = 1:length(True_var)
RMSE(1,k)  = sqrt(mean((True_var(k) - Q_AGVI(:,k)).^2));%AGVI
RMSE(2,k)  = sqrt(mean((True_var(k) - Q_VBAKF(:,k)).^2));%VBAKF 
RMSE(3,k)  = sqrt(mean((True_var(k) - Q_MDM(:,k)).^2));%MDM
end
RMSE
