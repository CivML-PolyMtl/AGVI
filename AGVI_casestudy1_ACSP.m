%% Online Inference of Univariate Process Error Variance for the Linear Time-Varying Model Using AGVI%%
% Section 5.1.1 in the article "Approximate Gaussian Variance Inference for State-Space Models"
%% Authors: Bhargob Deka and James-A. Goulet, 2023 %%
%%
clear;clc
rand_seed=4;
RandStream.setGlobalStream(RandStream('mt19937ar','seed',rand_seed));  %Initialize random stream number based on clock

%% Parameters
T          = 1000;                 % Time-serie length
n_x        = 1;                    % no. of hidden states
n_w        = n_x;                  % no. of process error terms

n_w2hat    = n_x*(n_x+1)/2;        % total number of variance and covariance terms   
y          = zeros(T,1);           % initialization of the vector of observations

%% Q matrix
sW      = sqrt(1.35);           %0.42, 1.35, 18.75, possible values for the true error variances
Q          = sW^2;
%% R matrix
QR_ratio   = 1;                    % Q/R = (\sigma_W)^2/(\sigma_V)^2
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

% save('Datasets_CaseStudy1_ACSP/Data1_sigmatrue2.mat');
%% State estimation
% Initialization for the error variance random variable, i.e., \overline{W^2}
% Note: choose [mw2,sw2]=[0.2,0.1] for sW^2=1.35 in Line 17, [2,1] for sW^2=1.35, and [20,10] for sW^2=18.75;
mw2 = 2; %[0.2, 2, 20]
sw2 = 1; %[0.1, 1, 10]
EX  =zeros(3,T); %   X  = [X  W  \overline{W^2}]
EX(:,1)   = [0 nan mw2]'; % initial mean is zero
PX(:,:,1) = diag([100,nan,sw2^2]); % initial variance is 100


%% Prediction
for t=2:T
    A        = (0.8-0.1*sin(7*pi*t/T)); % Transition equation
    Ep       = [A * EX(1,t-1); 0];      % mean of W is zero     

    m_w2hat  = EX(3,t-1);
    s_w_sq   = m_w2hat;

    Sp       = [A.^2 * PX(1,1,t-1)+s_w_sq    s_w_sq  ;        %  
                            s_w_sq           s_w_sq  ];

    C        = [(1-0.99*sin(100*pi*t/T)) 0]; % observation equation
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
estim_sigma_w   = EX(3,t);
estim_sigma_w_P = PX(3,3,t);
%% Hypothesis Testing
alpha = 0.05; % significance level
df    = T-1; % degrees of freedom
t_critical = tinv(1-alpha/2,df); % t-critical value for two-tailed test

% Calculating the test statistic and p-value
x_bar       = estim_sigma_w;
mu          = sW^2;
s           = sqrt(estim_sigma_w_P);
t_statistic = (x_bar - mu)/(s/sqrt(1));

% Testing the null hypothesis
if abs(t_statistic) > t_critical
    fprintf('Reject the null hypothesis at the %0.2f%% significance level.\n',100*(1-alpha));
else
    fprintf('Fail to reject the null hypothesis at the %0.2f%% significance level.\n',100*(1-alpha));
end
%%  Plotting Sigma_W
t  = 1:length(EX);
xw = EX(3,t);
sX = permute(sqrt(PX(3,3,t)),[1,3,2]);
figure;
plot(t,repmat(sW^2,[1,length(EX)]),'-.r','Linewidth',1.5)
hold on;
patch([t,fliplr(t)],[xw+sX,fliplr(xw-sX)],'g','FaceAlpha',0.2,'EdgeColor','none')
hold on;
plot(t,xw,'k')
hold off
legend('true','$\mu \pm \sigma$','$\mu$','Interpreter','latex')
% ylim([sW_AR^2-sW_AR/2,sW_AR^2+sW_AR^2/2])

xlabel('$t$','Interpreter','latex')
ylabel('$\sigma^2_W$','Interpreter','latex')

