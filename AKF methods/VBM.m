function [est] = VBM(sys,z,param)
  %ICM (sys,z,param) Variational Bayes Method
  %
  % ICM - Section 4.4
  %
  % based on:
  % S. Sarkka, A Nummenmaa, "Recursive noise adaptive Kalman filtering by
  % variational bayesian approximations", IEEE Transactions on Automatic
  % Control, vol. 54, no. 3, pp. 596-600, 2009.
  %
  % estimates diagonal elements of R
  % SYS.F, SYS.H and sys.Q are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP, PARAM.PP describes initial estimate of the state and its variance
  % PARAM.ALFA, PARAM.BETA are initial estimates of inverse Gamma distribution
  % PARAM.ITCNT determines number of filtering iterations (2-3 are recommended)
  % PARAM.RHO(N) governs behavior of precision factor (recomm.: 1-alfa) alfa->0
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I MODEL and ESTIMATE INITIALIZATION
  xp = param.xp;
  Pp = param.Pp;
  Rest = zeros(nz,N);
  alfap = param.alfa;
  betap = param.beta;
  %%%%%%%%%%%%%%%%%%%%%%%%% STEP II + III RECURSIVE ESTIMATION OF STATE AND CM R 
  for i = 1:N
    alfaf = ones(nz,1)/2+alfap; % parameter adjustment
    betaf = betap; 
    innov = (z(:,i)-sys.H*xp); % innovation
    
    %filtration/iteration
    for j = 1:param.itCnt
      Rest(:,i) = betaf./alfaf; % estimation of R
      K = Pp*sys.H'/(sys.H*Pp*sys.H'+diag(Rest(:,i))); % filter gain
      xf = xp+K*innov; % state estimate filtering update
      Pf = Pp-K*sys.H*Pp; % filtering state covariance update
      betaf = betap+(z(:,i)-sys.H*xf).^2/2+diag(sys.H*Pf*sys.H')/2;% par. change
    end
    
    %prediction
    xp = sys.F*xf; % state estimate time update
    Pp = sys.F*Pf*sys.F'+sys.Q; % variance time update
    alfap = param.rho(N)*alfaf; % parameter alfa time update
    betap = param.rho(N)*betaf; % parameter beta time update
  end
  est.R = Rest;
end

