function [est] = ICM(sys,z,param)
  %ICM (sys,z,param) Indirect Correlation Method
  %
  % ICM - Section 3.1
  %
  % based on:
  % R. K. Mehra, "On the identification of variances and adaptive filtering",
  % IEEE Transactions on Automatic Control, vol. 15, no. 2, pp. 175-184, 1970.
  %
  % estimates Q and R in a two step procedure
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describes initial estimate of the state
  % PARAM.LAGS time lag of autocovariance function
  % PARAM.K stable linear filter gain
  start=tic;
  [nz,N] = size(z);  % obtain measurement dimension and number of measurements
  nx     = size(sys.F,2);  % obtain state dimension
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE AUTOCOVARIANCE FUNCTION
  % compute innovation sequence for the stable linear gain PARAM.K
  innov = LF(sys,param.xp,param.K,z);
  % estimate autocovariance of the innovation sequence for lag 0
  Chat0 = innov(:,1:end)*innov(:,1:end)'/N; 
  % estimate autocovariance of the innovation sequence for lag 1 ... PARAM.LAGS
  Chat = zeros(param.lags*nz,nx); % create empty array
  for k = 0:param.lags-1
    Chat(k*nz+1:(k+1)*nz,:) = innov(:,k+2:N)*innov(:,1:N-k-1)'/(N-k-1);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - ESTIMATE CMs Q and R
  %% estimate R
  % compute regressors
  F = zeros(nz*param.lags,nz); % create empty array
  Fbar = sys.F*(eye(nx)-param.K*sys.H);%state prediction error transition matrix
  for k = 0:param.lags-1
    F(k*nz+1:(k+1)*nz,:) = sys.H*Fbar^(k)*sys.F; % auxiliary variable F
  end
  PHT = param.K*Chat0+F\Chat; % estimate PH'
  est.R = Chat0-sys.H*PHT; % return estimate R
  est.R = (est.R+est.R')/2; % symmetrise
  %% estimate Q
  a = zeros(param.lags*nz^2,nx^2);  % create empty array
  for k = 1:param.lags
    c = zeros(nz^2,nx^2); % create empty array
    for j = 0:k-1
      c = c+kron(sys.H*sys.F^(j-k),sys.H*sys.F^(j)); % auxiliary variable c
    end
    a((k-1)*nz^2+1:k*nz^2,:) = c; % auxiliary variable a
  end
  omega = sys.F*(-param.K*PHT-PHT'*param.K'+param.K*Chat0*param.K')*sys.F';% aux
  b = zeros(param.lags*nz^2,1); % create empty array
  for k = 1:param.lags
    d = PHT'*sys.F^(-k)'*sys.H'-sys.H*sys.F^k*PHT; % auxiliary variable d
    for j = 1:k
      d = d-sys.H*sys.F^(j-1)*omega*sys.F^(-j)'*sys.H';
    end
    b((k-1)*nz^2+1:k*nz^2) = d(:); % auxiliary variable b
  end
  est.Q = reshape(a\b,nx,nx); % return estimate Q
  est.Q = (est.Q+est.Q')/2; % symmetrise
  runtime=toc(start);
  est.runtime=runtime;
end

