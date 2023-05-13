function [est] = WCM(sys,z,param)
  %WCM (sys,z,param) Weighted Correlation Method
  %
  % WCM - Section 3.3
  %
  % based on:
  % R. Belanger, "Estimation of noise covariance matrices for a linear
  % time-varying stochastic process", Automatica, vol. 10, no. 3, pp. 267-275,
  % 1974.
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describes initial estimate of the state
  % PARAM.NEQ number of processed equations
  % PARAM.K stable filter gain
  % PARAM.QAPR basis matrices for Q
  % PARAM.RAPR basis matrices for R
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE AUTOCOVARIANCE FUNCTION
  % compute innovation sequence for the stable linear gain PARAM.K
  innov = LF(sys,param.xp,param.K,z);
  %estimate autocovariance of the innovation sequence for 1..PARAM.NEQ equations
  Chat = zeros(nz^2*param.Neq,1); % create empty array
  for k = 0:param.Neq-1
    temp = innov(:,k+1:N)*innov(:,1:N-k)'/(N-k);
    Chat(k*nz^2+1:(k+1)*nz^2) = reshape(temp,nz^2,1); % rearrange for kronecker
  end
  
  F = sys.F*(eye(nx)-param.K*sys.H); % state prediction error transition matrix
  LAMBDA = size(param.Rapr,3);  % number of parameters LAMBDA 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - COMPUTE Fi
  Fi = zeros(nz^2*param.Neq,LAMBDA); % create empty array
  for i = 1:LAMBDA
    % auxiliary variable Si
    Si = (eye(nx*nx)-kron(F,F))\(reshape(param.Qapr(:,:,i)+...
            +sys.F*param.K*param.Rapr(:,:,i)*param.K'*sys.F',nx*nx,1));
    for k = 1:param.Neq
      if(k==1)
        Fi(1:nz^2,i) = kron(sys.H,sys.H)*Si+reshape(param.Rapr(:,:,i),nz*nz,1);
      else
        Fi(1+(k-1)*nz^2:nz^2+(k-1)*nz^2,i) = kron(sys.H,sys.H*F^(k-1))*Si...
         -kron(eye(nz),sys.H*F^(k-2)*sys.F*param.K)...
          *reshape(param.Rapr(:,:,i),nz*nz,1);
      end
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE CMs Q and R
  estQR = Fi\Chat; % least squares
  est.Q = reshape(estQR(1:nx^2),nx,nx); % return estimate Q
  est.Q = (est.Q+est.Q')/2; % symmetrise
  est.R = reshape(estQR(nx^2+1:end),nz,nz); % return estimate R
  est.R = (est.R+est.R')/2; % symmetrise
end

