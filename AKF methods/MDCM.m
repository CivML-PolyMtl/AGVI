function [est] = MDCM(sys,z,param)
  %MDCM (sys,z,param) Measurement Difference Correlation Method
  %
  % MDCM - Section 3.6
  %
  % based on:
  % J. Dunik, O. Straka, O. Kost "Measurement difference autocovariance
  % for noise covariance matrices estimation method",
  % accepted for the 55th IEEE Conference on Decision and Control, 2016.
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describes initial estimate of the state
  % PARAM.LAGS time lag of autocovariance function

  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE AUTOCOVARIANCE FUNCTION
  % Prediction error
  ePred = zeros(nz,N);
  for i = (param.lags+1):N
    ePred(:,i) = z(:,i)-sys.H*sys.F^(param.lags)*pinv(sys.H)*z(:,i-param.lags);
  end
  
  % Prediction error covariance
  b = zeros(nz^2*(param.lags+1),N);
  for i = 2*param.lags+1:N
    for j = 0:param.lags
      b(1+j*nz^2:(j+1)*nz^2,i) = reshape(ePred(:,i)*ePred(:,i-j)',nz^2,1);
    end
  end
  b = mean(b(:,2*param.lags+1:end),2);% Average of prediction error covariances
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - COMPUTE MATRIX F and ESTIMATE Q and R
  % computation of the least-squares data matrix
  F = zeros((param.lags+1)*nz^2,nx^2+nz^2);
  for iL = 1:param.lags+1
    if(iL==param.lags+1)
      Aa = kron(eye(nz),-sys.H*sys.F^(param.lags)*pinv(sys.H));
      F(param.lags*nz^2+1:(param.lags+1)*nz^2,:) = [zeros(nx^2,nz^2) Aa];
    elseif(iL ==1)
      Aa = zeros(nz^2,nx^2);
      for k = 0:param.lags-1
        Aa = Aa+kron(sys.H*sys.F^(k),sys.H*sys.F^(k));
      end
      Ab = kron(sys.H*sys.F^(param.lags)*pinv(sys.H),...
        sys.H*sys.F^(param.lags)*pinv(sys.H))+eye(nz^2);
      F(1:nz^2,:) = [Aa Ab];
    else
      Aa = zeros(nz^2,nx^2);
      for k = 0:param.lags-iL
        Aa = Aa+kron(sys.H*sys.F^(k),sys.H*sys.F^(k+(iL-1)));
      end
      F(nz^2*(iL-1)+1:nz^2*iL,:) = [Aa zeros(nz^2)];
    end
  end
  
  % estimate CMs Q and R
  estQR = F\b; % least-squares estimate
  est.Q = reshape(estQR(1:nx^2),nx,nx);
  est.Q = (est.Q+est.Q')/2; %symmetrise
  est.R = reshape(estQR(nx^2+1:end),nz,nz);
  est.R = (est.R+est.R')/2; %symmetrise
end

