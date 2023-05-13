function est = MLM(sys,z,param)
  %MLM (sys,z,param) Maximum Likelihood Method
  %
  % MLM - Section 3.7
  %
  % based on:
  % R. H. Shumway, D. S. Stoffer, "Time series analysis and its applications",
  % Springer-Verlag, 2000
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP,PARAM.PP describe initial estimate of the state and its covariance
  % PARAM.Qa initial estimate of Q
  % PARAM.Ra initial estimate of R
  % PARAM.M maximum number of EM steps
  runtime_s = tic;
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - SET A PRIORI ESTIMATES of Q and R    
  Qa = param.Qa;
  Ra = param.Ra;
  M = param.M;
  
  % EM algorithm iterations
  for j = 1:M        
    xf = zeros(nx,N);
    Pf = cell(1,N);
    xp = zeros(nx,N+1);
    Pp = cell(1,N+1);
    xs = zeros(nx,N);
    Ps = cell(1,N);
    J = cell(1,N-1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - DESIGN KF AND RTSS / EXPECTATION STEP
    % Kalman filter estimate
    Pp{1} = param.Pp;
    xp(:,1) = param.xp;
    for k = 1:N
      K = Pp{k}*sys.H'/(sys.H*Pp{k}*sys.H'+Ra); % Kalman gain
      % measurement update, filtering
      xf(:,k) = xp(:,k) + K*(z(:,k)-sys.H*xp(:,k));
      Pf{k} = Pp{k} - K*sys.H*Pp{k};
      % time update, prediction
      xp(:,k+1) = sys.F*xf(:,k);
      Pp{k+1} = sys.F*Pf{k}*sys.F' + Qa;
    end
    
    % Rauch-Tung-Striebel smoother
    xs(:,N) = xf(:,N);
    Ps{N} = Pf{N};
    for k = N-1:-1:1
      J{k} = Pf{k}*sys.F'/Pp{k+1};
      % smoothing
      xs(:,k) = xf(:,k) + J{k}*(xs(:,k+1)-xp(:,k+1));
      Ps{k} = Pf{k} + J{k}*(Ps{k+1}-Pp{k+1})*J{k}';
    end
    
    % lag 1 covariance smoother
    P1s = cell(1,N);
    P1s{N} = (eye(nx)-K*sys.H)*sys.F*Pf{N-1};
    for k = N-1:-1:2
      P1s{k} = Pf{k}*J{k-1}' + J{k}*(P1s{k+1}-sys.F*Pf{k})*J{k-1}';
    end
    
    % negative log-likelihood function related parameters
    S11 = zeros(nx);
    S10 = zeros(nx);
    S00 = zeros(nx);
    for k = 2:N
      S11 = S11 + xs(:,k)*xs(:,k)' + Ps{k};
      S10 = S10 + xs(:,k)*xs(:,k-1)' + P1s{k};
      S00 = S00 + xs(:,k-1)*xs(:,k-1)' + Ps{k-1};
    end
    
    %%%%%%%%%%%%%%% STEP III - COMPUTE A POSTERIORI ESTIMATE / MAXIMIZATION STEP
    Qa = (S11-S10/S00*S10')/N;
    Ra = zeros(nz);
    for k = 1:N
      Ra = Ra+(z(:,k)-sys.H*xs(:,k))*(z(:,k)-sys.H*xs(:,k))'+sys.H*Ps{k}*sys.H';
    end
    Ra = Ra/N;
    
  end
  est.R = Ra;
  est.Q = Qa;
  est.runtime_e = toc(runtime_s);