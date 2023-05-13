function est=CMM(sys,z,param)
  %CMM (sys,z,param) Covariance Matching Method
  %
  % DCM - Section 4.1
  %
  % based on:
  % K. A. Myers, B. D. Tapley, "Adaptive sequential estimation with unknown
  % noise statistics", IEEE Transactions on Automatic Control,
  % vol. 21, no. 8, pp. 520-523, 1976
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP, PARAM.PP describe initial estimate of the state
  % PARAM.Q, PARAM.R initial estimates of Q and R
  % PARAM.EQR initial time instant for matrices estimation
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  runtime_s = tic;
  xf = zeros(nx,N);
  Pf = zeros(nx,nx,N);
  innov = zeros(nz,N);
  R = zeros(nz,nz,N);
  Gam = zeros(nz,nz,N);
  de = zeros(nx,nx,N);
  Q = zeros(nx,nx,N);
  q = zeros(nx,N);
  r = zeros(nz,N);
  K = zeros(nx,nz,N);
  
  xp  = param.xp;
  erq = param.erq;
  Pp  = param.Pp;
  Q(:,:,1)   = param.Q;
  R(:,:,1)   = param.R;
  Q(:,:,erq) = param.Q;
  R(:,:,erq) = sys.R;
  
  for i = 1:N
    if i <= erq
      % filter algorithm (no Q,R estimation)
      
      % filtering step
      K(:,:,i) = Pp*sys.H'/(sys.H*Pp*sys.H'+sys.R);
      innov(:,i) = z(:,i)-sys.H*xp;
      xf(:,i) = xp+K(:,:,i)*innov(:,i);
      Pf(:,:,i) = (eye(nx)-K(:,:,i)*sys.H)*Pp;
      
      % auxiliary variables for R estimation
      r(:,i) = innov(:,i);
      Gam(:,:,i) = sys.H*Pp*sys.H';
      
      % auxiliary variables for Q estimation
      if i > 1
        q(:,i) = xf(:,i)-sys.F*xf(:,i-1);
        de(:,:,i) = sys.F*Pf(:,:,i-1)*sys.F'-Pf(:,:,i);
      end
      
      % prediction step
      xp = sys.F*xf(:,i);
      Pp = sys.F*Pf(:,:,i)*sys.F'+Q(:,:,1);
    else
      % CMM Algorithm
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE THE MEASUREMENT NOISE CM R
       r(:,i) = z(:,i)-sys.H*xp;
       Gam(:,:,i) = sys.H*Pp*sys.H';
%       R(:,:,i) = cov(r(:,i-erq:i)') - mean(Gam(:,:,i-erq:i),3);
      
      %%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - ESTIMATE THE FILTERING STATE ESTIMATE
      K(:,:,i) = Pp*sys.H'/(sys.H*Pp*sys.H'+sys.R);
      innov(:,i) = r(:,i);
      xf(:,i) = xp+K(:,:,i)*innov(:,i);
      Pf(:,:,i) = (eye(nx)-K(:,:,i)*sys.H)*Pp;
      
      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE THE STATE NOISE CM Q
      q(:,i)    = xf(:,i)-sys.F*xf(:,i-1);
      de(:,:,i) = sys.F*Pf(:,:,i-1)*sys.F'-Pf(:,:,i);
      Q(:,:,i)  = cov(q(:,i-erq:i)') - mean(de(:,:,i-erq:i),3);
      
      %%%%%%%%%%%%%%%%%%%%%%%%% STEP IV - ESTIMATE THE PREDICTION STATE ESTIMATE
      xp = sys.F*xf(:,i);
      Pp = sys.F*Pf(:,:,i)*sys.F'+Q(:,:,i);
    end
  end
  est.Q = squeeze(Q(:,:,N));
  est.R = squeeze(R(:,:,N));
  est.runtime_e = toc(runtime_s);
end
