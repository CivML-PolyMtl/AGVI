function est = MACM(sys,z,param)
  %MACM (sys,z,param) Measurement Averaging Correlation Method
  %
  % MACM - Section 3.4
  %
  % based on:
  % J. Zhou, R.H. Luecke, "Estimation of the covariances of the process noise
  % and measurement noise for a linear discrete dynamic system", Computers &
  % Chemical Engineering, vol. 19, no. 2, pp. 187-195, 1995.
  %
  % !!! implementation suited only for 2 dimensional models as in the paper
  %
  % estimates diagonal elements of Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.W1, PARAM.W2 weight matrices
  
  nx = size(sys.F,2); % obtain state dimension
  N = size(z,2); % obtain number of measurements
  
  % user defined weights
  W1 = param.W1; 
  W2 = param.W2;
  
  % empty arrays
  xs1 = zeros(nx,N-2);
  zs1 = zeros(nx,N-2);
  es1 = zeros(nx,N-2);
  xs2 = zeros(nx,N-2);
  zs2 = zeros(nx,N-2);
  es2 = zeros(nx,N-2);
  
  %%%%%%%%%% STEP I - COMPUTE THE COVARIANCE MATRICES OF MEASUREMENT PRED. ERROR
  for k = 1:N-2
    % weighted least squares(WLS) estimate of state and measurement
    z1 = [z(:,k); z(:,k+1)];
    O1 = [sys.H; sys.H*sys.F];
    pO1 = (O1'*W1*O1)\O1'*W1; % pseudoinverse of O1
    % split for further use
    pO1a = pO1(1:nx,1:nx);
    pO1b = pO1(1:nx,nx+1:2*nx);    
    xs1(:,k) = pO1*z1; % WLS estimate of state
    zs1(:,k) = sys.H*xs1(:,k); % WLS estimate of measurement
    es1(:,k) = z(:,k) - zs1(:,k); % measurement prediction error
    
    % WLS estimate of state and measurement for shifted data
    z2 = [z(:,k); z(:,k+1); z(:,k+2)]; 
    O2 = [sys.H; sys.H*sys.F; sys.H*sys.F^2];
    pO2 = (O2'*W2*O2)\O2'*W2;% pseudoinverse of O2
    % split for further use
    pO2a = pO2(1:nx,1:nx);
    pO2b = pO2(1:nx,nx+1:2*nx);
    pO2c = pO2(1:nx,2*nx+1:3*nx);
    
    xs2(:,k) = pO2*z2;% WLS estimate of state
    zs2(:,k) = sys.H*xs2(:,k);% WLS estimate of measurement
    
    % measurement prediction error
    es2(:,k) = z(:,k) - zs2(:,k);
  end
  
  % estimate sample CM
  C1est = cov(es1');
  C2est = cov(es2');
  bN = [C1est(:);C2est(:)];
  
  %%%%%%%%% STEP II -  SOLVE SYSTEM OF LINEAR EQUATIONS and ESTIMATE CMs Q and R
  
  % Kronecker product based computation of LS data matrix
  a11 = kron(sys.H*pO1a,sys.H*pO1a)+kron(sys.H*pO1b,sys.H*pO1b)...
    -kron(eye(nx),sys.H*pO1a)-kron(sys.H*pO1a,eye(nx))+kron(eye(nx),eye(nx)); 
  a12 = kron(sys.H*pO1b*sys.H,sys.H*pO1b*sys.H); 
  a21 = kron(sys.H*pO2a,sys.H*pO2a)+kron(sys.H*pO2b,sys.H*pO2b)...
    +kron(sys.H*pO2c,sys.H*pO2c)+kron(eye(nx),eye(nx))...
    -kron(eye(nx),sys.H*pO2a)-kron(sys.H*pO2a,eye(nx)); 
  a22 = kron(sys.H*pO2b*sys.H,sys.H*pO2b*sys.H)...
    +kron(sys.H*pO2b*sys.H,sys.H*pO2c*sys.H*sys.F)...
    +kron(sys.H*pO2c*sys.H*sys.F,sys.H*pO2b*sys.H)...
    +kron(sys.H*pO2c*sys.H*sys.F,sys.H*pO2c*sys.H*sys.F)...
    +kron(sys.H*pO2c*sys.H,sys.H*pO2c*sys.H);
  AN = [a11 a12; a21 a22]; % merge results
   
  % artificial restriction of the method for estimation of diagonal elements 
  %  of Q and R as assumed in the baseline paper
  ANred = AN(:,[1 4 5 8]);
  ANrem = AN(:,[2 3 6 7]);
  bNred = bN - ANrem*[sys.R(2,1), sys.R(1,2), sys.Q(2,1), sys.Q(1,2)]';
  RQest = pinv(ANred)*bNred;
  
  est.R = [RQest(1), NaN; NaN, RQest(2)];
  est.Q = [RQest(3), NaN; NaN, RQest(4)];

