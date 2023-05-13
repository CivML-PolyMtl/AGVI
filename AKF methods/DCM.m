function [est] = DCM(sys,z,param)
  %DCM (sys,z,param) Direct Correlation Method
  %
  % DCM - Section 3.5
  %
  % based on:
  % B. J. Odelson, M. R. Rajamani, J. B. Rawlings, "A new autocovariance
  % least-squares method for estimating noise covariances", Automatica,
  % vol. 42, no. 2, pp. 303-308, 2006
  %
  % estimates Q and R in one-step procedure, keeping symmetry of Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describe initial estimate of the state
  % PARAM.Neq number of equations taken into account
  % PARAM.K stable linear filter gain
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - ESTIMATE AUTOCOVARIANCE FUNCTION
  % compute innovation sequence for the stable linear gain PARAM.K
  innov = LF(sys,param.xp,param.K,z);
  % estimate autocovariance of the innovation sequence for lag 1 ... PARAM.LAGS
  Chat = zeros(nz*param.Neq,nz,1);
  for j = 0:param.Neq-1
    Chat(j*nz+1:(j+1)*nz,:) = innov(:,j+1:N)*innov(:,1:N-j)'/(N-j);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP II - ESTIMATE CMs Q and R
  Fbar = sys.F*(eye(nx)-param.K*sys.H);%state prediction error transition matrix
  
  % stacking of [H,HFbar,HFbar^2,...]
  Oo  =  zeros(param.Neq*nz,nx);
  Oo(1:nz,:) = sys.H;
  for k = 1:param.Neq-1
    Oo(k*nz+1:(k+1)*nz,:) = Oo((k-1)*nz+1:k*nz,:) * Fbar;
  end
  
  % stacking of [-HFbarK,-HFbarFK,...]
  Gamma = zeros(nz*param.Neq,nz);
  Gamma(1:nz,:) = eye(nz);
  for k = 1:param.Neq-1
    Gamma(k*nz+1:(k+1)*nz,:) = -sys.H*Fbar^(k-1)*sys.F*param.K;
  end
  
  D = kron(sys.H,Oo)/(eye(nx^2)-kron(Fbar,Fbar));
  Fa1 = D;
  Fa2 = D*kron(sys.F*param.K,sys.F*param.K)+kron(eye(nz),Gamma);
  
  % symmetric matrix enforcement
  % conversion matrices from n^2 elements -> (n+1)n/2 construction
  t = logical(eye(nx^2));
  A = ones(nx); A(:) = 1:nx^2; B = A';% index matrices A and B
  % symmetrisation matrix
  Fa1Sym = t(:,A(triu(ones(nx))~= 0))|t(:,B(triu(ones(nx))~=0)); 
  
  t = logical(eye(nz^2));
  A = ones(nz); A(:) = 1:nz^2; B = A';% index matrices A and B
  % symmetrisation matrix
  Fa2Sym = t(:,A(triu(ones(nz))~=0))|t(:,B(triu(ones(nz))~=0));
  
  Fnc = [Fa1*Fa1Sym Fa2*Fa2Sym]; % enforcing symmetry
  
  estQR = Fnc\Chat(:); % least-squares
  est.Q = reshape(estQR(1:nx*(nx+1)/2)'*Fa1Sym',nx,nx);
  est.R = reshape(estQR(nx*(nx+1)/2+1:end)'*Fa2Sym',nz,nz);
end

