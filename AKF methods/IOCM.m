function est = IOCM(sys,z,param)
  % IOCM (sys,z,param) Input - Output Correlation Method
  %
  % IOCM - Section 3.2
  %
  % based on:
  % R. L. Kashyap, "Maximum likelihood identification of stochastic linear
  % systems", IEEE Transactions on Automatic Control, vol. 15, no. 1, pp. 25-34,
  % 1970.
  %
  % estimates Q and R in a two step procedure
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.B0 initial condition for estimating B
  
  nx = size(sys.F,2); % obtain state dimension
  N = size(z,2); % obtain number of measurements
  
  %%%%%%%%%%%%%%%%%%%%%%% STEP I + II - ESTIMATE COVARIANCE MATRIX OF INNOVATION
  % compute optimal measurement prediction parameter
  [Bmin] = fminsearch(@(Bseeked) objective_function(Bseeked,z,sys.F),param.B0);
  Be = [Bmin(1:2);Bmin(3:4)]; % rearrange array
  
  % compute measurement prediction error
  e = zeros(nx,N); % create empty array
  for i = 2:N
    e(:,i) = z(:,i)  -sys.F*z(:,i-1) -Be*e(:,i-1); % compute m. prediction error
  end
  C0 = (e*e')/N; % estimate variance C0
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE CMs Q AND R
  est.R = -sys.F\(Be*C0); % estimate R
  est.R = (est.R+est.R')/2; % symmetrise
  est.Q = C0 + Be*C0*Be' - sys.F*est.R*sys.F' - est.R; % estimate Q
  est.Q = (est.Q+est.Q')/2; % symmetrise
end

function [val] = objective_function(B,z,F)
  % prediction error method
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  val =  zeros(size(B,1),1); % create empty array
  for j = 1:size(B,1)
    Bmat = [B(j,1:2);B(j,3:4)]; % rearrange array
    e = zeros(nz,N); % create empty array
    for i = 2:N
      e(:,i) = z(:,i)  -F*z(:,i-1) -Bmat*e(:,i-1); % compute prediction error
    end
    val(j) = det((e*e')/N); % estimate variance
  end
end
