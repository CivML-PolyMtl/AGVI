function [est] = GMBM(sys,z,param)
  %GMM (sys,z,param) Gaussian Mixture Bayesian Method
  %
  % GMM - Section 4.2
  %
  % based on:
  % D. G. Lainiotis, "Optimal adaptive estimation: Structure and parameters
  % adaptation", IEEE Transactions on Automatic Control, vol. 16, no. 2,
  % pp. 160-170, 1971.
  %
  % estimates Q and R
  % SYS.F, SYS.H are system matrices
  % Z is nz/N matrix of measurements from N time instants
  % PARAM.XP describes initial estimate of the state
  % PARAM.PP describes initial estimate of the state variance
  % PARAM.Qquant quantised matrices for Q
  % PARAM.Rquant quantised matrices for R
  
  [nz,N] = size(z); % obtain measurement dimension and number of measurements
  nx = size(sys.F,2); % obtain state dimension
  
  nq = size(param.Qquant,3); % number of basis matrices for Q
  nr = size(param.Rquant,3); % number of basis matrices for R
  
  weight = ones(nq*nr,N);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP I - DESIGN SET OF nq*nr LINEAR FILTERS
  xp = cell(nq*nr);
  Pxp = cell(nq*nr);
  xf = cell(nq*nr);
  Pxf = cell(nq*nr);
  for i = 1:nq
    for j = 1:nr
      xp{i+(j-1)*nq} = param.xp;
      Pxp{i+(j-1)*nq} = param.Pp;
      xf{i+(j-1)*nq} = zeros(size(param.xp));
      Pxf{i+(j-1)*nq} =zeros(size(param.Pp));
    end
  end
  
  %%%%%%%%%% STEP II - RUN EACH FILTER AND EVALUATE THE A POSTERIORI PROBABILITY
  for k = 1:N
    for i = 1:nq
      for j = 1:nr
        % evaluation of the likelihood function
        weight(i+(j-1)*nq,k) = 1/(sqrt((2*pi)^nz*det(param.Rquant(:,:,j))))...
          *exp(-0.5*(z(:,k)-sys.H*xp{i+(j-1)*nq})'...
          /(param.Rquant(:,:,j))*(z(:,k)-sys.H*xp{i+(j-1)*nq}));
        % filtering
        K = Pxp{i+(j-1)*nq}*sys.H'...
          /(sys.H*Pxp{i+(j-1)*nq}*sys.H'+param.Rquant(:,:,j));
        xf{i+(j-1)*nq} = xp{i+(j-1)*nq}+K*(z(:,k)-sys.H*xp{i+(j-1)*nq}); 
        Pxf{i+(j-1)*nq} = (eye(nx)-K*sys.H)*Pxp{i+(j-1)*nq};
        % prediction
        xp{i+(j-1)*nq} = sys.F*xf{i+(j-1)*nq}; 
        Pxp{i+(j-1)*nq} = sys.F*Pxf{i+(j-1)*nq}*sys.F'+param.Qquant(:,:,i);
      end
    end
    if k > 1
      weight(:,k) =weight(:,k).*weight(:,k-1);
    end
    weight(:,k) = weight(:,k)/sum(weight(:,k));
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEP III - ESTIMATE Q and R
  est.Q = zeros(nx);
  est.R = zeros(nz);
  for i = 1:nq
    for j = 1:nr
      est.Q = est.Q+weight(i+(j-1)*nq,end)*param.Qquant(:,:,i);
      est.R = est.R+weight(i+(j-1)*nq,end)*param.Rquant(:,:,j);
    end
  end
end

