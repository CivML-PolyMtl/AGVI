function [est] = MDM_LTV(sys,z)
%%
% Measurement Difference Method by J. Dunik, O. Kost, O. Straka, E. Blasch
%%
F = sys.F;
H = sys.H;
nx = sys.nx;
nz = sys.nz;
N  = sys.N;
% NEGA METHOD
% - Noise variances Q and R estimation
% -- definition of design parameters
L = 2;
Lp = L + 1;
% -- number of unknown and estimated variances (Q and R are scalars)
nTheta = 2;
% -- computation of observability and auxilliary matrices in (12), (15)
O = zeros(L*nz,nx,N-1);
Gamma = zeros(L*nz,L*nx,N-1);
for k = 1:N-1
    O(:,:,k) = [H(k); H(k+1)*F(k)];
    Gamma(:,:,k) = [0 0; H(k+1) 0];
end
% -- computation of augmented measurement prediction error 
Z = [z(:,1:end-1); z(:,2:end)]; % augmented measurement (14)
Zhat = zeros(L*nz,N-1);   % augmented measurement prediction (16)
Ztilde = zeros(L*nz,N-1); % augmented measurement prediction error (18)
for k = 2:N-1    
    Zhat(:,k) = O(:,:,k)*F(k-1)*pinv(O(:,:,k-1))*Z(:,k-1);
    Ztilde(:,k) = Z(:,k) - Zhat(:,k);
end
% -- computation of matrices for LS solution (38)
% --- A(k) matrix in (20)
Aw = zeros(L*nz,Lp*nx,N-1); % (23)
Av = zeros(L*nz,Lp*nz,N-1); % (24)
Ak = zeros(L*nz,Lp*(nx+nz),N-1); % (22)
for k = 2:N-1
    Aw(:,:,k) = [eye(L*nz), eye(L*nz)] * [[O(:,:,k), Gamma(:,:,k)]; [-O(:,:,k)*F(k-1)*pinv(O(:,:,k-1))*Gamma(:,:,k-1), zeros(L*nz,nx)]];
    Av(:,:,k) = [eye(L*nz), eye(L*nz)] * [[zeros(L*nz,nx), eye(L*nz)]; [-O(:,:,k)*F(k-1)*pinv(O(:,:,k-1)), zeros(L*nz,nx)]];
    Ak(:,:,k) = [Aw(:,:,k), Av(:,:,k)];
end
% % --- SIGMA (26) - just for illustration, not necessary for NEGA
% %     method itself
% SIGMA = blkdiag(kron(eye(Lp),Q),kron(eye(Lp),R));
% SIGMAs = SIGMA(:);
% --- design of duplication matrix PSI in (29) and (32) (note that theta in (12) is [Q; R])
PSI = [1 0; zeros(6,2); 1 0; zeros(6,2); 1 0; zeros(6,2); 0 1; zeros(6,2); 0 1; zeros(6,2); 0 1];
% --- matrix LAMBDA(k) (29)
LAMBDAk = zeros((L*nz)^2,nTheta,N-1);
for k = 2:N-1
    LAMBDAk(:,:,k) = kron(Ak(:,:,k),Ak(:,:,k))*PSI;
end
% --- matrix LAMBDA in (34)
LAMBDA = [];
for k = 2:N-1
    LAMBDA = [LAMBDA; LAMBDAk(:,:,k)]; %#ok<AGROW>
end
% -- construction of sample based estimate of vector b in (36), (37)
bHatk = zeros((L*nz)^2,N-1);
bHat = [];
for k = 2:N-1
    bHatk(:,k) = kron(Ztilde(:,k),Ztilde(:,k));
    bHat = [bHat; bHatk(:,k)]; %#ok<AGROW>
end
% -- noise variances estimation by (38)
thetaHat = pinv(LAMBDA)*bHat;
Qhat     = thetaHat(1);
% Rhat     = thetaHat(2);
est.Qhat = Qhat;
% est.Rhat = Rhat;
% -- disp
% disp(['True noise variances are: Q = ', num2str(sys.Q), ', R = ', num2str(sys.R)])
% disp(['Estimated noise variances are: Qhat = ', num2str(Qhat), ', Rhat = ', num2str(Rhat)])
