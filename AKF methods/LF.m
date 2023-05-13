function [innov]  =  LF(sys,xp,K,z)
% LF(sys,xp,K,z) linear filter with constant gain K
  [nz,N] = size(z);
  innov = zeros(nz,N);
  for i = 1:N
      H = sys.H;
      F = sys.F;
      innov(:,i) = z(:,i)-H*xp;
      xf = xp+K*innov(:,i);           % measurement update / filtering
    xp = F*xf;                 % time update / prediction
  end
end

