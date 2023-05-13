%% Main Class File for applying AGVI
%% Authors: Bhargob Deka and James-A. Goulet, 2023
classdef multiagvi
    methods(Static)
        %% convert from cholesky space(L) to real space (P)
        function [E_P,P_P,Cov_L_P] = LtoP(n,mL_mat,SL_mat,SL,ind_covij)
            total   = n*(n+1)/2;
            m       = logical(triu(ones(n)));
            E_P     = zeros(total,1);
            P_P     = zeros(total);
            Cov_L_P = zeros(total);
            M       = mL_mat(m);
            [ind_i, ind_j] = ind2sub(size(mL_mat), find(logical(mL_mat)));
            for l = 1:size(ind_i,1)
                covxy = zeros(n,1);
                %     if l == 4
                %         stop = 1;
                %     end
                i   = ind_i(l);
                j   = ind_j(l);
                ind = cell2mat(ind_covij(l));
                covxy(1:size(ind,1)) = diag(SL(ind(:,1),ind(:,2)));
                [mv2_, Sv2_] = multiagvi.xy(mL_mat(:,i), mL_mat(:,j), SL_mat(:,i), SL_mat(:,j), covxy);
                E_P(l)   = sum(mv2_);
                P_P(l,l) = sum(Sv2_);
                C = zeros(total,size(ind,1));
                for k = 1:size(ind,1)
                    C(:,k)  = multiagvi.Cxyz(M(ind(k,1)), M(ind(k,2)), SL(:,ind(k,1)), SL(:,ind(k,2)));
                end
                Cov_L_P(:,l) = sum(C,2);
            end
        end
        %% Eigen value reconstruction to form PSD matrix
        function Q = makePSD(Q_W)
            [V,E] = eig(Q_W);
            E     = diag(E);
            v     = logical(E<0);
            m     = sum(v);
            if m > 0
                S = sum(v.*E);
                W = (S.^2.*100)+1;
                P = 0.01; % user-defined smallest positive eigen value
                i = find(E<0);
                C1    = E(i);
                E(i) = (P.*(S-C1).*(S-C1)./W)+1e-06;
            end
            Q = V*diag(E)*V';
        end
        function X_F = makePSDV2(B)
            [Q,d]   = eig(B,'vector'); 
            X_F     = Q*(max(d,0.1).*Q'); 
            X_F     = (X_F + X_F')/2;
        end
        %% Low-rank approximation of covariance matrix
        function Sigma_low_rank = lowRankApprox(Sigma, explained_variance)
            % This function computes the low-rank approximation of a covariance matrix
            % Input:
            % - Sigma: covariance matrix
            % - rank: integer rank for low-rank approximation
            % Output:
            % - Sigma_low_rank: low-rank approximation of covariance matrix

            % Compute the eigenvalue decomposition of the covariance matrix
            [V, D] = eig(Sigma);

            % Sort the eigenvalues in descending order
            [d, idx] = sort(diag(D), 'descend');
            
            % Compute the cumulative explained variance
            cumulative_variance = cumsum(d) / sum(d);
            
            % Determine the smallest rank that retains the desired percentage of variance
            rank = find(cumulative_variance >= explained_variance, 1);

            % Compute the low-rank approximation
            Sigma_low_rank = V(:, idx(1:rank)) * D(idx(1:rank), idx(1:rank)) * V(:, idx(1:rank))';

        end
        %% Low-rank and sparse matrix of a covariance matrix
        function [kron_matrix] = low_rank_kron(cov_matrix)
            % This function computes a low rank approximation of a covariance matrix using SVD,
            % and then computes a sparse matrix using inverse Kronecker factorization.
            % Input arguments:
            % cov_matrix: The input covariance matrix
            % rank: The rank of the low rank approximation
            % sparsity: The desired sparsity of the output sparse matrix
            % Output arguments:
            % sparse_matrix: The sparse matrix obtained using inverse Kronecker factorization

            % Compute the eigenvalue decomposition of the covariance matrix
            [V, D] = eig(cov_matrix);

            % Sort the eigenvalues in descending order
            [d, idx] = sort(diag(D), 'descend');
            
            % Compute the cumulative explained variance
            cumulative_variance = cumsum(d) / sum(d);
            
            % explained variance -- user defined
            explained_variance = 0.99;

            % Determine the smallest rank that retains the desired percentage of variance
            r = find(cumulative_variance >= explained_variance, 1);

            % Compute the low-rank approximation
            low_rank_matrix = V(:, idx(1:r)) * D(idx(1:r), idx(1:r)) * V(:, idx(1:r))';
%             clear r
            % Kronecker factorization
            [U,D,V] = svd(low_rank_matrix);
            k       = rank(U);
            Ur      = U(:,1:r);
            Dr      = D(1:r,1:r);
            Vr      = V(:,1:r);
            A       = Ur * sqrt(Dr);
            B       = sqrt(Dr) * Vr';
            % Compute the Kronecker product
            kron_matrix = kron(A, B);
            
        end
        
        % Sparse approximation using threshold approach
        function [C_sp] = sparse_approx(C, threshold)
            % SPARSE_APPROX - Obtain sparse approximation of covariance matrix using thresholding approach
            %
            % Syntax:  [C_sp] = sparse_approx(C, threshold)
            %
            % Inputs:
            %    C - full covariance matrix
            %    threshold - threshold value
            %
            % Outputs:
            %    C_sp - sparse approximation of covariance matrix
            %
            
            % Compute absolute values of entries in C
            abs_C = abs(C);

            % Apply threshold to absolute values
            abs_C(abs_C < threshold) = 0;

            % Set diagonal entries of C_sp to the diagonal entries of C
            C_sp = diag(diag(C));

            % Set off-diagonal entries of C_sp to the thresholded absolute values
            C_sp(~eye(size(C))) = abs_C(~eye(size(C)));

        end


        %% Converts structure from [d1, d2, d3, ...., c1, c2,c3,...] to [d1, c1, d2,.....]
        function[mL_new, SL_new] = convertstructure(mL,SL,n) % mL_old: my definition
            ind = multiagvi.indices_convert(n);
%             ind = [1 6 2 7 10 3 8 11 13 4 9 12 14 15 5];
%             ind = [1 4 2 5 6 3];
            mL_new = mL(ind);
            SL_new = SL(ind,:);
            SL_new = SL_new(:,ind);
%             SL_new = (SL_new+SL_new')/2;
        end
        function [E_P_old,P_P_old,Cov_L_P_old]  = convertback(E_P,P_P,Cov_L_P,n)
            ind = multiagvi.indices_convertback(n);
%             ind = [1 3 6 2 4 5];
            E_P_old     = E_P(ind);
            P_P_old     = P_P(ind,:); % do not swap rows and columns together
            P_P_old     = P_P_old(:,ind);
%             P_P_old     = (P_P_old+P_P_old')/2;
            if ~isempty(Cov_L_P)
                Cov_L_P_old = Cov_L_P(ind,:);
                Cov_L_P_old = Cov_L_P_old(:,ind);
            end
        end
        %% Indices for converting one system to another
        function index = indices_convert(n)
            m = triu(ones(n));
            total = n*(n+1)*0.5;
            index_diag = sub2ind(size(m),[1:1:n],[1:1:n]);
            values = 1:1:total;
            values_var = values(1:1:n);
            values(1:1:n)=[];
            index_cov  = cell(1,n-1);
            values_cov = cell(1,n-1);
            for i  = 1:n-1
                index_cov{i}  = sub2ind(size(m),[i*ones(1,n-i)],[i+1:1:n]);
                values_cov{i} = values(1:size(index_cov{1,i},2));
                values(1:1:size(index_cov{1,i},2))=[];
                m(index_cov{1,i})=values_cov{1,i};
            end
            m(index_diag)=values_var;
            ind=triu(true(size(m)));
            index = m(ind)';
        end
        function index = indices_convertback(n)
            total = n*(n+1)*0.5;
            m = logical(triu(ones(n)));
            A = zeros(n);
            A(m)=[1:1:total];
            A = A.';
            diag_A = diag(A);
            m1 = tril(true(size(A)),-1);
            index=[diag_A;A(m1)]';
        end
        %% Create indices for the covariance terms
        function ind_covij = index_covij(n)
            total          = n*(n+1)/2;
            m              = logical(triu(randn(n)));
            oldind         = find(m);
            newind         = find(find(m));
            newlist        = [oldind newind];
            newlist(1,:)   = [];
            [ind_i, ind_j] = ind2sub(size(m), find(m));
            ind_covij = cell(total,1);
            k = 1;
            while k <= size(ind_i,1)
                ind_covij{k,1} = zeros(ind_i(k),1);
                k = k+1;
            end
            for l = 1:size(ind_i,1)
                i   = ind_i(l);
                j   = ind_j(l);
%                 if i == 3 && j == 4
%                     stop=1;
%                 end
                % build indices for cov_ij
                r   = find(m(:,i).*m(:,j))';
                len = length(r);
                if i == 1
                    R = [i repmat(r,[1, len])];

                else
                    R = repmat(r,[1, 2]);%!! %len
                end
                if i == j
                    if i == 1
                        c = repmat(i,[1,len]);%len
                        C = repmat(c,[1,len]);
                        C = [i C];
                    else
                        c = repmat(i,[1,len]);%len
                        C = repmat(c,[1,2]);
                    end

                else
                    if i == 1
                        vec = [i j];
                        C = repmat(vec,[1,len]);
                    else
                        C   = [i*ones(1,len) j*ones(1,len)];
%                         vec = [i j];
%                         vec = repmat(vec,[1,len]);
%                         vec = reshape(vec,[len,2]);%len
%                         C   = reshape(vec',[1,2*len]);
                    end

                end
                ind = sub2ind(size(m),R,C);
                if i ~= 1
                    ind = reshape(ind,[len, 2]);%BD len->2
                end
                index_r = find(ismember(ind,newlist(:,1)));
                if ~isempty(index_r)
                    pos=arrayfun(@(x) find(newlist==x,1),ind(index_r));
                    ind(index_r) = newlist(pos,2);
                end
                ind_covij{l,1} = ind;

            end
        end
        function [EX_pos,PX_pos,NIS] = KFPredict(A,C,Q,R,y,P,Ep)
            Sx   = A*P*A';
            Sx   = (Sx+Sx')/2;
            Sx   = Sx + Q;
            Sp   = [Sx Q;Q Q];
            Sp   = (Sp+Sp')/2;
%             eig_Sp(:,t) = eig(Q_W);
            % Observation matrix
%             C   = [eye(n_x) zeros(n_x)];
            SY  = C*Sp*C' + R;
            SYX = Sp*C';
            my  = C*Ep;
            K   = SYX/(SY + 1e-08);
            e   = y-my; %YT(:,t)
            NIS = e' * pinv(SY) * e;
            %% Ist Update step:
            if any(isnan(y))
                EX_pos = Ep;
                PX_pos = Sp;
            else
                EX_pos  = Ep+K*e;
                PX_pos  = Sp-K*SYX';
            end
            
        end
        function [ES,PS] = agviSmoother(E_wp,s_wsqhat,PX_wp,EX_wpy,PX_wpy,E_P,P_P)
            E_Wp_prior    = E_wp;
            C_Wp_W2hat    = s_wsqhat;
            P_Wp_prior    = PX_wp;
            E_Wp_pos      = EX_wpy;
            P_Wp_pos      = PX_wpy;
            J             = C_Wp_W2hat/(P_Wp_prior+1e-08);

            ES  = E_P   +  J*(E_Wp_pos' - E_Wp_prior);
            PS  = P_P   +  J*(P_Wp_pos - P_Wp_prior)*J';
        end
        function [EL_pos,PL_pos] = PtoL(Cov_L_P,E_P,P_P,EL_pr,PL_pr,E_Pw_y,P_Pw_y)
            Jc = Cov_L_P/(P_P+1e-08);
            EL_pos   = EL_pr + Jc*(E_Pw_y - E_P);
            PL_pos   = PL_pr + Jc*(P_Pw_y - P_P)*Jc';
        end
        function ind_wijkl = indcov_wijkl(n_x)
            n_w2             = n_x*(n_x+1)/2;
            cov_wijkl        = cell(1,n_w2-1);
            cov_prior_wijkl  = cell(1,n_w2-1);
            ind_wijkl        = cell(1,n_w2-1);
            ind_cov          = cell(1,n_w2-1);
            ind_cov_prior    = cell(1,n_w2-1);
            i = 1; s = n_w2;
            while i <= n_w2-1
                cov_wijkl{i}       = zeros(1,s-1);
                cov_prior_wijkl{i} = zeros(1,s-1);
                ind_wijkl{i}       = repmat(zeros(1,s-1)',1,4);
                ind_cov{i}         = repmat(zeros(1,s-1)',1,4);
                ind_cov_prior{i}   = repmat(zeros(1,s-1)',1,4);
                i = i+1;
                s = s-1;
            end
            v   = 1:1:n_x;
            n_w = n_x;
            I1  = [[[1:n_w]' [1:n_w]']; nchoosek(v,2)];
            I2  = I1;
            I1(end,:) = [];
            for i = 1:n_w2-1
                ind_wijkl{i}(:,1:2) = repmat(I1(i,:),size(ind_wijkl{i},1),1);
            end
            i = 1; j = 2;
            while i <= n_w2-1
                ind_wijkl{i}(:,3:4) = I2(j:end,:);
                j = j+1;
                i = i+1;
            end
        end
        function  ind_mu = ind_mean(n_x,ind_wijkl)
            n_w2   = n_x*(n_x+1)/2;
            ind_mu = ind_wijkl;
            i = 1;
            while i <= n_w2-1
                ind_mu{i} = [ind_mu{i}(:,1) ind_mu{i}(:,4) ind_mu{i}(:,2) ind_mu{i}(:,3)];
                i = i+1;
            end
        end
        function ind_cov = ind_covariance(n_w2,n_w,ind_wijkl)
            i = 1;
            ind_cov  = cell(1,n_w2-1);
            while i <= n_w2-1
                m_ijkl = ind_wijkl{i};  % matrix of all ijkl
                r      = size(m_ijkl,1); % no of rows
                for j = 1:r
                    ijkl = m_ijkl(j,:);  % vector ijkl for row j
                    ind_cov{i}(j,1) = n_w*(ijkl(3)-1)+ijkl(1);
                    ind_cov{i}(j,2) = n_w*(ijkl(4)-1)+ijkl(2);
                    ind_cov{i}(j,3) = n_w*(ijkl(3)-1)+ijkl(2);
                    ind_cov{i}(j,4) = n_w*(ijkl(4)-1)+ijkl(1);
                end
                i = i+1;
            end
        end
        function [m_wii_y,s_wii_y,m_wiwj_y, s_wiwj_y] = meanvar_w2pos(EX_wy,PX_wy,cwiwjy, P_wy, n_w2hat,n_x)
            m_wii_y       = EX_wy.^2+diag(PX_wy);
            s_wii_y       = 2.*diag(PX_wy).^2+4.*diag(PX_wy).*EX_wy.^2;
            m_wiwj_y      = zeros(1,n_w2hat-n_x);
            s_wiwj_y      = zeros(1,n_w2hat-n_x);
            i = 1; j = 1; k = 1;
            while i <= n_x-1
                m_wiwj_y(k) = EX_wy(i)*EX_wy(j+1) + cwiwjy(k);
                s_wiwj_y(k) = P_wy(i)*P_wy(j+1) + cwiwjy(k)^2 + 2*cwiwjy(k)*EX_wy(i)*EX_wy(j+1) + P_wy(i)*EX_wy(j+1)^2 + P_wy(j+1)*EX_wy(i)^2;
                j = j+1;
                k = k+1;
                if j == n_x
                    i = i+1;
                    j = i;
                end
            end
        end
        function cov_wijkl = cov_w2pos(PX_wy,EX_wy,n_w2,ind_cov,ind_mu)
            cov_wijkl = cell(1,n_w2-1);
            i = 1; s = n_w2;
            while i <= n_w2-1
                cov_wijkl{i} = zeros(1,s-1);
                i = i+1;
                s = s-1;
            end
            i = 1;
            while i <= n_w2-1
                ind_C = ind_cov{i};
                ind_M = ind_mu{i};
                for j = 1:size(ind_C,1)
                    cov_wijkl{i}(j) = multiagvi.cov1234(ind_C(j,:),ind_M(j,:),PX_wy,EX_wy);
                end
                n1 = (n_w2 - 1) - size(cov_wijkl{i},2);
                if n1 > 0
                    add_zeros = zeros(1,n1);
                else
                    add_zeros = [];
                end
                cov_wijkl{i} = [add_zeros cov_wijkl{i}];
                i = i+1;
            end
        end
        function PX_wpy  = var_wpy(cov_wijkl,s_wii_y,s_wiwj_y)
            cov_wpy              = cell2mat(reshape(cov_wijkl,size(cov_wijkl,2),1));
            PX_wpy               = diag([s_wii_y' s_wiwj_y]);
            s_wpy                = zeros(size(PX_wpy,1));
            s_wpy(1:end-1,2:end) = cov_wpy;
            PX_wpy               = PX_wpy + s_wpy;
            PX_wpy               = triu(PX_wpy)+triu(PX_wpy,1)'; % adding the lower triangular matrix
        end
        function PX_wp = covwp(P_P,m_wsqhat,n_x,n_w2,n_w)
            v           = 1:1:n_x;
            s_wsqhat    = P_P;
            PX_wp       = zeros(size(s_wsqhat,1));
            ind12       = nchoosek(v,2);
            i = 1; j = 1;
            while i <= n_w2
                if i <= n_w
                    PX_wp(i,i)   = 2*m_wsqhat(i)^2 + 3*s_wsqhat(i,i);
                else
                    PX_wp(i,i)   = s_wsqhat(i,i) + (m_wsqhat(i)^2/(m_wsqhat(ind12(j,1))*m_wsqhat(ind12(j,2))+m_wsqhat(i)^2))*s_wsqhat(i,i)...
                        + m_wsqhat(ind12(j,1))*m_wsqhat(ind12(j,2)) + m_wsqhat(i)^2;
                    j = j+1;
                end
                i = i+1;
            end
        end
        function ind_cov_prior = indcov_priorWp(ind_wijkl,n_w2,n_x)
            ind_cov_prior = cell(1,n_w2-1);
            i = 1;
            while i <= n_w2-1
                m_ijkl = ind_wijkl{i};  % matrix of all ijkl
                r      = size(m_ijkl,1); % no of rows
                v      = 1:1:n_x;
                for j = 1:r
                    ijkl = m_ijkl(j,:);  % vector ijkl for row j
                    if ijkl(1)==ijkl(3)
                        ind_cov_prior{i}(j,1) = find(v==ijkl(1));
                    else
                        diff = abs(ijkl(3) - ijkl(1));
                        ind_cov_prior{i}(j,1) = diff + n_x;
                    end
                    if ijkl(2)==ijkl(4)
                        ind_cov_prior{i}(j,2) = find(v==ijkl(2));
                    else
                        diff = abs(ijkl(4) - ijkl(2));
                        ind_cov_prior{i}(j,2) = diff + n_x;
                    end
                    if ijkl(1)==ijkl(4)
                        ind_cov_prior{i}(j,3) = find(v==ijkl(1));
                    else
                        diff = abs(ijkl(4) - ijkl(1));
                        ind_cov_prior{i}(j,3) = diff + n_x;
                    end
                    if ijkl(2)==ijkl(3)
                        ind_cov_prior{i}(j,4) = find(v==ijkl(2));
                    else
                        diff = abs(ijkl(3) - ijkl(2));
                        ind_cov_prior{i}(j,4) = diff + n_x;
                    end

                end
                i = i+1;
            end
        end
        function cov_prior_wijkl = priorcov_wijkl(m_wsqhat,ind_cov_prior,n_w2)
            cov_prior_wijkl  = cell(1,n_w2-1);
            i = 1; s = n_w2;
            while i <= n_w2-1
                cov_prior_wijkl{i} = zeros(1,s-1);
                i = i+1;
                s = s-1;
            end
            i = 1;
            while i <= n_w2-1
                ind_Mpr = ind_cov_prior{i};
                for j = 1:size(ind_Mpr,1)
                    cov_prior_wijkl{i}(j) = m_wsqhat(ind_Mpr(j,1)).*m_wsqhat(ind_Mpr(j,2)) + m_wsqhat(ind_Mpr(j,3)).*m_wsqhat(ind_Mpr(j,4));
                end
                n1 = (n_w2 - 1) - size(cov_prior_wijkl{i},2);
                if n1 > 0
                    add_zeros = zeros(1,n1);
                else
                    add_zeros = [];
                end
                cov_prior_wijkl{i} = [add_zeros cov_prior_wijkl{i}];
                i = i+1;
            end
        end
        %% GMA operations for computing E[W^2hat], V[W^2hat], Cov(L,W^2hat)
        function [mxy, Sxy] = xy(mx, my, Sx, Sy, Cxy)
            mxy = mx.*my + Cxy;
            Sxy = Sx.*Sy + Cxy.^2 + 2 * Cxy .* mx .* my + mx.^2 .* Sy + my.^2 .* Sx;
        end
        function C_x_yz = Cxyz(my, mz, Cxy, Cxz)
            C_x_yz = Cxy .* mz + Cxz .* my;
        end
        function V = cov1234(ind_C,ind_M,PX_wy,EX_wy)
            V = 2*PX_wy(ind_C(1)).*PX_wy(ind_C(2)) + 2*PX_wy(ind_C(3)).*EX_wy(ind_M(1)).*EX_wy(ind_M(2)) + 2*PX_wy(ind_C(4)).*EX_wy(ind_M(3)).*EX_wy(ind_M(4));
        end
    end
end