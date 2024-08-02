function [ W ] = OSLIM( Y, beta, lambda, variance, tolX, maxiter)
%oSLIM: optimal SLIM algorithm for top-N recommendations.
%   Convergence is reached when the max change in W between two consecutives
%   iterations is less than tolX or the number of iterations is maxiter.
%   input:
%          Y: the binary n x m matrix.
%          beta: L2- regularization parameter.
%          lambda: L1- regularization parameter.
%          variance: variance of the initial rand distribution.
%          tolX: the max change tolerable in W (convergence).
%          maxiter: maximun number of iterations.
%   output:
%          W: sparse coefficient matrix W.
    
    [~, m] = size(Y);

    % initialization 
     W = rand(m, m)*sqrt(variance);     
     gamma = 1e4; % for diag(W) = 0.
     I = eye(m); % identity matrix
     W0 = W;
    % numerator
     numer = Y'*Y;         
    
     sqrteps = sqrt(eps);


     for iter = 1:maxiter

            denominator = (numer + beta)*W  + lambda + gamma*I +  eps(numer);

            W = max(0, W.* (numer./denominator)); 
            
            % get the max change in W 
            dw = max(max(abs(W-W0) / (sqrteps+max(max(abs(W0))))));
            W0 = W;

            if dw <= tolX                     
                fprintf('\n iter: %d, dw = %e \n',iter, dw);
                 break;
             end

             fprintf('\n iter = %d \n', iter);

     end             

end

