function [nll,g,H,T] = SquareLoss(deltaAlpha,A,y,mAlpha,mW,lambda,totalN)
% w(feature,1)
% X(instance,feature)
% y(instance,1)

H=[];
T=[];

g=zeros(size(deltaAlpha));


nll=1/totalN* (0.5*sum(  (deltaAlpha+ mAlpha).^2    )  - sum((deltaAlpha).*y  )  )+...
      lambda/2 *   norm(  1/(lambda*totalN) * A'*(deltaAlpha)+mW    )^2;

%[n,p] = size(X);

%Xw = X*w;
%yXw = y.*Xw;

%nll = sum(mylogsumexp([zeros(n,1) -yXw]));

if nargout > 1
 
    
    
    g = 1/totalN * ( deltaAlpha +  mAlpha - y  ) + ...
           lambda   * (  1/(lambda*totalN)^2 *  (A*(A'*(deltaAlpha))) + 1/(lambda*totalN)*  (A*mW)  );
    
   
    
end

g=g*totalN;
nll=nll*totalN;

if nargout > 2
  %  H = X.'*diag(sparse(sig.*(1-sig)))*X;
end

if nargout > 3
   % T = zeros(p,p,p);
   % for j1 = 1:p
   %     for j2 = 1:p
   %         for j3 = 1:p
   %             T(j1,j2,j3) = sum(y(:).^3.*X(:,j1).*X(:,j2).*X(:,j3).*sig.*(1-sig).*(1-2*sig));
   %         end
   %     end
   % end
end
