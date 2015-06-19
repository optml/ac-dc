function [f,g] = myfun( X,r,A,mask )
global A;
global r;
global mask; 

[m,n]=size(A);
 L=X(:,1:m)';
 R=X(:,m+1:end);
 DIFF=(L*R-A).*mask;
 
 f=norm(DIFF,2)^2;
 
 
if nargout > 1
 grad=zeros(size(X));
 
 
 for i=1:r
  grad(1+m*(i-1):i*m) = sum((diag(L(1+m*(i-1):i*m))*DIFF)');
 
  grad(m*r+1+n*(i-1):m*r+i*n) = sum((diag(R(1+n*(i-1):i*n))*DIFF')');
 end
 g=grad;
end
 
 
 
 
end

