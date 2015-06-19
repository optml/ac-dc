clc
clear all

M = importdata('matica.txt');
A=sparse(M(:,1)+1, 1+M(:,2), M(:,3));
AA=A;
b = importdata('vec.txt');

xx =importdata('x.txt')
  xO =importdata('xo.txt')

ms = 2;
rho=1;
[m,n]=size(A);

yy=b;
%yy=b/norm(b);
%yy=yy/norm(yy);
D = A'*yy;
SS=sort(abs(D),'descend');
TR = SS(ms)

for i=1:n
    
    x(i)=0;
   if ( abs(D(i)) >= TR)
       
       alfa = 1/abs(D(i));
       
       x(i) = sign(D(i))*rho/sqrt(ms);
   elseif ( abs(D(i)) <= 0.1 )
       alfa = 1;
       
   else
       alfa =  1/abs(D(i));
       
   end
   A(:,i)=A(:,i)*alfa;    
    
end
x=x';
b=yy+A*x;


0.5*norm(yy)^2+norm(x,1)


cvx_begin
  variable z(n)
  minimize (0.5*(A*z-b)' *(A*z-b)+norm(z,1))
cvx_end
%%
(0.5*(A*z-b)' *(A*z-b)+norm(z,1))
(0.5*(A*x-b)' *(A*x-b)+norm(x,1))
(0.5*(A*xx-b)' *(A*xx-b)+norm(xx,1))
%%
V=[x,xO, z, D];
V( abs(x)>0.01 | abs(z) > 0.01 ,:)

return
%%
A*xx-b;
sum(A(:,1:3)')'
AA(:,3)
D(3)
find(xO)

