
%mb=2
%nb=2
%rb=9

%    a*delta + b * delta^2 + c* epsilon+ d* epsilon^2 + h*(epsilon delta)^2 
%    + e delta^2 epsilon + f delta epsilon^2 + g epsilon * delta

R1 = U(mb,:)*V(:,nb)-input(mb,nb);

R2 = (U(mb,:)*V(:,:)-input(mb,:)).*mask(mb,:);
R2(1,nb)=0;


R3 = (U(:,:)*V(:,nb)-input(:,nb)).*mask(:,nb);
R3(mb)=0;

a=2*R1*V(rb,nb)  +  2 * sum(R2.*V(rb,:)) +mu ;
%b=V(rb,nb)^2 +                 V(rb,nb)^2;
b=sum(V(rb,mask(mb,:)).^2);
c=2*R1*U(mb,rb)  +  2*   sum(U(:,rb).*R3) +mu;
%d=U(mb,rb)^2  +               U(mb,rb)^2;
d=sum(U(mask(:,nb),rb).^2);
e=2*V(rb,nb);
f=2*U(mb,rb);
g=2*R1+2*U(mb,rb)*V(rb,nb);
h=0;
if (mask(mb,nb)==1)
   h=1;
end

%IN=0.5*sum(sum(((U*V-input).*mask).^2));

% We want to solve following equaiton of two variables

%0= a + 2 *b * delta + 2 *h*epsilon^2 delta + 2 e delta  epsilon + f epsilon^2 + g epsilon

%0=  c + 2* d* epsilon + 2 h* epsilon delta^2 + e delta^2  + 2 f delta epsilon + g  delta

  try
 mpol x1 x2
  g0 = a*x1 + b * x1^2 + c* x2+ d* x2^2 + h*(x1*x2  )^2 + e * x1^2 *x2 + f *x1* x2^2 + g *x1 * x2;
 
  P = msdp(min(g0));
 
 [status,obj] = msol(P) ;
 
 
 
 x = double([x1 x2])  ;
 
 

delta=x(1);
epsilon=x(2);


vall= a*delta + b * delta^2 + c* epsilon+ d* epsilon^2 + h*(epsilon *delta)^2 + e* delta^2 *epsilon + f *delta *epsilon^2 + g *epsilon * delta;
if(vall<=0)
 U(mb,rb)=U(mb,rb)+delta;
 V(rb,nb)=V(rb,nb)+epsilon;
end
  catch
     disp('Error')
  end

%AFT=0.5*sum(sum(((U*V-input).*mask).^2));


 

%AFT=0.5*sum(sum(((U*V-input).*mask).^2));

%ROZ=  a*delta + b * delta^2 + c* epsilon+ d* epsilon^2 + h*(epsilon *delta)^2    + ...
%  e * delta^2 *epsilon + f *delta *epsilon^2 + g * epsilon * delta;

%AFT-IN

%ROZ/2


 