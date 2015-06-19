function [q,v,w_star,t]=recompute_weights(td,DEBUG,VOLUME_BADGET)
v=[];
A=td.DATA;
[n m]=size(A);
disp('lp solver')
%options = optimset('disp','iter');
%options = optimset('MaxIter',200);
d=td.Forces;
if (~DEBUG)
  %[x,opt_LP,exitflag,output] = linprog(ones(2*m,1),[],[],[A -A],d,zeros(2*m,1),[],[],options);
 
  
  
  n=td.Bars_count;
  cvx_begin
   cvx_precision low
   variable q(n)
   dual variables v
   minimize(norm(q,1))
   subject to
      A * q == d : v
  cvx_end
  
  
  
else
  q=zeros( m,1);  
end  
w_star = sum(abs(q));
t=VOLUME_BADGET/w_star*abs(q);

