clc
clear all
s = RandStream('mt19937ar','Seed',1)
RandStream.setGlobalStream(s);
m=10000;
n=500;
A=sprand(m,n,0.1);
b=rand(m,1);


format long
lambda=.1
prefix ='../../data/unittests/';
vals=full(A(:));
storeDataIntoFile([prefix 'vectorB.txt'],b,false)
storeDataIntoFile([prefix 'matrixA_values.txt'],vals(vals~=0),false)
rowIds=mod(find(A)-1,m);
storeDataIntoFile([prefix 'matrixA_rowIdx.txt'],rowIds,true)
%%
c = full(sum(spones(A)));
c=[0; c'];
for i=2:length(c)
  c(i)=c(i)+c(i-1);
end
 storeDataIntoFile([prefix 'matrixA_colPtr.txt'],c,true)
 
cvx_begin
  variable x(n)
  minimize( lambda*norm( x, 1 )+0.5*(A*x-b)'*(A*x-b) )

cvx_end  
cvx_optval 

cvx_begin
  variable x(n)
  minimize( 0.5*lambda*x'*x+0.5*(A*x-b)'*(A*x-b) )

cvx_end  
cvx_optval 
