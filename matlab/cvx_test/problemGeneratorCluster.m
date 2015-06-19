clc
clear all
s = RandStream('mt19937ar','Seed',1)
RandStream.setGlobalStream(s);
m=1000;
nLocal=125;
parts=4;
n=nLocal*parts;
ALocal=sprand(m,n,.1);
AGlobal=sprand(m,n,.1);
 
A=[];
for i=1:parts
   A=[ A  zeros(m*(i-1), nLocal)
       zeros(m, (i-1)*nLocal)   ALocal(:,1+(i-1)*nLocal:i*nLocal)] ;
end
A=[ AGlobal; A];
b=rand(m*(parts+1),1);


 

format long
lambda=.1
prefix ='../../data/unittests/distributed_';
for p=1:parts
    
    storeDataIntoFile([prefix sprintf('%d',p-1) '_global_vectorB.txt'],b(1:m),false)
    storeDataIntoFile([prefix sprintf('%d',p-1) '_local_vectorB.txt'],b( p*m+1 : (p+1)*m ),false)
    
    
    
    B=AGlobal(:,(p-1)*nLocal+1:p*nLocal);
    vals=full(B(:));
    storeDataIntoFile([prefix sprintf('%d',p-1) '_global_matrixA_values.txt'],vals(vals~=0),false)
    rowIds=mod(find(B)-1,m);
    storeDataIntoFile([prefix sprintf('%d',p-1) '_global_matrixA_rowIdx.txt'],rowIds,true)
    c = full(sum(spones(B)));
    c=[0; c'];
    for i=2:length(c)
      c(i)=c(i)+c(i-1);
    end
    storeDataIntoFile([prefix sprintf('%d',p-1) '_global_matrixA_colPtr.txt'],c,true)
    
    
    B=ALocal(:,(p-1)*nLocal+1:p*nLocal);
     vals=full(B(:));
    storeDataIntoFile([prefix sprintf('%d',p-1) '_local_matrixA_values.txt'],vals(vals~=0),false)
    rowIds=mod(find(B)-1,m);
    storeDataIntoFile([prefix sprintf('%d',p-1) '_local_matrixA_rowIdx.txt'],rowIds,true)
    c = full(sum(spones(B)));
    c=[0; c'];
    for i=2:length(c)
      c(i)=c(i)+c(i-1);
    end
    storeDataIntoFile([prefix sprintf('%d',p-1) '_local_matrixA_colPtr.txt'],c,true)    
    
    
end


 

 
 %%
 
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
0.5*lambda*x'*x+0.5*(A*x-b)'*(A*x-b)