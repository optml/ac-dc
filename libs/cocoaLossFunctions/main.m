clear all
clc
addpath('../minFunc/minFunc')
addpath('../minFunc/autoDif')

nInst = 30;
nVars = 10;
A = randn(nInst,nVars);
w = randn(nVars,1);
y = sign(A*w + 0*randn(nInst,1));

  mAlpha = rand(nInst,1);
  
  lambda=1
totalN=nInst;
  mW = 1/(lambda*totalN)*A'*(y.*mAlpha);

 alphaInit = zeros(nInst,1);
 

initial = SquareLoss(alphaInit,A,y,mAlpha,mW,lambda,totalN )

 

options.Method = 'sd';
alphaInit =  minFunc(@SquareLoss,alphaInit,options,A,y,mAlpha,mW,lambda,totalN );
 
[final,fg] = SquareLoss(alphaInit,A,y,mAlpha,mW,lambda,totalN );
alphaInit=alphaInit*0;
options.optTol=0.1
options.Method = 'cg';
options.Method='csd'
options.Method='bb'
options.Method='lbfgs'
options.Method='newton0' 
options.Method='pnewton0'
options.Method='pcg'
options.Method='pnewton0'
options.Method='scg'
tic
alphaInit =  minFunc(@SquareLoss,alphaInit,options,A,y,mAlpha,mW,lambda,totalN );
toc
% 
%  alphaInit =  fminunc(@SquareLoss,alphaInit,options,A,y,mAlpha,mW,lambda,totalN );

[final2,fg2] = SquareLoss(alphaInit,A,y,mAlpha,mW,lambda,totalN );

final
final2
norm(fg)
norm(fg2)

return


funObj = @(w)LogisticLoss(w,X,y);

fprintf('\nRunning Steepest Descent\n');
options.Method = 'sd';
minFunc(@LogisticLoss,w_init,options,X,y);
pause;

fprintf('\nRunning Cyclic Steepest Descent\n');
options.Method = 'csd';
minFunc(@LogisticLoss,w_init,options,X,y);
pause;

fprintf('\nRunning Conjugate Gradient\n');
options.Method = 'cg';
minFunc(@LogisticLoss,w_init,options,X,y);
pause;
