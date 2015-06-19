clc
clear all

 rng('default')

 addpath('../lib/gloptipoly3')
 
 
  
  
 
 
input = importfile('../../data/matrixCompletition/images/lenna.csv');
  input=input(1:10,1:10);
UU=rand;

[m,n]=size(input);
sparsity=0.3

mask = createMask(m,n,sparsity);
mask=sparse(mask);
maskT=mask';
rank=5;
iters=max(m,n)*rank;
iters=1000
mu=.0001;
 

MS=sparse(mask);
[rows,cols,vals] = find(MS);
for f=1:2
    figure(f)
    hold off
end

for type=5:-1:1
    rng('default')
    tic
[ output, U, V, log,obj ] = matrixCompletion( input, mask,maskT, rank,iters,mu,rows,cols,type);

min(obj(2:end)-obj(1:end-1))


colours=['b-';'r-';'k-';'g-';'m-'];

toc
figure(1)
 
semilogy(log,colours(type,:))
title('Error in reconstruction')

hold on
figure(2)
 
semilogy(obj,colours(type,:))
title('Objective value')
hold on
end
for f=1:2
    figure(f)
legend('type 5','type 4','type 3', 'type 2', 'type 1', 'Location','Best')
end


%%
input(1:5,1:5);
output(1:5,1:5);

figure(4)
imshow(input)
figure(3)
oo=output;
oo(oo>1)=1;
oo(oo<0)=0;
imshow(output)
diff=obj(1:end-1)-obj(2:end);
min(diff)

figure(5)
ii=input;
ii(mask~=1)=0;
imshow(ii)

