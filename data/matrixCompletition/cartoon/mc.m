clc
if (true)
load  -ascii imagedata_big_red.csv
im_r=imagedata_big_red;
load -ascii imagedata_big_blue.csv
im_b=imagedata_big_blue;
load -ascii imagedata_big_green.csv
im_g=imagedata_big_green;
global mask;
load -ascii mask.csv
 mask(mask<1)=0;
end

 
%% 
global A;
global r;


%imagedata_big_green=imagedata_big_green(1:100,1:100);
%mask=mask(1:100,1:100);
options = optimset('GradObj','on')
 
x0 = [1,1];
r=5;
A=imagedata_big_red; 
[m,n]=size(A);
 
 
clock
 
[x,fval,exitflag,output,grad] = fminunc(@(x)(myfun(x)),ones(r,m+n)/((m+n)*r));
  L=x(:,1:m)';
 R=x(:,m+1:end);
 DIFF=(L*R-A).*mask;
res=L*R;
clock

imshow(res)
