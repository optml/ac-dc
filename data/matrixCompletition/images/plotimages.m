 load -ascii lenna.csv
 load -ascii lenna.out.csv
 load -ascii lenna.sample.csv
orig=lenna';
rec=lenna_out';

sampl = lenna_sample';
sampl(sampl==0)=1;


figure(1)
imshow(orig)
print -deps lenna_orog.eps
figure(2)
imshow(sampl)
print -deps lenna_sampl.eps
figure(3)
imshow(rec)
print -deps lenna_rec.eps
figure(5)
recc=rec;
recc(sampl~=1)=sampl(sampl~=1);
imshow(recc)
print -deps lenna_recc.eps
norm(recc-orig)
figure(4)

error=[];
rozdiel=abs(rec-orig);
error(:,:,1)=5*rozdiel;
error(:,:,3)=1-rozdiel;
error(:,:,2)=0*rozdiel;

%error(:,:,1)=rozdiel;
%error(:,:,2)=rozdiel;
%error=1-rozdiel;

imshow(error)
print -deps lenna_error.eps
