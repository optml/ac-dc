
load  -ascii lenna.out.csv
load  -ascii lenna.sample.csv
load  -ascii lenna.csv
im=lenna_out;

 image(:,:)=im;
 
figure(1) 
imshow(image')

load  -ascii lenna.sample.csv
im=lenna_sample;


image(:,:)=im;
 
figure(2) 
imshow(image')


image(:,:)=lenna;
figure(3) 
imshow(image')


image(:,:)=abs(lenna-lenna_out);
figure(4) 
imshow(image')



 % imwrite (image,"reconstruction.jpg")

