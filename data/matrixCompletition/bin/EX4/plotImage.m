
load  -ascii imagedata_big_red_out.csv
im_r=imagedata_big_red_out;

load -ascii imagedata_big_blue_out.csv
im_b=imagedata_big_blue_out;

load -ascii imagedata_big_green_out.csv
im_g=imagedata_big_green_out;

im_r(im_r<0)=0;
im_r(im_r>1)=1;
im_b(im_b<0)=0;
im_b(im_b>1)=1;
im_g(im_g<0)=0;
im_g(im_g>1)=1;


image(:,:,1)=im_r;
image(:,:,2)=im_b;
image(:,:,3)=im_g;
 
 
imshow(image)


 % imwrite (image,"reconstruction.jpg")

