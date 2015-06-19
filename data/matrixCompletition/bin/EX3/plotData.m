
load -ascii "imagedata_big_red_out.csv"
im_r=imagedata_big_red_out;

load -ascii "imagedata_big_blue_out.csv"
im_b=imagedata_big_blue_out;

load -ascii "imagedata_big_green_out.csv"
im_g=imagedata_big_green_out;


image(:,:,1)=im_r;
image(:,:,2)=im_b;
image(:,:,3)=im_g;
 
figure(1)
imshow(image)


imwrite (image,"reconstruction.jpg")



load -ascii "imagedata_big_red_sample.csv"
im_r=imagedata_big_red_sample;

load -ascii "imagedata_big_blue_sample.csv"
im_b=imagedata_big_blue_sample;

load -ascii "imagedata_big_green_sample.csv"
im_g=imagedata_big_green_sample;


image(:,:,1)=im_r;
image(:,:,2)=im_b;
image(:,:,3)=im_g;
 
figure(2)
imshow(image)

imwrite (image,"sample.jpg")



