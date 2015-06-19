



load  -ascii imagedata_big_red_out.csv
im_r=imagedata_big_red_out;


[Ur,Sr,Vr]=svd(im_r);



load -ascii imagedata_big_blue_out.csv
im_b=imagedata_big_blue_out;

[Ub,Sb,Vb]=svd(im_b);




load -ascii imagedata_big_green_out.csv
im_g=imagedata_big_green_out;
[Ug,Sg,Vg]=svd(im_g);



 

