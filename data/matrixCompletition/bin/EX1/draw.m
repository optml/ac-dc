

 

Sc=Sr;
Sc(rank+1:end,rank+1:end)=0*Sc(rank+1:end,rank+1:end);
im_r = Ur*Sc*Vr';

Sc=Sb;
Sc(rank+1:end,rank+1:end)=0*S(rank+1:end,rank+1:end);
im_b = Ub*Sc*Vb';

Sc=Sg;
Sc(rank+1:end,rank+1:end)=0*S(rank+1:end,rank+1:end);
im_g = Ug*Sc*Vg';



image(:,:,1)=im_r;
image(:,:,2)=im_b;
image(:,:,3)=im_g;
 
 
imshow(image)


