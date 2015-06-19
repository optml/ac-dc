




load -ascii "imagedata_out.csv"

m = min(min(imagedata_out))
M = max(max(imagedata_out))

%imagedata_out=(imagedata_out-m)/(M-m);

#m = min(min(imagedata_out))
#M = max(max(imagedata_out))
 
imagedata_out(imagedata_out<0)=0;
imagedata_out(imagedata_out>1)=1;

#imagedata_out(imagedata_out<1)=0;
#imagedata_out(imagedata_out==2)=1;

figure(1)
imshow(imagedata_out)

figure(2)
load -ascii "imagedata_sample.csv"
imshow(imagedata_sample)











