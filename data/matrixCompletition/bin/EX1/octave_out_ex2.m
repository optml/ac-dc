






load -ascii "imagedata_ex2_19.csv"

imagedata_out = imagedata_ex2_19;

m = min(min(imagedata_out))
M = max(max(imagedata_out))

  
imagedata_out(imagedata_out<0)=0;
imagedata_out(imagedata_out>1)=1;

 
figure(1)
imshow(imagedata_out)

figure(2)
load -ascii "imagedata_sample.csv"
imshow(imagedata_sample)











