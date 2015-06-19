

[img, map, alpha] = imread ('source_big.jpg');
%db=im2double(img); 
db = double(img);
db=db/255;
maxrank=150;

for i=1:3
  data = db(:,:,i);
  [U, S, V] = svd (data);
   S(maxrank+1:end,maxrank+1:end)=0;



          data = U*S*V';
db(:,:,i)=data;



end

csvwrite ('imagedata_big_red.csv', db(:,:,1));
csvwrite ('imagedata_big_blue.csv', db(:,:,2));
csvwrite ('imagedata_big_green.csv', db(:,:,3));



[img2, map2, alpha2] = imread ('mask.tiff');
db2=img2/255;
mask = db2(:,:,1);
mask = rand(size(mask));
mask = (mask>0.2);
csvwrite ('mask.csv', mask);

[m,n]=size(mask);
for j=1:m
for i=1:n
  if (mask(j,i)) 
       
	img(j,i,1)=255;
	 
        img(j,i,2)=255;
	img(j,i,3)=255;
        

        
  end
end
end
imwrite(img, 'masked.tiff')
 








