

[img, map, alpha] = imread ('source_big.jpg');
db=im2double(img); 

csvwrite ('imagedata_big_red.csv', db(:,:,1));
csvwrite ('imagedata_big_blue.csv', db(:,:,2));
csvwrite ('imagedata_big_greed.csv', db(:,:,3));










