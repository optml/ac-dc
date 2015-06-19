function  storeDataIntoFile( fileName, data, isInt )
  fileID = fopen(fileName,'w'); 
  if (isInt)
    fprintf(fileID, '%d\n',data);
  else
    fprintf(fileID, '%1.16f\n',data);  
  end
  fclose(fileID);


end

