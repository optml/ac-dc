classdef TTDData  < handle
   properties
      GridPoints=0;
      TotalGridPoints;
      Bars_count=0;
      Bars_id=[];
      DATA=[];
      GRID=[];
      GRID_FREE=[];
      Forces=[];
      Nodes_under_load=[];
      Nuber_of_nodes=0;
      BARS_LENGTH=[];
      IS_OUTSIDE_GRID_POINT=[];
   end
   
   
   methods
      function TTD = TTDData()
          
      end
      function  addBar(TTD,from, to)
        TTD.Bars_count=TTD.Bars_count+1;
        TTD.Bars_id(TTD.Bars_count,1:2)=[from,to];
      end 
      
      function  addForce(TTD,gp,F)
         TTD.Forces([gp,gp+TTD.TotalGridPoints,gp+2*TTD.TotalGridPoints])= TTD.Forces([gp,gp+TTD.TotalGridPoints,gp+2*TTD.TotalGridPoints])+F';
         TTD.Nodes_under_load(gp)=1;
      end 
      
      function  setForce(TTD,gp,F)
         TTD.Forces([gp,gp+TTD.TotalGridPoints,gp+2*TTD.TotalGridPoints])=F;
         TTD.Nodes_under_load(gp)=1;
      end 
      
      
      function applyExternalForce(TTD, X,D)
          for gp=1:TTD.GridPoints
              if (TTD.GRID_FREE(gp) && TTD.IS_OUTSIDE_GRID_POINT(gp))
                 TTD.addForce(gp,D) 
              end
              
          end
      end
      
      
      function fillDataMatrix(TTD)
         ROW_ID = zeros( TTD.Bars_count*6,1);
         COL_ID = zeros( TTD.Bars_count*6,1); 
         VALS = zeros( TTD.Bars_count*6,1);
         TTD.BARS_LENGTH=zeros(TTD.Bars_count,1);
         used = 0;
         for i=1:TTD.Bars_count

           from = TTD.Bars_id(i,1);
           to = TTD.Bars_id(i,2);
           A = TTD.GRID(from,:);
           B = TTD.GRID(to,:);
           Displacement = B-A;
           
            
           
           TTD.BARS_LENGTH(i)=norm(Displacement);
           beta=Displacement/(TTD.BARS_LENGTH(i)^2); 
 
           
           if (TTD.GRID_FREE(from))
               used=used+1;
               ROW_ID(used,1) = from;
               COL_ID(used,1) = i;
               VALS(used,1) = -beta(1);               
               used=used+1;
               ROW_ID(used,1) = from+TTD.GridPoints;
               COL_ID(used,1) = i;
               VALS(used,1) = -beta(2);
               used=used+1;
               ROW_ID(used,1) = from+TTD.GridPoints*2;
               COL_ID(used,1) = i;
               VALS(used,1) = -beta(3);
           end
           if (TTD.GRID_FREE(to))
               used=used+1;
               ROW_ID(used,1) = to;
               COL_ID(used,1) = i;
               VALS(used,1) = beta(1);               
               used=used+1;
               ROW_ID(used,1) = to+TTD.GridPoints;
               COL_ID(used,1) = i;
               VALS(used,1) = beta(2);
               used=used+1;
               ROW_ID(used,1) = to+TTD.GridPoints*2;
               COL_ID(used,1) = i;
               VALS(used,1) = beta(3);
           end                       
         end  
         TTD.DATA=sparse(ROW_ID(1:used),COL_ID(1:used),VALS(1:used),TTD.GridPoints*3,TTD.Bars_count);
             
      end
      
      function node = addGrid(TTD,A,is_free)
        TTD.GridPoints =TTD.GridPoints +1;
        node=TTD.GridPoints;
        TTD.GRID(TTD.GridPoints,:)=A;
        TTD.GRID_FREE(TTD.GridPoints,1)=is_free;
      end       
      function node = addOutsideGrid(TTD,A,is_free)
        TTD.GridPoints =TTD.GridPoints +1;
        node=TTD.GridPoints;
        TTD.GRID(TTD.GridPoints,:)=A;
        TTD.GRID_FREE(TTD.GridPoints,1)=is_free;
        if (is_free)
          TTD.IS_OUTSIDE_GRID_POINT(TTD.GridPoints,1)=1;
        end
      end       
      
      
      function initDataStructures(TTD,m,buffer)
        TTD.GRID=zeros(m,3);  
        TTD.IS_OUTSIDE_GRID_POINT=zeros(m,1);
        TTD.Forces=zeros(3*m,1);
        TTD.Nodes_under_load=zeros(m,1);
        TTD.Bars_id=zeros(buffer,2);
        TTD.Nuber_of_nodes=m;
        TTD.TotalGridPoints = m;
      end
      
   end
end

