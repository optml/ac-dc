


weights = t./td.BARS_LENGTH;
v_max = max(abs(weights));

treshhold=0.001*v_max;
weights(abs(weights)<treshhold)=0;

weights = weights/v_max*10;
idx=find(weights);
for j=1:length(idx)
 % PLOT BARS
 i=idx(j);
   
 
    from = td.Bars_id(i,1);
    to = td.Bars_id(i,2);
 if(  weights(i)~=0)
    A=td.GRID(from,:);
    B=td.GRID(to,:);
    if SHOW_DISPLACEMENT
       A=A+displacements([from, from+td.TotalGridPoints,from+td.TotalGridPoints*2])';
       B=B+displacements([to, to+td.TotalGridPoints,to+td.TotalGridPoints*2])'; 
        
    end 
     if SHOW_DISPLACEMENT
         if(q(i)>0) %  
       plot3( [A(1),B(1)] ,...
              [A(2),B(2)] ,...
              [A(3),B(3)] ,...
              'b--','LineWidth',abs(  weights (i)))
       elseif(q(i)<0)       %  
       plot3( [A(1),B(1)] ,...
              [A(2),B(2)] ,...
              [A(3),B(3)] ,...
              'r--','LineWidth',abs(weights(i)))
         end 

  
     else
     
     if(q(i)>0) %  
   plot3( [A(1),B(1)] ,...
          [A(2),B(2)] ,...
          [A(3),B(3)] ,...
          'b-','LineWidth',abs(  weights (i)))
   elseif(q(i)<0)       %  
   plot3( [A(1),B(1)] ,...
          [A(2),B(2)] ,...
          [A(3),B(3)] ,...
          'r-','LineWidth',abs(weights(i)))
   end 
     end
 end
    
    
 
end
 