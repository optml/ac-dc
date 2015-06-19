function td = small_test()
 
  
td =   TTDData();
td.initDataStructures(25,100);
 
gp = td.addGrid([2,2,0],0);
gp = td.addGrid([-2,2,0],0);
gp = td.addGrid([-2,-2,0],0);
gp = td.addGrid([2,-2,0],0);

gp = td.addGrid([2,2,1],1);
gp = td.addGrid([-2,2,1],1);
gp = td.addGrid([-2,-2,1],1);
gp = td.addGrid([2,-2,1],1);

gp = td.addGrid([1.5,1.5,2],1);
gp = td.addGrid([-1.5,1.5,2],1);
gp = td.addGrid([-1.5,-1.5,2],1);
gpV = td.addGrid([1.5,-1.5,2],1);

gp = td.addGrid([1.5,1.5,3],1);
gp = td.addGrid([-1.5,1.5,3],1);
gp = td.addGrid([-1.5,-1.5,3],1);
gp = td.addGrid([1.5,-1.5,3],1); 
 
gp = td.addGrid([0.5,0.5,4],1);
gp = td.addGrid([-0.5,0.5,4],1);
gp = td.addGrid([-0.5,-0.5,4],1);
gp = td.addGrid([0.5,-0.5,4],1); 

gp = td.addGrid([0.5,0.5,5],1);
gp = td.addGrid([-0.5,0.5,5],1);
gp = td.addGrid([-0.5,-0.5,5],1);
gp = td.addGrid([0.5,-0.5,5],1); 

gp = td.addGrid([0,0,6],1); 
 
for j=1:gp-1
  for i=j+1:min(j+13,gp)
     if (i>4 || j> 4)  
         td.addBar(i,j);   
     end
  end
end
 

k=-1;
 
td.setForce(gp,[0.,0.,k ]);  
 