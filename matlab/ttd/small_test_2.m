function td = small_test_2()
 
  
td =   TTDData();
td.initDataStructures(17,100);
 
gp = td.addGrid([0,0,0],0);
gp = td.addGrid([0,0,1],1);
gp = td.addGrid([0,0,2],1);
gp = td.addGrid([1,0,2],1);
gp = td.addGrid([-1,0,2],1);
gp = td.addGrid([0,0,3],1);
gp = td.addGrid([0,0,4],1);
gp = td.addGrid([0,0,5],1);
gp = td.addGrid([1,0,5],1);
gp = td.addGrid([-1,0,5],1);
gp = td.addGrid([0,1,5],1);
gp = td.addGrid([0,-1,5],1);
gp = td.addGrid([0,0,6],1);


gp = td.addGrid([2,0,0],0);
gp = td.addGrid([-2,0,0],0);

gp = td.addGrid([2,0,3],1);
gp = td.addGrid([-2,0,3],1);



td.addBar(16,1);  
td.addBar(17,1); 

td.addBar(14,16);  
td.addBar(15,17); 

td.addBar(16,4);  
td.addBar(17,5); 

td.addBar(1,2);          
td.addBar(3,2);          
td.addBar(3,4);   
td.addBar(3,5); 
td.addBar(3,6); 
td.addBar(7,6); 
td.addBar(7,8); 

td.addBar(4,6); 
td.addBar(6,5); 


td.addBar(9,8); 
td.addBar(10,8); 
td.addBar(11,8); 
td.addBar(12,8); 

td.addBar(13,8); 
td.addBar(13,9); 
td.addBar(13,10); 
td.addBar(13,11); 
td.addBar(13,12); 


td.addBar(7,9); 
td.addBar(7,10); 
td.addBar(7,11); 
td.addBar(7,12); 


td.addBar(5,11); 
td.addBar(5,10); 
td.addBar(5,12); 

td.addBar(4,9); 
td.addBar(4,11); 
td.addBar(4,12); 

k=-1;
td.setForce(9,[0.0,0.0,k ]); 
td.setForce(10,[0.0,0.0, k ]);          
td.setForce(11,[0.0,0.0, k ]);      
td.setForce(12,[0.0,0.0, k ]);  
td.setForce(4,[0.0,0.0, k ]);  
 td.setForce(5,[0.0,0.0, k ]);  
 