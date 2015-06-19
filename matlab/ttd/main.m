td = small_test( );
    
HAS_EXTERNAL_FORCE_SCENARIO =0;
DEBUG=0;
VOLUME_BUDGET=1;
SHOW_DISPLACEMENT=0;
td.fillDataMatrix()

weight_per_unit =0.1;
    
    
MAX_FORCE_LENGTH=.5;     % NORMALIZE FORCES: The biggest force will have norm "1"

%PLOT SETTINGS
PLOT_ALL_BARS=0         % Show all bars
PLOT_FORCES=1           % Show forces  
PLOT_SHOW_GRID_POINTS=1 % Show grid points

 
InitialForces = td.Forces;
InitialForcesIdx = td.Nodes_under_load;

[q,v,w_star,t]=recompute_weights (td,DEBUG,VOLUME_BUDGET)

draw_result
    
     
    
  