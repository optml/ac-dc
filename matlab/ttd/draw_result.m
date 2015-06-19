
if (~SHOW_DISPLACEMENT)
 hold off
 clf
end
hold on
view([1,1,1])

if(false && HAS_EXTERNAL_FORCE_SCENARIO)
       MULT = 3;
       quiver3(EXTERNAL_FORCE_POSSITION(1),EXTERNAL_FORCE_POSSITION(2),EXTERNAL_FORCE_POSSITION(3),...
          EXTERNAL_FORCE_DIRECTION(1)*MULT,EXTERNAL_FORCE_DIRECTION(2)*MULT,EXTERNAL_FORCE_DIRECTION(3)*MULT,...
          'm','LineWidth',3);
end 

%%


if (PLOT_SHOW_GRID_POINTS) 
    for i=1:td.GridPoints
       plot3( td.GRID(i,1), td.GRID(i,2),td.GRID (i,3),'k<','MarkerSize',10)
    end
    
    
end

xlabel('x')
ylabel('y')
zlabel('z')

 
 
 
%axis([-a a -b b c*sin(zlow) c*sin(zhigh)])
 
 
if (PLOT_FORCES)
    % Draw Forces
    FW = td.Forces;
    FORCE = [FW(1:td.TotalGridPoints) FW(td.TotalGridPoints+1: td.TotalGridPoints*2)  FW(td.TotalGridPoints*2+1:end)];
    FW = (FORCE(:,1).^2+FORCE(:,2).^2+FORCE(:,3).^2).^0.5;
    MAX_FORCE = max(FW);
    FORCE_SCALE=MAX_FORCE_LENGTH/MAX_FORCE;
    FW = 5 *FW/max(FW);
    %FW(FW>0 & FW<5)=5;
    for gp=1:td.GridPoints
        if (td.Nodes_under_load(gp)  )
           quiver3( td.GRID(gp,1), td.GRID(gp,2),td.GRID (gp,3),...
            FORCE(gp,1)*FORCE_SCALE,   FORCE(gp,2)*FORCE_SCALE,   FORCE(gp,3)*FORCE_SCALE,   ...
            'g','LineWidth',1+FW(gp)/3);
        end
    end

end

if (PLOT_ALL_BARS)
    for i=1:td.Bars_count
        from = td.Bars_id(i,1);
        to = td.Bars_id(i,2);
        plot3( [td.GRID(from,1),td.GRID(to,1)] ,...
               [td.GRID(from,2),td.GRID(to,2)] ,...
               [td.GRID(from,3),td.GRID(to,3)] ,...
               'k-','LineWidth',1)
    end
end

 

% DISPLAY ALL COMPUTED BARS

plot_truss