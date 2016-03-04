function plot_trap_analysis(topGrid, trapStructure)
% PLOT_TRAP_ANALYSIS Plot a trap structure after doing trap analysis
% plot_trap_analysis takes a trap structure object trapStructure and
% creates a plot of this showing all traps

% show map
plotCellData(topGrid, topGrid.cells.H, 'edgecolor', 'none');
cmap = flipud(copper);
colormap(cmap);
colorbar;

%set(gca, 'zdir', 'normal'); 

% Plot all trap cells (i.e. "lake cells")
hold on;
plotGrid(extractSubgrid(topGrid, find(trapStructure.traps~=0)));
title('Plot of trap cells')
xlabel('Meters')
ylabel('Meters')

end