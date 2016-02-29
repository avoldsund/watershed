mrstModule add co2lab; % for spill point analysis
mrstModule add libgeometry; % for mcomputeGeometry

%% Loading image
I = GEOTIFF_READ('anders_hoh.tiff'); 

% show the height of the landscape
%imshow(I.z, 'xdata', I.x, 'ydata', I.y); caxis([120, 1000]);
%title('Height of the area')
%axis on
%xlabel('Meters')
%ylabel('Meters')

% downsampling
ds_fac = 8; % valid is 1, 2 and 4  (to avoid fractions)

% computing lateral extent
X = abs(I.x(end) - I.x(1));
Y = abs(I.y(end) - I.y(1));

% I.info.map_info confirms that step length is uniform
xres = numel(I.x)/ds_fac - 1;
yres = numel(I.y)/ds_fac - 1;
G = cartGrid([xres, yres, 1], [X, Y, 1]);

% Setting correct z-coordinates and computing geometry
zvals = I.z(1:ds_fac:(end-1), 1:ds_fac:(end-1));
zvals = zvals(:);

G.nodes.coords(:,3) = [zvals; ones(size(zvals)) * max(zvals) + 1];
G = mcomputeGeometry(G);

% Want to shift the landscape of every lake and water down by some constant
% to detect more waters and lakes.

Gt = topSurfaceGrid(G);

% performing spill point analysis
tic; ts = trapAnalysis(Gt, false); toc

% show map
plotCellData(Gt, Gt.cells.H, 'edgecolor', 'none');
cmap = flipud(copper);
colormap(cmap);
set(gca, 'zdir', 'normal'); colorbar;

% Plot all trap cells (i.e. "lake cells")
hold on;
plotGrid(extractSubgrid(Gt, find(ts.traps~=0)));
