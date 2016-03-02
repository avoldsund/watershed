mrstModule add co2lab; % for spill point analysis
mrstModule add libgeometry; % for mcomputeGeometry
addpath('../Inpaint_nans/')

%% Loading image
I = GEOTIFF_READ('anders_hoh.tiff'); 
%imshow(I.z, 'xdata', I.x, 'ydata', I.y); caxis([120, 1000]);

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

J = GEOTIFF_READ('anders_innsjo.tiff');
%K = GEOTIFF_READ('anders_elvbekk.tiff');

X_J = abs(J.x(end) - J.x(1));
Y_J = abs(J.y(end) - J.y(1));
xres_J = numel(J.x)/ds_fac - 1;
yres_J = numel(J.y)/ds_fac - 1;
G_J = cartGrid([xres_J, yres_J, 1], [X_J, Y_J, 1]);
zvals_J = J.z(1:ds_fac:(end-1), 1:ds_fac:(end-1));

zvals = zvals - double(zvals_J) * 20;
zvals = max(0, zvals);

%X_K = abs(K.x(end) - K.x(1));
%Y_K = abs(K.y(end) - K.y(1));
%xres_K = numel(K.x)/ds_fac - 1;
%yres_K = numel(K.y)/ds_fac - 1;
%G_K = cartGrid([xres_K, yres_K, 1], [X_K, Y_K, 1]);
%zvals_K = K.z(1:ds_fac:(end-1), 1:ds_fac:(end-1));
%zvals = zvals - double(zvals_J) * 20 - double(zvals_K) * 20;


% Interpolate all heights which are clearly wrong
if (max(max(zvals)) > 2469 || min(min(zvals)) < 0)
    zvals(zvals > 2469) = NaN;
    zvals(zvals < 0) = NaN;
    num_of_nans = sum(sum(isnan(zvals)));
    zvals = inpaint_nans(double(zvals));
end

zvals = zvals(:);

G.nodes.coords(:,3) = [zvals; ones(size(zvals)) * max(zvals) + 1];
G = mcomputeGeometry(G);
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
title('Plot of trap cells')
xlabel('Meters')
ylabel('Meters')
