mrstModule add co2lab; % for spill point analysis
mrstModule add libgeometry; % for mcomputeGeometry
addpath('../Inpaint_nans/')

% Set downsampling factor and import data
ds_fac = 32;
[I_landscape, G_landscape, zvals_landscape] = add_tiff_data('anders_hoh.tiff', ds_fac);
[I_lakes, G_lakes, zvals_lakes] = add_tiff_data('anders_innsjo.tiff', ds_fac);

% Do some processing to make sure that the matrices are correct

% Make sure that anders_innsjo is a boolean matrix
zvals_lakes = helpFunctions.make_matrix_boolean(zvals_lakes);
% Interpolate all heights which are clearly wrong
zvals_landscape = helpFunctions.interpolate_extreme_values(0, 2469, zvals_landscape);

% Lower terrain for all lakes and make sure the heights are positive 
zvals_landscape = zvals_landscape - double(zvals_lakes) * 20;
zvals_landscape = max(0, zvals_landscape);

% Make the input for the trapAnalysis function
zvals_landscape = zvals_landscape(:);
G_landscape.nodes.coords(:,3) = [zvals_landscape; ones(size(zvals_landscape)) * max(zvals_landscape) + 1];
G_landscape = mcomputeGeometry(G_landscape);
Gt = topSurfaceGrid(G_landscape);

% performing spill point analysis
tic; ts = trapAnalysis(Gt, false); toc

plot.plot_trap_analysis(Gt, ts);