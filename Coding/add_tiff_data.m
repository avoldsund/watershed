function dataInfo, cartGrid = add_tiff_data(fileName, downSamplingFactor)
% ADD_TIFF_DATA 

% Loading image
I = GEOTIFF_READ(fileName);

% Computing lateral extent in meters
X = abs(I.x(end) - I.x(1));
Y = abs(I.y(end) - I.y(1));

% I.info.map_info confirms that step length is uniform
xres = numel(I.x)/ds_fac - 1;
yres = numel(I.y)/ds_fac - 1;
G = cartGrid([xres, yres, 1], [X, Y, 1]);

% Setting correct z-coordinates and computing geometry
zvals = I.z(1:ds_fac:(end-1), 1:ds_fac:(end-1));



end

