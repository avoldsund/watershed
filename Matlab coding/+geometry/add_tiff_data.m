function [info, grid, zvals] = add_tiff_data(fileName, downSamplingFactor)
% ADD_TIFF_DATA Returns info object, a cartesian grid and zvals for an area
% add_tiff(fileName, downSamplingFactor) takes as input the filename of a
% tiff data file and a downsampling factor.

% Loading image
info = geometry.GEOTIFF_READ(fileName);

% Computing lateral extent in meters
X = abs(info.x(end) - info.x(1));
Y = abs(info.y(end) - info.y(1));

% I.info.map_info confirms that step length is uniform
xres = numel(info.x)/downSamplingFactor - 1;
yres = numel(info.y)/downSamplingFactor - 1;
grid = cartGrid([xres, yres, 1], [X, Y, 1]);

% Setting correct z-coordinates and computing geometry
zvals = info.z(1:downSamplingFactor:(end-1), 1:downSamplingFactor:(end-1));

end