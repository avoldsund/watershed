function interpolatedMatrix = interpolate_extreme_values(minValue, maxValue, generalMatrix)
% INTERPOLATE_EXTREME_VALUES Interpolation of extreme values in a general
% matrix.
% interpolate_extreme_values(min, max, generalMatrix) sets all
% elements in generalMatrix outside of the interval (min, max) to NaN. The
% function inpaint_nans will interpolate all NaN-values using the values of
% the neighboring points in the matrix.
% 
% Example:
%   interpolatedMatrix = interpolate_extreme_values(0, 1000, heightMatrix)
%   can be used to interpolate height data in a terrain. If you know that
%   the minimum height of the terrain is 0 meters, and the max height is
%   1000 meters, all values outside this interval will be interpolated.
%
% inpaint_nans was written by John D'Errico and can be found at
% http://www.mathworks.com/matlabcentral/fileexchange/4551-inpaint-nans
 
addpath('../Inpaint_nans/')

hasValuesAboveMax = max(max(generalMatrix)) > maxValue;
hasValuesBelowMin = min(min(generalMatrix)) < minValue;

if (hasValuesAboveMax || hasValuesBelowMin)
    generalMatrix(generalMatrix > maxValue) = NaN;
    generalMatrix(generalMatrix < minValue) = NaN;
    interpolatedMatrix = inpaint_nans(double(generalMatrix));
    return;
end

interpolatedMatrix = generalMatrix;

end