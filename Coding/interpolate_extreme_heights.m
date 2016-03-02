function interpolatedMatrix = interpolate_extreme_heights(heightMin, heightMax, heightMatrix)
% INTERPOLATE_EXTREME_HEIGHTS Removes and interpolates NaN values in a
% matrix
% 
% Example:
%   interpolatedMatrix = interpolate_extreme_heights(0, 1000, heightMatrix)
%   Replaces all values below 0 and above 1000 by NaN, then calls a
%   function INPAINT_NANS() to interpolate the NaN-values. Returns
%   interpolatedMatrix.

addpath('../Inpaint_nans/')

hasValuesAboveMax = max(max(heightMatrix)) > heightMax;
hasValuesBelowMin = min(min(heightMatrix)) < heightMin;

if (hasValuesAboveMax || hasValuesBelowMin)
    heightMatrix(heightMatrix > heightMax) = NaN;
    heightMatrix(heightMatrix < heightMin) = NaN;
    interpolatedMatrix = inpaint_nans(double(heightMatrix));
    return;
end

interpolatedMatrix = heightMatrix;

end