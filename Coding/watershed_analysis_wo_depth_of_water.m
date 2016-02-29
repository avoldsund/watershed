%% Read the height of an area. Plot the height of the landscape
I = GEOTIFF_READ('C:\Users\Anders O. Voldsund\Dropbox\watershed\Matlab examples\Data\anders_hoh.tiff');

% Flip y axis so image is not mirrored up-down
% caxis shows the range of the heights
im = imshow(I.z, 'xdata', I.x, 'ydata', fliplr(I.y)); caxis([0, 1200]), colorbar