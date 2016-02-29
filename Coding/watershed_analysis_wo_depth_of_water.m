%% Read the height of an area. Plot the height of the landscape
I = GEOTIFF_READ('anders_hoh.tiff');

% Flip y axis so image is not mirrored up-down
% caxis shows the range of the height
im = imshow(I.z, 'xdata', I.x, 'ydata', fliplr(I.y)); caxis([0, 800]), colorbar
title('Height of the area')
axis on
xlabel('Meters')
ylabel('Meters')