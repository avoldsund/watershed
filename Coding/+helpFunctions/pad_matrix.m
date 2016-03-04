function paddedMatrix = pad_matrix(xDim, yDim, generalMatrix)

[sizeX, sizeY] = size(generalMatrix);
paddedMatrix = zeros(xDim, yDim);
paddedMatrix(1:sizeX, 1:sizeY) = generalMatrix;

end