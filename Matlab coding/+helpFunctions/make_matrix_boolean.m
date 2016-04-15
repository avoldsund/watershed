function booleanMatrix = make_matrix_boolean(generalMatrix)
% MAKE_MATRIX_BOOLEAN Turns a general matrix into a boolean matrix. All
% nonzero elements are set to 1, and the rest will be 0.
%
% Example:
% A = [1 2 3; 0 -1 3; 0 0 2]
% becomes
% B = [1 1 1; 0 1 1; 0 0 1]

booleanMatrix = (generalMatrix ~= 0);

end