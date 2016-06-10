import numpy as np
import math


def get_row_and_col_from_index(node_index, number_of_cols):
    """
    Given an index in the 1d-grid, the row number and column coordinates in the 2d-grid is returned
    :param node_index: Index of node in 1d-grid
    :param number_of_cols: Number of columns in the 2d-grid
    :return row_col: The coordinates of the node with index node_index in the 2d-grid, (r, c)
    """

    r = node_index/number_of_cols
    c = node_index % number_of_cols
    row_col = (r, c)

    return row_col


def get_row_and_col_from_indices(node_indices, number_of_cols):
    """
    Return (r, c) for all indices in node_indices.
    :param node_indices: Indices in the 1d-grid.
    :param number_of_cols: Number of columns in the 2d-grid.
    :return row_col: (r, c) for every index
    """

    row_col = np.empty((len(node_indices), 2), dtype=int)
    row_col[:, 0] = np.divide(node_indices, number_of_cols)
    row_col[:, 1] = node_indices % number_of_cols

    return row_col


def get_index_from_row_and_col(row_number, col_number, number_of_cols):
    """
    Returns the node index in the 1d-array given the row number and column number in the 2d-grid
    :param row_number: Row number in the 2d-grid
    :param col_number: Column number in the 2d-grid
    :param number_of_cols: Number of columns in the 2d-grid
    :return node_index: Index for the (r, c)-node in the 1d-array
    """

    node_index = col_number + row_number * number_of_cols

    return node_index


def get_node_index_from_coords(x_coord, y_coord, num_of_nodes_x, num_of_nodes_y):
    """
    Given a coordinate in the 2d-grid we return the index in the 1d-array
    :param x_coord: Coordinate in the x-direction
    :param y_coord: Coordinate in the y-direction
    :param num_of_nodes_x: Number of grid points in the x-direction
    :param num_of_nodes_y: Number of grid points in the y-direction
    :return node_index: Index in the 1d-array
    """
    node_index = x_coord + num_of_nodes_x * (num_of_nodes_y - (y_coord + 1))

    return node_index


def get_neighbors_boundary(node_index, num_of_nodes_x, num_of_nodes_y):
    """
    Returns the indices of the neighbors of boundary nodes given the node index in 1d-grid
    :param node_index: Index of node
    :return neighbors: List of neighbor indices
    """

    r, c = get_row_and_col_from_index(node_index, num_of_nodes_x)
    neighbors = []

    for l in range(max(0, r-1), min(num_of_nodes_y, (r+1) + 1)):  # Add 1 so range includes r+1
        for k in range(max(0, c-1), min(num_of_nodes_x, (c+1) + 1)):  # Add 1 so range includes c+1
            neighbors.append(get_index_from_row_and_col(l, k, num_of_nodes_x))

    neighbors.remove(node_index)

    return neighbors


def get_all_neighbors_interior(num_of_cols, num_of_rows):
    """
    Returns the neighbors of the interior nodes
    :param num_of_cols:
    :param num_of_rows:
    :return:
    """

    indices = get_interior_indices(num_of_cols, num_of_rows)
    nr_of_nodes = len(indices)
    neighbors = np.empty((nr_of_nodes, 8), dtype=int)
    one_array = np.ones(nr_of_nodes)
    n_array = one_array * num_of_cols

    neighbors[:, 0] = indices - n_array - one_array
    neighbors[:, 1] = indices - n_array
    neighbors[:, 2] = indices - n_array + one_array
    neighbors[:, 3] = indices - one_array
    neighbors[:, 4] = indices + one_array
    neighbors[:, 5] = indices + n_array - one_array
    neighbors[:, 6] = indices + n_array
    neighbors[:, 7] = indices + n_array + one_array

    return indices, neighbors


def get_neighbors_for_interior_indices(indices, num_of_cols):
    """
    Returns neighbors for all interior indices in the 2d-grid.
    :param indices: The indices in the 1d-grid.
    :param num_of_cols: Number of columns in the 2d-grid.
    :return neighbors: All neighbors in the 1d-grid for every interior node.
    """

    nr_of_nodes = len(indices)
    neighbors = np.empty((nr_of_nodes, 8), dtype=int)
    one_array = np.ones(nr_of_nodes, dtype=int)
    n_array = one_array * num_of_cols

    neighbors[:, 0] = indices - n_array - one_array
    neighbors[:, 1] = indices - n_array
    neighbors[:, 2] = indices - n_array + one_array
    neighbors[:, 3] = indices - one_array
    neighbors[:, 4] = indices + one_array
    neighbors[:, 5] = indices + n_array - one_array
    neighbors[:, 6] = indices + n_array
    neighbors[:, 7] = indices + n_array + one_array

    return neighbors


def get_neighbors_for_indices(indices, num_of_nodes_x, num_of_nodes_y):
    """
    Given a vector of arbitrary indices, either at the boundary or interior nodes, return their neighbors.
    :param indices: Indices in the 1d-grid.
    :param num_of_nodes_x: Number of nodes in the x-direction
    :param num_of_nodes_y: Number of nodes in the y-direction
    :return neighbors: The indices' neighbors.
    """

    neighbors = []
    for index in indices:
        neighbors.append(get_neighbors_boundary(index, num_of_nodes_x, num_of_nodes_y))

    return neighbors


def get_padded_neighbors(indices, num_of_nodes_x, num_of_nodes_y):
    """
    Returns a numpy array with eight elements for each node. If the node is a boundary node, many of these will be None.
    :param indices: An array of indices
    :param num_of_nodes_x: Number of columns in the 2d-grid
    :param num_of_nodes_y: Number of rows in the 2d-grid
    :return neighbors: A 2d numpy array with the neighbors for all the nodes. The boundary nodes are padded with None
    """

    nr_of_indices = len(indices)
    neighbors = np.ones((nr_of_indices, 8), dtype=int) * -1

    for i in range(nr_of_indices):
        neighbors_index = np.asarray(get_neighbors_boundary(indices[i], num_of_nodes_x, num_of_nodes_y))
        neighbors[i, 0:len(neighbors_index)] = neighbors_index

    return neighbors


def get_neighbors_for_indices_array(indices, num_of_cols, num_of_rows):
    """
    Returns the neighbors for a set of indices. All nodes will have 8 elements in their neighbor array, but boundary
    nodes are padded with None.
    :param indices: An array of indices
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return nbrs: The 2d-array with the neighbors for all the indices
    """

    are_boundary = are_boundary_nodes_bool(indices, num_of_cols, num_of_rows)
    boundary = indices[are_boundary]
    interior = indices[are_boundary == False]

    nbrs_interior = get_neighbors_for_interior_indices(interior, num_of_cols)
    nbrs_boundary = get_padded_neighbors(boundary, num_of_cols, num_of_rows)

    nbrs = np.concatenate((np.column_stack((interior, nbrs_interior)),
                           np.column_stack((boundary, nbrs_boundary))))
    nbrs = nbrs[np.argsort(nbrs[:, 0])][:, 1:]  # Remove the column used for sorting

    return nbrs


def is_boundary_node(node_index, num_of_cols, num_of_rows):
    """
    Returns true if the node is on the boundary, otherwise returns false
    :param node_index: Index of node in 1d-array
    :param num_of_cols: Number of columns in 2d-grid
    :param num_of_rows: Number of rows in 2d-grid
    :return is_boundary: True if node is a boundary point, otherwise false
    """

    is_top = node_index < num_of_cols
    is_left = node_index % num_of_cols == 0
    is_right = (node_index + 1) % num_of_cols == 0
    is_bottom = ((num_of_cols * num_of_rows - num_of_cols) <= node_index) and (node_index < num_of_cols * num_of_rows)

    is_boundary = is_top or is_left or is_right or is_bottom

    return is_boundary


def are_boundary_nodes_bool(indices, num_of_cols, num_of_rows):
    """
    Given an array of indices, returns an array with true for boundary nodes, false for interior nodes
    :param indices: Array of indices
    :param num_of_cols: Number of columns in 2d-grid
    :param num_of_rows: Number of rows in 2d-grid
    :return bool_arr: An array with True or False. True if element in indices is a boundary node, false if not.
    """

    bool_arr = (indices < num_of_cols) | (indices % num_of_cols == 0)| ((indices + 1) % num_of_cols == 0) \
               | ((num_of_cols * num_of_rows - num_of_cols) <= indices) & (indices < num_of_cols * num_of_rows)

    return bool_arr


def are_boundary_nodes(indices, num_of_cols, num_of_rows):
    """
    Given an array of indices, returns the indices of the boundary nodes
    :param indices: All indices to be checked if they are boundary nodes
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return are_boundary: An array of the boundary indices
    """

    are_top = indices[indices < num_of_cols]
    are_left = indices[indices % num_of_cols == 0]
    are_right = indices[(indices + 1) % num_of_cols == 0]

    are_bottom = indices[((num_of_cols * num_of_rows - num_of_cols) <= indices) &
                         (indices < num_of_cols * num_of_rows)]

    are_boundary = np.unique(np.sort(np.concatenate((are_top, are_left, are_right, are_bottom))))

    return are_boundary


def get_boundary_indices(num_of_cols, num_of_rows):
    """
    Returns an array of all indices in the 1d-array that are boundary nodes in the 2d-grid
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return boundary_indices: Array of boundary indices
    """

    top = np.arange(0, num_of_cols, 1)
    bottom = np.arange(num_of_cols * num_of_rows - num_of_cols, num_of_cols * num_of_rows, 1)
    left = np.arange(num_of_cols, num_of_cols * num_of_rows - num_of_cols, num_of_cols)
    right = np.arange(2 * num_of_cols - 1, num_of_cols * num_of_rows - 1, num_of_cols)

    boundary_indices = np.concatenate((top, bottom, left, right))
    boundary_indices.sort()

    return boundary_indices


def get_interior_indices(num_of_cols, num_of_rows):
    """
    Returns array of interior indices
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return interior_indices: Indices of all interior nodes in the 2d-grid
    """

    indices = np.arange(0, num_of_cols * num_of_rows, 1)
    interior_indices = np.setdiff1d(indices, get_boundary_indices(num_of_cols, num_of_rows))

    return interior_indices


def get_different_boundaries(nx, ny):
    """
    Returns indices for the different boundary types: corners, top, right, bottom, left.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :return corners, top, right, bottom, left: Arrays with the indices of the different boundaries.
    """

    total_nodes = nx * ny

    corners = np.array([0, nx - 1, total_nodes - nx, total_nodes - 1])
    top_wo_corners = np.arange(1, nx - 1, 1)
    bottom_wo_corners = np.arange(total_nodes - nx + 1, total_nodes - 1, 1)
    left_wo_corners = np.arange(nx, total_nodes - nx, nx)
    right_wo_corners = np.arange(2 * nx - 1, nx * ny - 1, nx)

    return corners, top_wo_corners, right_wo_corners, bottom_wo_corners, left_wo_corners


def get_neighbors_derivatives_dictionary(heights, nx, ny):
    """
    Returns a dictionary where the key is the index of the node, and the value is (neighbor_indices, derivatives)
    :param heights: Heights of all nodes in the grid.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :return neighbors_derivatives_dict: A dictionary with key as index, value as (neighbors, derivatives)
    """

    neighbors_derivatives_dict = {}

    corners, top, right, bottom, left = get_different_boundaries(nx, ny)
    interior = get_interior_indices(nx, ny)

    corner_nbrs = np.array(get_neighbors_for_indices(corners, nx, ny))
    top_nbrs = np.array(get_neighbors_for_indices(top, nx, ny))
    right_nbrs = np.array(get_neighbors_for_indices(right, nx, ny))
    bottom_nbrs = np.array(get_neighbors_for_indices(bottom, nx, ny))
    left_nbrs = np.array(get_neighbors_for_indices(left, nx, ny))
    interior_nbrs = get_neighbors_for_interior_indices(interior, nx)

    corner_distance = np.array([[10, 10, math.sqrt(200)],
                                [10, math.sqrt(200), 10],
                                [10, math.sqrt(200), 10],
                                [math.sqrt(200), 10, 10]])
    top_distance = np.array([10, 10, math.sqrt(200), 10, math.sqrt(200)])
    right_distance = np.roll(top_distance, 1)
    left_distance = np.roll(top_distance, 2)
    bottom_distance = np.roll(top_distance, 3)
    interior_distance = np.array([math.sqrt(200), 10, math.sqrt(200), 10, 10, math.sqrt(200), 10, math.sqrt(200)])

    corner_delta_height = np.subtract((np.array([heights[corners]])).T, heights[corner_nbrs])
    top_delta_height = np.subtract((np.array([heights[top]])).T, heights[top_nbrs])
    right_delta_height = np.subtract((np.array([heights[right]])).T, heights[right_nbrs])
    bottom_delta_height = np.subtract((np.array([heights[bottom]])).T, heights[bottom_nbrs])
    left_delta_height = np.subtract((np.array([heights[left]])).T, heights[left_nbrs])
    interior_delta_height = np.subtract((np.array([heights[interior]])).T, heights[interior_nbrs])

    corner_derivatives = np.divide(corner_delta_height, corner_distance)
    top_derivatives = np.divide(top_delta_height, top_distance)
    right_derivatives = np.divide(right_delta_height, right_distance)
    bottom_derivatives = np.divide(bottom_delta_height, bottom_distance)
    left_derivatives = np.divide(left_delta_height, left_distance)
    interior_derivatives = np.divide(interior_delta_height, interior_distance)

    for i in range(len(corners)):
        neighbors_derivatives_dict[corners[i]] = (corner_nbrs[i], corner_derivatives[i])

    for i in range(len(top)):
        neighbors_derivatives_dict[top[i]] = (top_nbrs[i], top_derivatives[i])

    for i in range(len(right)):
        neighbors_derivatives_dict[right[i]] = (right_nbrs[i], right_derivatives[i])

    for i in range(len(bottom)):
        neighbors_derivatives_dict[bottom[i]] = (bottom_nbrs[i], bottom_derivatives[i])

    for i in range(len(left)):
        neighbors_derivatives_dict[left[i]] = (left_nbrs[i], left_derivatives[i])

    for i in range(len(interior)):
        neighbors_derivatives_dict[interior[i]] = (interior_nbrs[i], interior_derivatives[i])

    return neighbors_derivatives_dict


def get_downslope_indices_corners(num_of_cols, num_of_rows, heights):
    """
    Returns the indices of the downslope neighbors for all corners in the grid
    :param num_of_cols: Number of nodes in the x-direction
    :param num_of_rows: Number of nodes in the y-direction
    :param heights: Heights of all points in the grid
    :return indices_downslope_neighbors: Indices of the downslope neighbors for the corners
    """

    total_nodes = num_of_cols * num_of_rows
    corners = np.array([0, num_of_cols - 1, total_nodes - num_of_cols, total_nodes - 1])
    corner_neighbors = np.array(get_neighbors_for_indices(corners, num_of_cols, num_of_rows))
    corners_dist_to_neighbors = np.array([[10, 10, math.sqrt(200)],
                                          [10, math.sqrt(200), 10],
                                          [10, math.sqrt(200), 10],
                                          [math.sqrt(200), 10, 10]])
    corners_delta_z = np.transpose(np.tile(heights[corners], (3, 1))) - heights[corner_neighbors]

    corners_derivatives = np.divide(corners_delta_z, corners_dist_to_neighbors)
    corners_downslope = corners_derivatives.argmax(axis=1)
    flat_corners_derivatives = corners_derivatives.flatten()
    col_multiplier = np.arange(0, len(corners), 1) * 3
    indices_in_flat = col_multiplier + corners_downslope
    is_minimum = flat_corners_derivatives[indices_in_flat] <= 0
    indices_downslope_neighbors = np.choose(corners_downslope, corner_neighbors.T)
    indices_downslope_neighbors[is_minimum] = -1
    indices_downslope_neighbors = np.column_stack((corners, indices_downslope_neighbors))

    return indices_downslope_neighbors


def get_downslope_indices_sides(num_of_cols, num_of_rows, heights):
    """
    Returns a 2d-array where the first column is the node indices, and the second one is the downslope neighbor
    :param num_of_cols: Number of nodes in the x-direction
    :param num_of_rows: Number of nodes in the y-direction
    :param heights: The heights of all nodes
    :return downslope_neighbors: Indices of the downslope neighbor for the nodes
    """

    total_nodes = num_of_cols * num_of_rows

    top = np.arange(1, num_of_cols - 1, 1)
    bottom = np.arange(total_nodes - num_of_cols + 1, total_nodes - 1, 1)
    left = np.arange(num_of_cols, total_nodes - num_of_cols, num_of_cols)
    right = np.arange(2 * num_of_cols - 1, num_of_cols * num_of_rows - 1, num_of_cols)
    indices = np.concatenate((top, bottom, left, right))

    neighbors = np.array(get_neighbors_for_indices(indices, num_of_cols, num_of_rows))

    dist_to_neighbors_top = np.array([10, 10, math.sqrt(200), 10, math.sqrt(200)])
    dist_to_neighbors_right = np.roll(dist_to_neighbors_top, 1)
    dist_to_neighbors_left = np.roll(dist_to_neighbors_top, 2)
    dist_to_neighbors_bottom = np.roll(dist_to_neighbors_top, 3)

    distance_top = np.tile(dist_to_neighbors_top, (len(top), 1))
    distance_bottom = np.tile(dist_to_neighbors_bottom, (len(bottom), 1))
    distance_left = np.tile(dist_to_neighbors_left, (len(left), 1))
    distance_right = np.tile(dist_to_neighbors_right, (len(right), 1))
    dist_to_neighbors = np.concatenate((distance_top, distance_bottom, distance_left, distance_right))

    delta_z = np.transpose(np.tile(heights[indices], (5, 1))) - heights[neighbors]
    derivatives = np.divide(delta_z, dist_to_neighbors)

    downslope_indices = derivatives.argmax(axis=1)
    flat_derivatives = derivatives.flatten()
    col_multiplier = np.arange(0, len(indices), 1) * 5
    indices_in_flat = col_multiplier + downslope_indices
    is_minimum = flat_derivatives[indices_in_flat] <= 0

    downslope_neighbors = np.choose(downslope_indices, neighbors.T)
    downslope_neighbors[is_minimum] = -1
    downslope_neighbors = np.column_stack((indices, downslope_neighbors))

    downslope_neighbors = downslope_neighbors[np.argsort(downslope_neighbors[:, 0])]
    #downslope_neighbors = downslope_neighbors[:, 1]

    return downslope_neighbors


def get_downslope_indices_boundary(num_of_cols, num_of_rows, heights):
    """
    Returns the indices of the neighbors with the largest derivative for all boundary nodes
    :param num_of_cols: Number of nodes in the x-direction
    :param num_of_rows: Number of nodes in the y-direction
    :param heights: Heights of all nodes
    :return indices_with_largest_derivative: The indices of the neighbors with the largest derivatives for each node
    """

    corner_downslope_indices = get_downslope_indices_corners(num_of_cols, num_of_rows, heights)
    side_downslope_indices = get_downslope_indices_sides(num_of_cols, num_of_rows, heights)
    boundary_downslope_indices = np.concatenate((corner_downslope_indices, side_downslope_indices))
    boundary_downslope_indices = boundary_downslope_indices[np.argsort(boundary_downslope_indices[:, 0])]

    return boundary_downslope_indices


def get_downslope_indices_interior(num_of_cols, num_of_rows, heights):
    """
    Returns the indices of all neighbors with steepest derivatives
    :param num_of_cols: Number of nodes in x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param heights: The heights of all nodes
    :return indices_of_steepest_neighbors: The index of the steepest neighbor for each node
    """
    indices, neighbors = get_all_neighbors_interior(num_of_cols, num_of_rows)
    heights_of_indices = np.transpose(np.tile(heights[indices], (8, 1)))

    # Calculating the derivatives of all neighbors
    delta_z = heights_of_indices - heights[neighbors]
    delta_x = np.array([math.sqrt(200), 10, math.sqrt(200), 10, 10, math.sqrt(200), 10, math.sqrt(200)])
    derivatives = np.divide(delta_z, delta_x)

    # Get all indices of the maximum derivatives for each node
    indices_of_steepest = derivatives.argmax(axis=1)

    # Change the indices of nodes that are local minimums
    flat_derivatives = derivatives.flatten()
    col_multiplier = np.arange(0, len(indices), 1) * 8
    indices_in_flat = col_multiplier + indices_of_steepest
    is_minimum = flat_derivatives[indices_in_flat] <= 0

    # Setting all local minimum indices to -1
    indices_downslope_neighbors = np.choose(indices_of_steepest, neighbors.T)
    indices_downslope_neighbors[is_minimum] = -1

    indices_downslope_neighbors = np.column_stack((indices, indices_downslope_neighbors))

    return indices_downslope_neighbors


def get_downslope_indices(num_of_cols, num_of_rows, heights):
    """
    Returns the downslope neighbors for all indices
    :param num_of_cols: Number of nodes is x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param heights: Heighs of all nodes
    :return downslope_neighbors: Indices of all downslope neighbors for each node. Equal to -1 if the node is a minimum.
    """

    downslope_indices_boundary = get_downslope_indices_boundary(num_of_cols, num_of_rows, heights)
    downslope_indices_interior = get_downslope_indices_interior(num_of_cols, num_of_rows, heights)
    downslope_indices = np.concatenate((downslope_indices_boundary, downslope_indices_interior))

    # Change the order such that we get the downslope neighbors for index 0 to (nx x ny) chronologically
    downslope_indices = downslope_indices[np.argsort(downslope_indices[:, 0])][:, 1]

    return downslope_indices
