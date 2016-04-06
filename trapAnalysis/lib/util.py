import numpy as np
import math


def get_row_and_col_from_index(node_index, number_of_cols):
    """
    Given an index in the 1d-grid, the row number and column coordinates in the 2d-grid is returned
    :param node_index: Index of node in 1d-grid
    :param number_of_cols: Number of columns in the 2d-grid
    :return coordinates: The coordinates of the node with index node_index in the 2d-grid, (r, c)
    """

    r = node_index/number_of_cols
    c = node_index % number_of_cols
    row_col = (r, c)

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


def get_node_index(x_coord, y_coord, num_of_nodes_x, num_of_nodes_y):
    """
    Given a node in the 2d-grid we return the index in the 1d-array
    :param x_coord: Coordinate in the x-direction
    :param y_coord: Coordinate in the y-direction
    :param num_of_nodes_x: Number of grid points in the x-direction
    :param num_of_nodes_y: Number of grid points in the y-direction
    :return node_index: Index in the 1d-array
    """
    node_index = x_coord + num_of_nodes_x * (num_of_nodes_y - (y_coord + 1))

    return node_index


def get_node_neighbors_boundary(node_index, num_of_nodes_x, num_of_nodes_y):
    """
    Returns the indices of the neighbors of boundary nodes given the node index
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


def get_node_neighbors_for_indices_array(indices, num_of_nodes_x, num_of_nodes_y):
    """
    NOTE: This is returning the neighbors for all indices as 2d-array
    :param indices:
    :param num_of_nodes_x:
    :param num_of_nodes_y:
    :return:
    """
    neighbors = np.empty(len(indices), dtype=object)

    counter = 0
    for index in indices:
        neighbors[counter] = get_node_neighbors_boundary(index, num_of_nodes_x, num_of_nodes_y)
        counter += 1

    return neighbors


def get_node_neighbors_for_indices(indices, num_of_nodes_x, num_of_nodes_y):

    neighbors = []
    for index in indices:
        neighbors.append(get_node_neighbors_boundary(index, num_of_nodes_x, num_of_nodes_y))

    return neighbors


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


def get_boundary_indices(num_of_cols, num_of_rows):
    """
    Returns an array of all indices in the 1d-array that are boundary nodes in the 2d-grid
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return boundary_indices: Array of boundary indices
    """

    number_of_boundary_nodes = 2 * num_of_cols + 2 * num_of_rows - 4
    boundary_indices = np.empty(number_of_boundary_nodes, dtype=int)

    top = np.arange(0, num_of_cols, 1)
    bottom = np.arange(num_of_cols * num_of_rows - num_of_cols, num_of_cols * num_of_rows, 1)
    left = np.arange(num_of_cols, num_of_cols * num_of_rows - num_of_cols, num_of_cols)
    right = np.arange(2 * num_of_cols - 1, num_of_cols * num_of_rows - 1, num_of_cols)

    boundary_indices[0: num_of_cols] = top
    boundary_indices[num_of_cols: 2*num_of_cols] = bottom
    boundary_indices[2 * num_of_cols: 3 * num_of_cols - 2] = left
    boundary_indices[3 * num_of_cols - 2: 4 * num_of_cols - 2] = right

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


def get_downslope_neighbors_boundary(num_of_cols, num_of_rows, heights):
    """
    Returns the indices of the neighbors with the largest derivative for each node
    :param num_of_cols: Number of nodes in the x-direction
    :param num_of_rows: Number of nodes in the y-direction
    :param heights: Heights of all nodes
    :return indices_with_largest_derivative: The indices of the neighbors with the largest derivatives for each node
    """
    boundary_indices = get_boundary_indices(num_of_cols, num_of_rows)
    all_neighbors = get_node_neighbors_for_indices(boundary_indices, num_of_cols, num_of_rows)
    heights_indices = heights[boundary_indices]
    indices_downslope_neighbors = np.empty(len(boundary_indices), dtype=int)

    for i in range(len(boundary_indices)):
        neighbors = np.asarray(all_neighbors[i])
        index_height_vec = np.array([heights_indices[i]] * len(neighbors))
        heights_of_neighbors = heights[neighbors]

        diff = index_height_vec - heights_of_neighbors
        index_steepest = -1
        if np.amax(diff) > 0:
            index_steepest = neighbors[np.argmax(diff)]
        indices_downslope_neighbors[i] = index_steepest

    return indices_downslope_neighbors


def get_neighbors_interior(num_of_cols, num_of_rows):
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


def get_downslope_neighbors_interior(num_of_cols, num_of_rows, heights):
    """
    Returns the indices of all neighbors with steepest derivatives
    :param num_of_cols: Number of nodes in x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param heights: The heights of all nodes
    :return indices_of_steepest_neighbors: The index of the steepest neighbor for each node
    """
    indices, neighbors = get_neighbors_interior(num_of_cols, num_of_rows)
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
    is_minimum = flat_derivatives[indices_in_flat] < 0

    # Setting all local minimum indices to -1
    indices_downslope_neighbors = np.choose(indices_of_steepest, neighbors.T)
    indices_downslope_neighbors[is_minimum] = -1

    return indices_downslope_neighbors


def get_downslope_neighbors(num_of_cols, num_of_rows, heights):
    """
    Returns the downslope neighbors for all indices
    :param num_of_cols: Number of nodes is x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param heights: Heighs of all nodes
    :return downslope_neighbors: Indices of all downslope neighbors for each node. Equal to -1 if the node is a minimum.
    """

    boundary_indices = get_boundary_indices(num_of_cols, num_of_rows)
    interior_indices = get_interior_indices(num_of_cols, num_of_rows)

    indices = np.concatenate((boundary_indices, interior_indices))

    downslope_neighbors_boundary = get_downslope_neighbors_boundary(num_of_cols, num_of_rows, heights)
    downslope_neighbors_interior = get_downslope_neighbors_interior(num_of_cols, num_of_rows, heights)
    downslope_neighbors = np.concatenate((downslope_neighbors_boundary, downslope_neighbors_interior))

    # Change the order such that we get the downslope neighbors for index 0 to (nx x ny) chronologically
    downslope_neighbors = np.column_stack((indices, downslope_neighbors))
    downslope_neighbors = downslope_neighbors[np.argsort(downslope_neighbors[:, 0])][:, 1]

    return downslope_neighbors
