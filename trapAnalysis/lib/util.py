import numpy as np
import math


class Landscape:

    def __init__(self, geo_transform, nx, ny):

        self.num_of_nodes_x = nx
        self.num_of_nodes_y = ny
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * (self.num_of_nodes_x - 1)
        self.y_min = self.y_max + geo_transform[5] * (self.num_of_nodes_y - 1)
        self.total_number_of_nodes = nx * ny
        self.coordinates = np.empty((self.total_number_of_nodes, 3))
        self.steepest_neighbor = np.empty(self.total_number_of_nodes)

        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x


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
        neighbors[counter] = get_node_neighbors_boundary_array(index, num_of_nodes_x, num_of_nodes_y)
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


def get_steepest_neighbors(num_of_cols, num_of_rows, heights):

    boundary_indices = get_boundary_indices(num_of_cols, num_of_rows)
    all_neighbors = get_node_neighbors_for_indices(boundary_indices, num_of_cols, num_of_rows)
    heights_indices = heights[boundary_indices]
    indices_with_largest_derivative = np.empty(len(boundary_indices), dtype=int)

    for i in range(len(boundary_indices)):
        neighbors = np.asarray(all_neighbors[i])
        index_height_vec = np.array([heights_indices[i]] * len(neighbors))
        heights_of_neighbors = heights[neighbors]

        diff = index_height_vec - heights_of_neighbors
        index_steepest = -1
        if np.amax(diff) > 0:
            index_steepest = neighbors[np.argmax(diff)]
        indices_with_largest_derivative[i] = index_steepest

    return indices_with_largest_derivative


def get_steepest_neighbors_interior(num_of_cols, num_of_rows, heights):

    interior_indices = get_interior_indices(num_of_cols, num_of_rows)
    all_neighbors = get_node_neighbors_for_indices(interior_indices, num_of_cols, num_of_rows)
    heights_indices = heights[interior_indices]
    indices_with_largest_derivative = np.empty(len(interior_indices), dtype=int)

    for i in range(len(interior_indices)):
        print i
        neighbors = np.asarray(all_neighbors[i])
        index_height_vec = np.array([heights_indices[i]] * len(neighbors))
        heights_of_neighbors = heights[neighbors]

        diff = index_height_vec - heights_of_neighbors
        index_steepest = -1
        if np.amax(diff) > 0:
            index_steepest = neighbors[np.argmax(diff)]
        indices_with_largest_derivative[i] = index_steepest

    return indices_with_largest_derivative


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


def get_steepest_neighbors_interior_improved(num_of_cols, num_of_rows, heights):

    indices, neighbors = get_neighbors_interior(num_of_cols, num_of_rows)
    heights_array = np.transpose(np.tile(heights[indices], (8, 1)))

    delta_z = heights_array - heights[neighbors]
    delta_x = np.array([math.sqrt(200), 10, math.sqrt(200), 10, 10, math.sqrt(200), 10, math.sqrt(200)])
    derivatives = np.divide(delta_z, delta_x)

    indices_of_steepest = derivatives.argmax(axis=1)
    steepest_neighbors = np.choose(indices_of_steepest, neighbors.T)

    return steepest_neighbors
