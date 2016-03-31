import numpy as np


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
    boundary_indices = np.empty(number_of_boundary_nodes)

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
