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

        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x


def get_node_neighbors_boundary(node_index, num_of_nodes_x, num_of_nodes_y):
    """
    Returns the indices of the neighbors of boundary nodes given the node index
    :param node_index: Index of node
    :return neighbors: Array of neighbor indices
    """

    (r, c) = get_node_row_and_col_from_index(node_index, num_of_nodes_x)

    for k in range(max(0, c-1)), min(num_of_nodes_x, c+1):
        for l in range(max(0, r-1), min(num_of_nodes_y, r+1)):
            nbr_index = get_node_index_from_row_and_col(l, k)

    total_number_of_nodes = num_of_nodes_x * num_of_nodes_y
    valid_neighbors = np.array([node_index - 1, node_index + 1, node_index - num_of_nodes_x - 1,
                          node_index - num_of_nodes_x, node_index - num_of_nodes_x + 1,
                          node_index + num_of_nodes_x - 1, node_index + num_of_nodes_x,
                          node_index + num_of_nodes_x + 1])
    # Remove all neighbors with negative indices or indices exceeding the total number of nodes
    # valid_neighbors = neighbors[(neighbors >= 0) & (neighbors < total_number_of_nodes)]

    return valid_neighbors


def find_steepest_neighbors(node_index):
    return None


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

