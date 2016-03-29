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


def get_node_neighbors(node_index, num_of_nodes_x, num_of_nodes_y):
    """
    Returns the indices of the neighbors given the node index
    :param node_index: Index of node
    :return neighbors: Array of neighbor indices
    """
    total_number_of_nodes = num_of_nodes_x * num_of_nodes_y

    neighbors = np.array([node_index - 1, node_index + 1, node_index - num_of_nodes_x - 1,
                          node_index - num_of_nodes_x, node_index - num_of_nodes_x + 1,
                          node_index + num_of_nodes_x - 1, node_index + num_of_nodes_x,
                          node_index + num_of_nodes_x + 1])
    # Remove all neighbors with negative indices or indices exceeding the total number of nodes
    valid_neighbors = neighbors[(neighbors >= 0) & (neighbors < total_number_of_nodes)]

    return valid_neighbors


def find_steepest_neighbors(self, node_index):
    return None


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
