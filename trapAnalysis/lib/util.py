import numpy as np


class Landscape:

    coordinates = None

    num_of_nodes_x = None
    num_of_nodes_y = None
    total_number_of_nodes = None

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    step_size = None

    def __init__(self, geo_transform, nx, ny):
        self.num_of_nodes_x = nx
        self.num_of_nodes_y = ny
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * self.num_of_nodes_x
        self.y_min = self.y_max + geo_transform[5] * self.num_of_nodes_y
        self.total_number_of_nodes = nx * ny
        self.coordinates = np.empty((self.total_number_of_nodes, 3))
        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x


    def get_node_neighbors(self, node_index):
        """
        Returns the indices of the neighbors given the node index
        :param node_index: Index of node
        :return neighbors: Array of neighbor indices
        """

        neighbors = np.array([node_index - 1, node_index + 1, node_index - self.num_of_nodes_x - 1,
                              node_index - self.num_of_nodes_x, node_index - self.num_of_nodes_x + 1,
                              node_index + self.num_of_nodes_x - 1, node_index + self.num_of_nodes_x,
                              node_index + self.num_of_nodes_x + 1])
        # Remove all neighbors with negative indices or indices exceeding the total number of nodes
        valid_neighbors = neighbors[(neighbors >= 0) & (neighbors < self.total_number_of_nodes)]

        return valid_neighbors


    def get_node_index(self, x_coord, y_coord):
        """
        Given a node in the 2d-grid we return the index in the 1d-array
        :param x_coord: Coordinate in the x-direction
        :param y_coord: Coordinate in the y-direction
        :return node_index:
        """
        node_index = x_coord * self.num_of_nodes_x + y_coord + 1

        return node_index


    def find_steepest_neighbors(self, node_index):
        return None
