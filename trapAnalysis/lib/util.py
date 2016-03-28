import numpy as np


class Landscape:

    coordinates = None

    num_of_nodes_x = None
    num_of_nodes_y = None
    total_number_of_nodes = None

    x_coord_min = None
    x_coord_max = None
    y_coord_min = None
    y_coord_max = None

    def __init__(self, geo_transform, nx, ny):
        self.num_of_nodes_x = nx
        self.num_of_nodes_y = ny
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * self.num_of_nodes_x
        self.y_min = self.y_max + geo_transform[5] * self.num_of_nodes_y
        self.total_number_of_nodes = nx * ny
        self.coordinates = np.empty((self.total_number_of_nodes, 3))


    def get_node_neighbors(self, node_index):
        return None

    def find_steepest_neighbors(self, node_index):
        return None

    def get_node_index(self, x_coord, y_coord):
        return None
