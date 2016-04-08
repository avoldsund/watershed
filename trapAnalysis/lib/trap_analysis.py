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
        self.downslope_neighbors = None
        self.node_in_trap_index = None


def get_downslope_minimums(num_of_cols, num_of_rows, downslope_neighbors):

    num_of_nodes = num_of_cols * num_of_rows
    terminal_nodes = np.empty(num_of_nodes, dtype=object)

    for i in range(num_of_nodes):
        if terminal_nodes[i] is None:
            downslope_neighbor = downslope_neighbors[i]
            if downslope_neighbor == -1:  # The node itself is a minimum
                terminal_nodes[i] = i
            elif downslope_neighbors[downslope_neighbor] == -1:  # The downslope neighbor is a minimum
                terminal_nodes[i] = downslope_neighbor
            else:  # Follow the node path to the first minimum
                river = [i]
                while downslope_neighbor != -1:

                    river.append(downslope_neighbor)
                    prev = downslope_neighbor
                    downslope_neighbor = downslope_neighbors[downslope_neighbor]

                    if terminal_nodes[downslope_neighbor] is not None:  # If your river hits an existing river
                        terminal_nodes[river] = terminal_nodes[downslope_neighbor]

                    if downslope_neighbor == -1:
                        terminal_nodes[river] = prev
        else:
            continue

    return terminal_nodes


def get_downslope_minimums_alternative(num_of_cols, num_of_rows, downslope_neighbors):

    num_of_nodes = num_of_cols * num_of_rows
    terminal_nodes = np.empty(num_of_nodes, dtype=object)

    for i in range(num_of_nodes):
        if terminal_nodes[i] is None:
            downslope_neighbor = downslope_neighbors[i]
            if downslope_neighbor == -1:  # The node itself is a minimum
                terminal_nodes[i] = i
            elif downslope_neighbors[downslope_neighbor] == -1:  # The downslope neighbor is a minimum
                terminal_nodes[i] = downslope_neighbor
            else:  # Follow the node path to the first minimum
                river = [i]
                while downslope_neighbor != -1:
                    river.append(downslope_neighbor)
                    prev = downslope_neighbor
                    downslope_neighbor = downslope_neighbors[downslope_neighbor]

                    if terminal_nodes[downslope_neighbor] is not None:  # If your river hits an existing river
                        terminal_nodes[river] = terminal_nodes[downslope_neighbor]

                    if downslope_neighbor == -1:
                        terminal_nodes[river] = prev
        else:
            continue

    return terminal_nodes
