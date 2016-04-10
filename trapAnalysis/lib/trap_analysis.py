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

    # The nodes itself are minimums
    indices_node_is_minimum = np.where(downslope_neighbors == -1)[0]
    terminal_nodes[indices_node_is_minimum] = indices_node_is_minimum

    # The downslope nodes are minimums
    not_local_minimums = np.where(downslope_neighbors != -1)[0]
    indices_to_check = downslope_neighbors[not_local_minimums]
    downslope_is_minimum = np.where(downslope_neighbors[indices_to_check] == -1)
    values = indices_to_check[downslope_is_minimum]
    indices = not_local_minimums[downslope_is_minimum]
    terminal_nodes[indices] = values

    for i in range(num_of_nodes):
        if terminal_nodes[i] is None:

            downslope_neighbor = downslope_neighbors[i]
            river = [i]

            while downslope_neighbor != -1:
                river.append(downslope_neighbor)
                prev = downslope_neighbor
                downslope_neighbor = downslope_neighbors[downslope_neighbor]

                if terminal_nodes[downslope_neighbor] is not None:  # If your river hits an existing river
                    terminal_nodes[river] = terminal_nodes[downslope_neighbor]

                if downslope_neighbor == -1:
                    terminal_nodes[river] = prev

    return terminal_nodes


def get_downslope_minimums_alternative(num_of_cols, num_of_rows, downslope_neighbors):

    num_of_nodes = num_of_cols * num_of_rows
    terminal_nodes = np.empty(num_of_nodes, dtype=object)
    indices_in_terminal = np.zeros(num_of_nodes, dtype=bool)

    # The nodes itself are minimums
    indices_node_is_minimum = np.where(downslope_neighbors == -1)[0]
    terminal_nodes[indices_node_is_minimum] = indices_node_is_minimum
    indices_in_terminal[indices_node_is_minimum] = True

    num_inserted = len(indices_node_is_minimum)

    while num_inserted > 0:
        num_inserted, terminal_nodes = something(terminal_nodes, downslope_neighbors, indices_in_terminal)

    return terminal_nodes


def something(terminal_nodes, downslope_neighbors, indices_in_terminal):

    end_points_not_localized = np.where(indices_in_terminal == False)[0]
    indices_to_check = downslope_neighbors[end_points_not_localized]
    downslope_is_minimum = np.concatenate((np.where(terminal_nodes[indices_to_check] == 0)[0],
                                           np.nonzero(terminal_nodes[indices_to_check])[0])) # np.where(downslope_neighbors[indices_to_check] == -1)[0]
    indices = end_points_not_localized[downslope_is_minimum]
    values = terminal_nodes[indices_to_check[downslope_is_minimum]]

    terminal_nodes[indices] = values
    indices_in_terminal[indices] = True

    return len(values), terminal_nodes


"""
def something(terminal_nodes, downslope_neighbors, indices_in_terminal):

    print 'Terminal nodes: ', terminal_nodes
    print 'Indices in terminal: ', indices_in_terminal

    end_points_not_localized = np.where(indices_in_terminal == False)[0]
    print 'End points not localized: ', end_points_not_localized
    indices_to_check = downslope_neighbors[end_points_not_localized]
    print 'Indices to check: ', indices_to_check
    downslope_is_minimum = terminal_nodes[indices_to_check] # np.where(downslope_neighbors[indices_to_check] == -1)[0]
    print 'Downslope is minimum: ', downslope_is_minimum
    values = indices_to_check[downslope_is_minimum]
    indices = end_points_not_localized[downslope_is_minimum]
    indices_in_terminal[indices] = True
    terminal_nodes[indices] = values
    print 'Values: ', values
    print 'Indices: ', indices
    print 'Number of elements inserted: ', len(values)

    return len(values), terminal_nodes
"""