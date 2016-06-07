import sys
path = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/'
sys.path.insert(0, path + 'util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')
import numpy as np
import util
import networkx
import cPickle
import math
saved_file_dir = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
import time
import matplotlib.pyplot as plt


class Landscape:

    def __init__(self, ds):

        geo_transform = ds.GetGeoTransform()
        self.num_of_nodes_x = ds.RasterXSize
        self.num_of_nodes_y = ds.RasterYSize
        self.x_min = geo_transform[0]
        self.y_max = geo_transform[3]
        self.x_max = self.x_min + geo_transform[1] * (self.num_of_nodes_x - 1)
        self.y_min = self.y_max + geo_transform[5] * (self.num_of_nodes_y - 1)
        self.total_number_of_nodes = self.num_of_nodes_x * self.num_of_nodes_y
        self.coordinates = np.empty((self.total_number_of_nodes, 3))
        self.arr = None

        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x


def get_node_endpoints_alternative(num_of_cols, num_of_rows, downslope_neighbors):
    """
    Returns the end node if one follow the down slope until reaching a local minimum, for every node.
    NOTE: This is a much slower version than the get_node_endpoints-method.
    :param num_of_cols: Number of nodes is x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param downslope_neighbors: Indices of the downslope neighbor for each node. Equal to -1 if the node is a minimum.
    :return terminal_nodes: The indices of the end nodes
    """

    num_of_nodes = num_of_cols * num_of_rows
    terminal_nodes = np.empty(num_of_nodes, dtype=object)

    # The nodes itself are minimums
    indices_node_is_minimum = np.where(downslope_neighbors == -1)[0]
    terminal_nodes[indices_node_is_minimum] = indices_node_is_minimum

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


def get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors):
    """
    Returns the end node if one follow the down slope until reaching a local minimum, for every node
    :param num_of_cols: Number of nodes is x-direction
    :param num_of_rows: Number of nodes in y-direction
    :param downslope_neighbors: Indices of the downslope neighbor for each node. Equal to -1 if the node is a minimum.
    :return terminal_nodes: The indices of the end nodes
    """

    num_of_nodes = num_of_cols * num_of_rows
    terminal_nodes = np.empty(num_of_nodes, dtype=object)
    indices_in_terminal = np.zeros(num_of_nodes, dtype=bool)

    # The nodes itself are minimums
    indices_node_is_minimum = np.where(downslope_neighbors == -1)[0]
    terminal_nodes[indices_node_is_minimum] = indices_node_is_minimum
    indices_in_terminal[indices_node_is_minimum] = True

    num_of_end_nodes_inserted = len(indices_node_is_minimum)

    while num_of_end_nodes_inserted > 0:
        num_of_end_nodes_inserted, terminal_nodes = update_terminal_nodes(terminal_nodes, downslope_neighbors,
                                                                          indices_in_terminal)
    # DO WE REALLY NEED BOTH TERMINAL NODES AND INDICES_IN_TERMINAL????????????????????????????????????????????????
    # YES, BECAUSE WE CAN HAVE 0 (FALSE) AS AN ENDPOINT...?
    # POSSIBLE TO WORK AROUND THIS PROBLEM, I THINK
    return terminal_nodes


def update_terminal_nodes(terminal_nodes, downslope_neighbors, indices_in_terminal):
    """
    Returns an updated terminal_nodes and the number of new end points found.
    The method finds all indices which haven't gotten end nodes yet, and takes a step in the down slope direction. If
    the down slope is a local minimum, these indices will now get end points in terminal_nodes. These end points will be
    found from terminal_nodes.
    :param downslope_neighbors: Indices of the downslope neighbor for each node. Equal to -1 if the node is a minimum.
    :return terminal_nodes: The indices of the end nodes
    """

    indices_end_points_not_localized = np.where(indices_in_terminal == False)[0]
    indices_to_check_if_downslopes_are_minimum = downslope_neighbors[indices_end_points_not_localized]
    downslope_is_minimum = np.concatenate((np.where(terminal_nodes[indices_to_check_if_downslopes_are_minimum] == 0)[0],
                                           np.nonzero(terminal_nodes[indices_to_check_if_downslopes_are_minimum])[0]))

    indices = indices_end_points_not_localized[downslope_is_minimum]
    values = terminal_nodes[indices_to_check_if_downslopes_are_minimum[downslope_is_minimum]]

    terminal_nodes[indices] = values
    indices_in_terminal[indices] = True

    return len(values), terminal_nodes


def get_indices_leading_to_endpoints(endpoints):
    """
    Returns a list of the indices of all nodes ending up in each endpoint, as well as which endpoint it is
    :param endpoints: All nodes which are endpoints for other nodes
    :return unique: Indices of all endpoints in sorted order
    :return indices_to_endpoints: List of arrays where each array is the indices of all nodes going to that node
    """

    unique, counts = np.unique(endpoints, return_counts=True)
    sorted_indices = np.argsort(endpoints)
    indices_to_endpoints = np.split(sorted_indices, np.cumsum(counts))[0:-1]
    indices_to_endpoints_dict = dict(zip(unique, indices_to_endpoints))

    return indices_to_endpoints_dict


def combine_all_minimums_set(minimum_indices, min_neighbors):
    # This method is not used
    """
    The method will combine all the minimums in a landscape into larger local minimums. It will return a dictionary
    with the key being the region number, and the value being all indices in the region. The method will be used as
    part of the process of finding the watersheds.
    :param minimum_indices: Indices of all local minimums in the landscape.
    :param min_neighbors: The neighbors of all local minimums.
    :return watersheds: Dictionary with region number as key, and value being all indices in the region.
    """

    min_neighbors.update((index, neighbors.intersection(minimum_indices)) for index, neighbors in min_neighbors.items())
    watersheds = {}

    while minimum_indices:  # while more local minimums not assigned to a region
        loc_min = minimum_indices.pop()
        new_watershed = {loc_min}  # must have unique elements
        temp = {loc_min}

        while temp:  # while a region is not finished
            index = temp.pop()
            nbrs_of_loc_min = set(min_neighbors[index])
            new_minimums = nbrs_of_loc_min.difference(new_watershed)
            temp.update(new_minimums)
            new_watershed.update(new_minimums)
            minimum_indices = minimum_indices.difference(new_minimums)

        watersheds[len(watersheds)] = new_watershed

    return watersheds


def get_nodes_in_watersheds_set(watersheds, indices_to_endpoints):
    # This method is not used
    """
    The function returns a dictionary with key being the watershed number, and value being a set of all indices that
    terminate in the watershed. The length of nodes_in_watershed will be the number of watersheds in the landscape.
    :param watersheds: Dictionary with key being watershed number, and value being all minimum indices in the watershed.
    :param indices_to_endpoints: Dictionary where the key is the index of the local minimum, and the value is a set of
    all indices terminating in the local minimum.
    :return nodes_in_watershed: Dictionary with key being the watershed number, and value being a set of all indices
    that terminate in the watershed.
    """

    nodes_in_watershed = {}

    for key, value in watersheds.iteritems():
        temp_all_nodes = set()
        for val in value:
            temp_all_nodes = temp_all_nodes.union(indices_to_endpoints[val])

        nodes_in_watershed[key] = temp_all_nodes

    return nodes_in_watershed


def get_minimums_in_watersheds(minimum_indices, num_of_cols, num_of_rows):
    """
    Returns a list of sets where each set is a collection of all minimums in each watershed.
    :param minimum_indices: The indices of the minimums in the landscape.
    :param num_of_cols: Number of columns in the 2d-grid.
    :param num_of_rows: Number of rows in the 2d-grid.
    :return minimums_in_watershed: All minimums in each watershed
    """

    neighbors = util.get_neighbors_for_indices_array(minimum_indices, num_of_cols, num_of_rows)
    neighbor_is_a_min_bool = np.in1d(neighbors, minimum_indices)
    neighbor_is_a_min_bool = neighbor_is_a_min_bool.reshape(len(minimum_indices), 8)
    min_neighbors = np.multiply(neighbors, neighbor_is_a_min_bool).tolist()

    for i in range(len(minimum_indices)):
        min_neighbors[i] = [value for value in min_neighbors[i] if value != 0]
        min_neighbors[i].append(minimum_indices[i])

    min_neighbors = filter(None, min_neighbors)  # Remove the lists without neighbors being minimums

    graph = networkx.Graph()
    for l in min_neighbors:
        graph.add_nodes_from(l)
        edges = zip(l[:-1], l[1:])
        graph.add_edges_from(edges)

    minimums_in_watershed = networkx.connected_components(graph)
    #networkx.draw(graph)
    #plt.show()

    return minimums_in_watershed


def get_nodes_in_watersheds(endpoints, combined_minimums):
    """
    Returns a list where each element in the list is a watershed with its indices in an array.
    :param endpoints: The endpoints for each node in the landscape.
    :param combined_minimums: A list where each element is a set
    containing the indices of the minimums in the watershed.
    :return watersheds: All watersheds and their indices in each of them.
    """

    combined_minimums = [np.array(list(comb)) for comb in combined_minimums]
    dictionary_endpoints = get_indices_leading_to_endpoints(endpoints)

    watersheds = []

    for i in range(len(combined_minimums)):
        if len(combined_minimums[i]) == 1:
            watersheds.append(dictionary_endpoints[list(combined_minimums[i])[0]])
        else:
            ws = np.concatenate(list((dictionary_endpoints[i] for i in combined_minimums[i])))
            watersheds.append(ws)

    return watersheds


def get_watersheds(heights, num_of_cols, num_of_rows):
    """
    Returns all watersheds in an area given the heights of the landscape
    :param heights: z-coordinate for all indices
    :param num_of_cols: Number of grid points in x-direction
    :param num_of_rows: Number of grid points in y-direction
    :return nodes_in_watersheds: List of arrays where each array is a watershed and its nodes
    """

    downslope_neighbors = util.get_downslope_indices(num_of_cols, num_of_rows, heights)
    endpoints = get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)
    minimum_indices = np.where(downslope_neighbors == -1)[0]

    minimums_in_watersheds = sorted(get_minimums_in_watersheds(minimum_indices, num_of_cols, num_of_rows))
    nodes_in_watersheds = get_nodes_in_watersheds(endpoints, minimums_in_watersheds)

    return nodes_in_watersheds


def get_boundary_nodes_in_watersheds(watersheds, num_of_cols, num_of_rows):
    """
    Return the boundary nodes of the watershed, i.e. nodes bordering to other watersheds or to the landscape border.
    :param watersheds: Nodes of each watershed.
    :param num_of_cols: Number of nodes in the x-direction.
    :param num_of_rows: Number of nodes in the y-direction.
    :return boundary_nodes: The boundary nodes for each watershed.
    """

    boundary_nodes = []

    for watershed in watersheds:
        neighbors_for_watershed = util.get_neighbors_for_indices_array(watershed, num_of_cols, num_of_rows)
        neighbors_for_watershed_1d = np.concatenate(neighbors_for_watershed)
        not_in_watershed = np.in1d(neighbors_for_watershed_1d, watershed, invert=True)

        split_neighbors = np.split(neighbors_for_watershed_1d, len(watershed))
        split_boolean = np.split(not_in_watershed, len(watershed))

        watershed = np.sort(watershed)

        split_neighbors = [np.asarray([i for i in split if i != -1]) for split in split_neighbors]
        len_splits = [len(i) for i in split_neighbors]
        split_boolean = [split_boolean[i][:len_splits[i]] for i in range(len(split_boolean))]
        foreign_neighbors = [split_neighbors[i][split_boolean[i]] for i in range(len(split_neighbors))]
        boundary_indices = np.asarray([watershed[i] for i in range(len(foreign_neighbors))
                                       if len(foreign_neighbors[i]) != 0])
        landscape_boundary_nodes = util.are_boundary_nodes_bool(watershed, num_of_cols, num_of_rows)
        whole_boundary = np.unique(np.concatenate((boundary_indices, watershed[landscape_boundary_nodes])))
        boundary_nodes.append(whole_boundary)

    return boundary_nodes


def get_spill_points(boundary_nodes_in_watersheds, heights):

    heights_for_indices = [heights[bnd_nodes] for bnd_nodes in boundary_nodes_in_watersheds]
    spill_points = [boundary_nodes_in_watersheds[i][np.argmin(heights_for_indices[i])] for i
                    in range(len(heights_for_indices))]

    return np.asarray(spill_points)


def get_watershed_array(watersheds, number_of_nodes):
    """
    Returns the watershed index for every node.
    :param watersheds: List of all watersheds.
    :return watershed_array: Array of length N. Each index represents the watershed number for the node.
    """

    mapping_nodes_to_watersheds = np.empty(number_of_nodes, dtype=int)

    for i in range(len(watersheds)):
        mapping_nodes_to_watersheds[watersheds[i]] = i

    return mapping_nodes_to_watersheds


def get_downslope_neighbors_for_spill_points(spill_points, heights, watersheds, num_of_cols, num_of_rows):

    # Spill points at the boundary spills to the watershed it is located in.
    out_flow = np.empty(len(spill_points), dtype=int)
    spill_points_at_boundary = util.are_boundary_nodes_bool(spill_points, num_of_cols, num_of_rows)
    out_flow[spill_points_at_boundary] = spill_points[spill_points_at_boundary]

    # Get all interior spill points and their neighbors
    interior_indices = np.where(spill_points_at_boundary == False)[0]
    spill_points_interior = np.setdiff1d(spill_points, spill_points[spill_points_at_boundary])
    neighbors_of_spill_points = util.get_neighbors_for_interior_indices(spill_points_interior, num_of_cols)

    heights_of_spill_points = np.transpose(np.tile(heights[spill_points_interior], (8, 1)))

    # Calculating the derivatives of all neighbors
    delta_z = heights_of_spill_points - heights[neighbors_of_spill_points]
    delta_x = np.array([math.sqrt(200), 10, math.sqrt(200), 10, 10, math.sqrt(200), 10, math.sqrt(200)])
    derivatives = np.divide(delta_z, delta_x)

    mapping_nodes_to_watersheds = get_watershed_array(watersheds, num_of_cols * num_of_rows)
    in_flow = np.ones(len(spill_points), dtype=int) * -1

    for i in range(len(spill_points_interior)):  # For each interior spill point, find the node it is spilling to
        ws = watersheds[interior_indices[i]]
        spill_point_neighbors = neighbors_of_spill_points[i]
        derivatives_of_neighbors = derivatives[i]
        foreign_neighbors = np.setdiff1d(spill_point_neighbors, ws, assume_unique=True)  # Remove neighbors in the ws
        indices_of_foreign_neighbors = np.in1d(spill_point_neighbors, foreign_neighbors).nonzero()[0]
        foreign_derivatives = np.argmax(derivatives_of_neighbors[indices_of_foreign_neighbors])  # Find nbr in another
        # ws that will be the downslope neighbor
        downslope_foreign_neighbor = indices_of_foreign_neighbors[foreign_derivatives]
        flowing_to_ws = mapping_nodes_to_watersheds[spill_point_neighbors[downslope_foreign_neighbor]]
        in_flow[flowing_to_ws] = spill_points_interior[i]
        out_flow[interior_indices[i]] = spill_point_neighbors[downslope_foreign_neighbor]

    in_flow[np.where(in_flow == -1)[0]] = spill_points[np.where(in_flow == -1)[0]]

    return out_flow, in_flow

"""
def merge_indices_of_watersheds_using_spill_points(watersheds, downslope_neighbors, in_flow_indices, number_of_nodes):
    :param watersheds: List of all watersheds in the area.
    :param downslope_neighbors: The node each spill point is spilling to.
    :param number_of_nodes: Number of nodes in the grid.
    :return merged_indices_of_watersheds: List of arrays. Each array contains the indices of the watersheds that
    should be combined if doing spill point analysis.

    mapping_watershed_nodes = get_watershed_array(watersheds, number_of_nodes)
    out_flow = mapping_watershed_nodes[downslope_neighbors]
    in_flow = mapping_watershed_nodes[in_flow_indices]
    new_watershed_number = np.ones(len(watersheds), dtype=int) * -1
    new_watershed_counter = 0

    for i in range(len(watersheds)):
        current_ws = i
        next_ws = out_flow[current_ws]  # Spilling from current_ws to next_ws
        river = {current_ws}
        found_river = False
        while current_ws != next_ws and next_ws not in river:  # Following river downstream
            if new_watershed_number[next_ws] != -1:  # Found existing river
                new_watershed_number[list(river)] = new_watershed_number[next_ws]
                river = {i}
                found_river = True
                break
            else:
                river.add(next_ws)
                current_ws = next_ws
                next_ws = out_flow[current_ws]

        current_ws = i
        prev_ws = in_flow[i]
        while current_ws != prev_ws and prev_ws not in river:  # Following river upstream
            if new_watershed_number[prev_ws] != -1:
                new_watershed_number[list(river)] = new_watershed_number[prev_ws]
                found_river = True
                river = {}
                break
            else:
                river.add(prev_ws)
                current_ws = prev_ws
                prev_ws = in_flow[current_ws]

        if not found_river:
            new_watershed_number[list(river)] = new_watershed_counter
            new_watershed_counter += 1

    indices = np.arange(0, len(watersheds), 1)
    unique, counts = np.unique(new_watershed_number, return_counts=True)
    merged_indices_of_watersheds = np.column_stack((indices, new_watershed_number))
    p = merged_indices_of_watersheds[np.argsort(merged_indices_of_watersheds[:, 1])]

    new_watershed_indices = np.split(p[:, 0], np.cumsum(counts))[0:-1]
    print new_watershed_indices

    return new_watershed_indices
"""


def merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow_indices, out_flow_indices):
    """
    Combines the watersheds with connecting rivers.
    :param watersheds: The nodes of all watersheds.
    :param number_of_nodes: Total number of nodes in the landscape.
    :param in_flow_indices: Index of node leading into the watershed. Equal to spill point node if no river is
    leading into it.
    :param out_flow_indices: Index of the node the watershed is flowing to.
    :return merged_indices: The indices of the watersheds that are connected.
    """

    mapping_watershed_nodes = get_watershed_array(watersheds, number_of_nodes)
    in_flow = mapping_watershed_nodes[in_flow_indices]
    out_flow = mapping_watershed_nodes[out_flow_indices]

    ws_graph = networkx.Graph()
    ws_graph.add_nodes_from(np.arange(0, len(watersheds), 1))

    for i in range(len(in_flow)):
        ws_graph.add_edge(in_flow[i], i)
        ws_graph.add_edge(out_flow[i], i)

    merged_indices = sorted(networkx.connected_components(ws_graph))
    merged_indices = [np.array(list(el)) for el in merged_indices]

    return merged_indices


def merge_watersheds_using_merged_indices(watersheds, merged_watersheds):

    nodes_in_watersheds = []

    for i in range(len(merged_watersheds)):
        merged_ws = np.concatenate([watersheds[j] for j in merged_watersheds[i]])
        nodes_in_watersheds.append(merged_ws)

    return nodes_in_watersheds


def do_spill_point_analysis(heights, nx, ny):

    downslope_neighbors = util.get_downslope_indices(nx, ny, heights)
    endpoints = get_node_endpoints(nx, ny, downslope_neighbors)
    minimum_indices = np.where(downslope_neighbors == -1)[0]
    minimums_in_each_watershed = sorted(get_minimums_in_watersheds(minimum_indices, nx, ny))
    # indices_leading_to_endpoints = get_indices_leading_to_endpoints(endpoints)
    nodes_in_watersheds = get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)

    # Do spill point analysis
    boundary_nodes = get_boundary_nodes_in_watersheds(nodes_in_watersheds, nx, ny)
    spill_points = get_spill_points(boundary_nodes, heights)
    out_flow, in_flow = get_downslope_neighbors_for_spill_points(spill_points, heights, nodes_in_watersheds, nx, ny)
    merged_indices = merge_indices_of_watersheds_graph(nodes_in_watersheds, nx*ny, in_flow, out_flow)
    new_watersheds = merge_watersheds_using_merged_indices(nodes_in_watersheds, merged_indices)

    return new_watersheds
