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
    # networkx.draw(graph)
    # plt.show()

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
    :return boundary_nodes: The boundary nodes for each watershed. List of arrays.
    """

    boundary_nodes = []

    for watershed in watersheds:
        watershed = np.sort(watershed)
        neighbors_for_watershed = util.get_neighbors_for_indices_array(watershed, num_of_cols, num_of_rows)
        neighbors_for_watershed_1d = np.concatenate(neighbors_for_watershed)
        not_in_watershed_arr = np.in1d(neighbors_for_watershed_1d, watershed, invert=True)
        not_in_watershed = np.split(not_in_watershed_arr, len(watershed))

        bnd_nodes = np.asarray([watershed[i] for i in range(len(neighbors_for_watershed))
                                if len(neighbors_for_watershed[i][not_in_watershed[i]]) != 0])
        boundary_nodes.append(bnd_nodes)

    return boundary_nodes


def get_boundary_pairs_in_watersheds(watersheds, num_of_cols, num_of_rows):
    """
    Returns a list of arrays, where each array holds all boundary pairs for a watershed.
    :param watersheds: All watersheds.
    :param num_of_cols: Number of nodes in the x-direction.
    :param num_of_rows: Number of nodes in the y-direction.
    :return boundary_pairs: All boundary pairs in the landscape.
    """

    boundary_pairs = []  # List of arrays where the arrays have tuples

    for watershed in watersheds:
        watershed = np.sort(watershed)
        neighbors_for_watershed = util.get_neighbors_for_indices_array(watershed, num_of_cols, num_of_rows)
        neighbors_for_watershed_1d = np.concatenate(neighbors_for_watershed)
        not_in_watershed_arr = np.in1d(neighbors_for_watershed_1d, watershed, invert=True)
        not_in_watershed = np.split(not_in_watershed_arr, len(watershed))

        ext_nbrs = [(watershed[i], neighbors_for_watershed[i][not_in_watershed[i]]
                    [neighbors_for_watershed[i][not_in_watershed[i]] != -1])
                    for i in range(len(neighbors_for_watershed))
                    if len(neighbors_for_watershed[i][not_in_watershed[i]]
                    [neighbors_for_watershed[i][not_in_watershed[i]] != -1]) > 0]
        ext_bordering_nodes = [el[1] for el in ext_nbrs]
        repeat_nr = np.array([len(el[1]) for el in ext_nbrs])
        bordering_nodes = np.array([el[0] for el in ext_nbrs])
        nr_of_pairs = np.sum(repeat_nr)
        pair_arr = np.empty((2, nr_of_pairs), dtype=int)
        pair_arr[0, :] = np.repeat(bordering_nodes, repeat_nr)
        pair_arr[1, :] = np.concatenate(ext_bordering_nodes)
        boundary_pairs.append(pair_arr)

    return boundary_pairs


def get_min_height_of_max_of_all_pairs(boundary_pairs, heights):
    """
    Returns an array with the minimum height of the maximum value of all boundary pairs, essentially some kind of spill
    point.
    :param boundary_pairs: All boundary pairs for each watershed.
    :param heights: The heights of all nodes in the landscape.
    :return min_of_max_in_each_watershed: An array holding the spill point heights for each watershed.
    """

    max_heights_of_pairs = [np.max(heights[arr], axis=0) for arr in boundary_pairs]
    min_of_max_in_each_watershed = [np.min(arr) for arr in max_heights_of_pairs]

    return min_of_max_in_each_watershed


def get_spill_pair_indices(max_heights_of_pairs, min_of_max_in_each_watershed):
    """
    Returns a list of arrays where each array represents the indices of each pair that can be the spill point.
    :param max_heights_of_pairs: The max value of the pair.
    :param min_of_max_in_each_watershed: The minimum value of the max heights of pairs
    :return spill_pair_indices: The indices of the pairs which can be spill points.
    """

    spill_pair_indices = [np.nonzero(max_heights_of_pairs[i] == min_of_max_in_each_watershed[i])[0] for
                          i in range(len(min_of_max_in_each_watershed))]

    return spill_pair_indices


def get_steepest_spill_pair(boundary_pairs, spill_pairs):

    steepest_pairs = [boundary_pairs[i][:, spill_pairs[i][0]] for i in range(len(spill_pairs))]

    return steepest_pairs


def merge_watersheds_based_on_steepest_pairs(steepest_pairs, watersheds, nx, ny):

    ws_graph = networkx.Graph()
    ws_graph.add_nodes_from(np.arange(0, len(watersheds), 1))

    map_node_to_watersheds = map_nodes_to_watersheds(watersheds, nx * ny)

    for i in range(len(steepest_pairs)):
        from_node = steepest_pairs[i][0]
        to_node = steepest_pairs[i][1]
        #from_ws = map_node_to_watersheds[from_node]
        to_ws = map_node_to_watersheds[to_node]
        ws_graph.add_edge(i, to_ws)

    merged_indices = sorted(networkx.connected_components(ws_graph))
    merged_indices = [np.array(list(el)) for el in merged_indices]

    return merged_indices


def map_nodes_to_watersheds(watersheds, number_of_nodes):
    """
    Returns the watershed index for every node.
    :param watersheds: List of all watersheds.
    :return watershed_array: Array of length N. Each index represents the watershed number for the node.
    """

    mapping_nodes_to_watersheds = np.empty(number_of_nodes, dtype=int)

    for i in range(len(watersheds)):
        mapping_nodes_to_watersheds[watersheds[i]] = i

    return mapping_nodes_to_watersheds


def get_external_nbrs_dict_for_watershed(watershed_nr, nbrs_der_dict, watersheds, nx, ny):
    """
    Returns a dictionary for all boundary nodes with external neighbors for a watershed.
    :param watershed_nr: The watershed in question.
    :param nbrs_der_dict: Dictionary with neighbors and their derivatives for all nodes.
    :param watersheds: All watersheds.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :return external_dict: Dictionary with neighbors and their derivatives for all boundary nodes with external nbrs.
    """

    boundary_nodes = get_boundary_nodes_in_watersheds(watersheds, nx, ny)
    mapping_nodes_to_watersheds = map_nodes_to_watersheds(watersheds, nx * ny)
    boundary = boundary_nodes[watershed_nr]
    external_dict = {}

    for b in boundary:
        nbrs = nbrs_der_dict[b][0]
        external_nbrs = mapping_nodes_to_watersheds[nbrs] != watershed_nr
        if len(nbrs_der_dict[b][0][external_nbrs]) != 0:
            external_dict[b] = (nbrs_der_dict[b][0][external_nbrs], nbrs_der_dict[b][1][external_nbrs])

    return external_dict


def get_all_boundary_pairs_for_watershed(external_dict):
    """
    Returns all boundary pairs for the watershed.
    :param external_dict: Dictionary with all neighbors and derivatives for all boundary points with neighbors in
    other watersheds.
    :return boundary_pairs: List with tuples, where each tuple is a boundary pair.
    """

    boundary_pairs = []
    for key, value in external_dict.iteritems():
        nbrs = value[0]
        pair = [(key, nbr) for nbr in nbrs]
        boundary_pairs.extend(pair)

    return boundary_pairs


def get_lowest_height_on_landscape_boundary_for_watershed(watershed, heights, nx, ny):
    """
    If the watershed has nodes on the landscape boundary it will return the lowest height. If not, it returns -1.
    :param watershed: The watershed in question.
    :param heights: The heights of all nodes.
    :param nx: Number of nodes in the x-direction.
    :param ny: Number of nodes in the y-direction.
    :return min_on_landscape_boundary: A number if it exists, -1 otherwise.
    """

    landscape_boundary_nodes_in_ws = watershed[util.are_boundary_nodes_bool(watershed, nx, ny)]
    if len(landscape_boundary_nodes_in_ws) > 0:
        min_on_landscape_boundary = min(heights[landscape_boundary_nodes_in_ws])
    else:
        min_on_landscape_boundary = -1

    return min_on_landscape_boundary


def get_lowest_landscape_boundary_for_watersheds(watersheds, heights, nx, ny):
    """
    Return an array with the lowest landscape boundary height for the watersheds. If the watershed doesn't have a
    landscape boundary, that number will be -1.
    :param watersheds: List of all watersheds.
    :param heights: The heights of all nodes.
    :param nx: Number of nodes in the x-direction.
    :param ny: Number of nodes in the y-direction.
    :return lowest_landscape_boundary_for_ws: The lowest value the landscape boundary attains in the watershed. If there
    is no such value, it is set to -1.
    """

    boundary_nodes = util.get_boundary_indices(nx, ny)
    mapping_nodes_to_watersheds = map_nodes_to_watersheds(watersheds, nx * ny)
    map_boundary_nodes_watershed = mapping_nodes_to_watersheds[boundary_nodes]
    indices_of_ws_with_boundary, nr_of_boundary_nodes_in_watersheds = np.unique(map_boundary_nodes_watershed,
                                                                                return_counts=True)
    combined = np.vstack((map_boundary_nodes_watershed, boundary_nodes))
    combined_sorted_indices = np.argsort(combined[0])
    combined = combined[:, combined_sorted_indices]

    boundary_nodes_in_each_ws = np.split(combined[1, :], np.cumsum(nr_of_boundary_nodes_in_watersheds))[0:-1]
    min_of_boundary_on_watersheds = np.array([np.amin(heights[arr]) for arr in boundary_nodes_in_each_ws])
    lowest_landscape_boundary_for_ws = np.ones(len(watersheds), dtype=float) * -1

    lowest_landscape_boundary_for_ws[indices_of_ws_with_boundary] = min_of_boundary_on_watersheds

    return lowest_landscape_boundary_for_ws


def get_min_of_max_of_boundary_pairs(watershed, external_dict, boundary_pairs, heights, nx, ny):
    max_of_boundary_pair = []

    for i in range(len(boundary_pairs)):
        a = boundary_pairs[i][0]
        b = boundary_pairs[i][1]
        max_height = max(heights[a], heights[b])
        max_of_boundary_pair.append((max_height, boundary_pairs[i]))

    # If there are several tuples with the same max_height, can choose one with the greatest derivative. Not yet done.
    min_of_max = min(max_of_boundary_pair)

    return min_of_max


def merge_watersheds_using_boundary_pairs(watersheds, heights, nbrs_der_dict, nx, ny):
    ws_graph = networkx.Graph()
    ws_graph.add_nodes_from(np.arange(0, len(watersheds), 1))

    mapping_nodes_to_watersheds = map_nodes_to_watersheds(watersheds, nx * ny)

    for i in range(3):
        print i
        lowest_ws_landscape_boundary_node = get_lowest_height_on_landscape_boundary_for_watershed(watersheds[i],
                                                                                                  heights, nx, ny)
        external_dict = get_external_nbrs_dict_for_watershed(i, nbrs_der_dict, watersheds, nx, ny)
        boundary_pairs = get_all_boundary_pairs_for_watershed(external_dict)
        min_of_max = get_min_of_max_of_boundary_pairs(watersheds[i], external_dict, boundary_pairs, heights, nx, ny)
        can_merge = lowest_ws_landscape_boundary_node > min_of_max[0] or lowest_ws_landscape_boundary_node == -1
        if can_merge:
            from_node = min_of_max[1][0]
            to_node = min_of_max[1][1]
            to_ws = mapping_nodes_to_watersheds[to_node]
            ws_graph.add_edge(i, to_ws)  # Edge from watershed i to watershed to_ws

    merged_indices = sorted(networkx.connected_components(ws_graph))
    merged_indices = [np.array(list(el)) for el in merged_indices]

    return merged_indices


"""
def get_downslope_neighbors_for_spill_points(spill_points, heights, watersheds, num_of_cols, num_of_rows):
    # The downslope neighbor of the spill point must have a positive derivative, otherwise it won't spill anywhere

    # Spill points at the boundary spills to the watershed it is located in.
    out_flow = np.empty(len(spill_points), dtype=int)
    spill_points_at_boundary = util.are_boundary_nodes_bool(spill_points, num_of_cols, num_of_rows)
    out_flow[spill_points_at_boundary] = spill_points[spill_points_at_boundary]

    # Get all interior spill points and their neighbors
    interior_indices = np.where(spill_points_at_boundary == False)[0]
    spill_points_interior = spill_points[interior_indices]
    neighbors_of_spill_points = util.get_neighbors_for_interior_indices(spill_points_interior, num_of_cols)

    heights_of_spill_points = np.transpose(np.tile(heights[spill_points_interior], (8, 1)))

    # Calculating the derivatives of all neighbors
    delta_z = heights_of_spill_points - heights[neighbors_of_spill_points]
    delta_x = np.array([math.sqrt(200), 10, math.sqrt(200), 10, 10, math.sqrt(200), 10, math.sqrt(200)])
    derivatives = np.divide(delta_z, delta_x)

    mapping_nodes_to_watersheds = map_nodes_to_watersheds(watersheds, num_of_cols * num_of_rows)
    #in_flow = np.ones(len(spill_points), dtype=int) * -1

    for i in range(len(spill_points_interior)):  # For each interior spill point, find the node it is spilling to
        ws = watersheds[interior_indices[i]]
        spill_point_neighbors = neighbors_of_spill_points[i]
        derivatives_of_neighbors = derivatives[i]
        if np.max(derivatives_of_neighbors) < 0:
            out_flow[i] = spill_points_interior[i]
        else:
            foreign_neighbors = np.setdiff1d(spill_point_neighbors, ws, assume_unique=True)  # Remove neighbors in the ws
            indices_of_foreign_neighbors = np.in1d(spill_point_neighbors, foreign_neighbors).nonzero()[0]
            foreign_derivatives = np.argmax(derivatives_of_neighbors[indices_of_foreign_neighbors])  # Find nbr in another
            # ws that will be the downslope neighbor
            downslope_foreign_neighbor = indices_of_foreign_neighbors[foreign_derivatives]
            flowing_to_ws = mapping_nodes_to_watersheds[spill_point_neighbors[downslope_foreign_neighbor]]
            #in_flow[flowing_to_ws] = spill_points_interior[i]
            out_flow[interior_indices[i]] = spill_point_neighbors[downslope_foreign_neighbor]

    #in_flow[np.where(in_flow == -1)[0]] = spill_points[np.where(in_flow == -1)[0]]

    return out_flow#, in_flow

"""


def merge_watersheds_using_merged_indices(watersheds, merged_watersheds):
    nodes_in_watersheds = []

    for i in range(len(merged_watersheds)):
        merged_ws = np.concatenate([watersheds[j] for j in merged_watersheds[i]])
        nodes_in_watersheds.append(merged_ws)

    return nodes_in_watersheds
