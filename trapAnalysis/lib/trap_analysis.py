import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')
import numpy as np
import util
import networkx
import cPickle
saved_file_dir = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
import time


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
        self.arr = None

        step_size_x = geo_transform[1]
        step_size_y = geo_transform[5]
        unequal_step_size = (abs(step_size_x) != abs(step_size_y))
        if unequal_step_size:
            print 'The step size in the x- and y-direction is not equal'
            return
        self.step_size = step_size_x
        self.downslope_neighbors = None
        self.node_in_trap_index = None


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

    # THIS METHOD SUCKS. IT IS EXCRUCIATINGLY SLOW AND MUST BE CHANGED!!!!!!!!!!!!!!!!!!!! ITS NOT EVEN CORRECT...
    #unique, counts = np.unique(endpoints, return_counts=True)
    #indices_to_endpoints = [[unique[i]] for i in range(len(unique))]

    #for i in range(len(endpoints)):
    #    minimum = endpoints[i]
    #    min_index = unique.index(minimum)
    #    indices_to_endpoints[min_index].extend(i)

    # IF YOU THOUGHT THE ABOVE WAS SLOW, TRY THE METHOD BELOW....:(
    #for i in range(len(unique)):
    #    nr_of_indices_to_endpoint = counts[i]
    #    indices_to_endpoint = np.where(endpoints == unique[i])[0]
    #    indices_to_endpoints.append(indices_to_endpoint)

    # This however, this is the real deal
    unique, counts = np.unique(endpoints, return_counts=True)
    sorted_indices = np.argsort(endpoints)
    indices_to_endpoints = np.split(sorted_indices, np.cumsum(counts))[0:-1]

    return unique, indices_to_endpoints


def combine_all_minimums(minimum_indices, min_neighbors):
    # THIS IS A TEST USING SETS, NOT SURE IF THIS IS FINAL SOLUTION. IF IT IS, MUCH MUST BE REWRITTEN!
    # THIS FUNCTION IS OBSOLETE NOW, USING NUMPY INSTEAD
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


def combine_all_minimums_numpy(indices_minimums, num_of_cols, num_of_rows):
    """
    Returns a list of sets where each set is a cluster of minimums
    :param indices_minimums: The indices of the minimums in the landscape
    :param num_of_cols: Number of columns in the 2d-grid
    :param num_of_rows: Number of rows in the 2d-grid
    :return minimums_in_watershed: A combination of all minimums into larger minimums
    """

    neighbors = util.get_neighbors_for_indices_array(indices_minimums, num_of_cols, num_of_rows)
    neighbor_is_a_min_bool = np.in1d(neighbors, indices_minimums)
    neighbor_is_a_min_bool = neighbor_is_a_min_bool.reshape(len(indices_minimums), 8)
    min_neighbors = np.multiply(neighbors, neighbor_is_a_min_bool).tolist()

    for i in range(len(indices_minimums)):
        min_neighbors[i] = [value for value in min_neighbors[i] if value != 0]
        min_neighbors[i].append(indices_minimums[i])

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
    Returns all watersheds and the nodes within them
    :param endpoints: If you follow the downslope until you hit a minimum, that's the node's endpoint
    :param combined_minimums: A list of sets, where each set is a cluster of minimums
    :return nodes_in_watersheds: All watersheds and their corresponding nodes
    """

    nodes_in_watersheds = [None for i in range(len(combined_minimums))]
    combined_minimums = [np.array(list(comb)) for comb in combined_minimums]
    minimums, indices_leading_to_endpoints = get_indices_leading_to_endpoints(endpoints)
    nr_of_minimums_in_watersheds = [len(element) for element in combined_minimums]
    if nr_of_minimums_in_watersheds

        #if lst[0] in patterns: lst[0] = ''
        # http://stackoverflow.com/questions/10291997/how-can-i-do-assignments-in-a-list-comprehension
    [nodes_in_watersheds[i] = indices_leading_to_endpoints[2] for i in range(len(combined_minimums)) if nr_of_minimums_in_watersheds[i] == 1]


    min_index = np.where(minimums == combined_minimums[i])[0]
    nodes_in_watersheds.append(indices_leading_to_endpoints[min_index])

    # THIS IS TOO SLOW, MUST DO EVERYONE SIMULTANEOUSLY
    """
    nodes_in_watersheds = []
    minimums, indices_leading_to_endpoints = get_indices_leading_to_endpoints(endpoints)
    combined_minimums = [np.array(list(comb)) for comb in combined_minimums]    print len(combined_minimums)
    print 'Heeeeeeeeeeeeeeeere'
    start = time.time()
    for i in range(100):
        if len(combined_minimums[i]) == 1:
            min_index = np.where(minimums == combined_minimums[i])[0]
            nodes_in_watersheds.append(indices_leading_to_endpoints[min_index])
        else:
            indices_in_unique = np.where(np.in1d(minimums, combined_minimums[i]) == True)[0]
            ws = np.concatenate(list((indices_leading_to_endpoints[i] for i in indices_in_unique)))
            nodes_in_watersheds.append(ws)
    end = time.time()
    print 'Found all nodes in the watersheds in : ', end-start, ' seconds.'
    """

    return nodes_in_watersheds


def get_watersheds(heights, num_of_cols, num_of_rows):
    """
    Returns all watersheds in an area given the heights of the landscape
    :param heights: z-coordinate for all indices
    :param num_of_cols: Number of grid points in x-direction
    :param num_of_rows: Number of grid points in y-direction
    :return nodes_in_watersheds: List of lists where each list is a watershed and its nodes
    """

    downslope_neighbors = util.get_downslope_indices(num_of_cols, num_of_rows, heights)
    endpoints = get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)
    minimum_indices = np.where(downslope_neighbors == -1)[0]

    minimums_in_each_watershed = sorted(combine_all_minimums_numpy(minimum_indices, num_of_cols, num_of_rows))
    nodes_in_watersheds = get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)

    return nodes_in_watersheds


def get_watersheds_using_saved_files():
    """
    Does the save as get_watersheds, but using pickled and save data so it doesn't have to redo everything
    :return nodes_in_watersheds: A list of sets, where each set is a watershed with all its nodes
    """

    endpoints = np.load(saved_file_dir + 'endpoints.npy')
    minimums_in_each_watershed = cPickle.load(open(saved_file_dir + 'minimumsInEachWatershed.p', 'rb'))

    nodes_in_watersheds = get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)

    return nodes_in_watersheds
