import sys
path = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/'
sys.path.insert(0, path + 'lib')
sys.path.insert(0, path + 'util')
import trap_analysis
import numpy as np
import util
import networkx
from networkx.algorithms.components.connected import connected_components
import math
import compare_methods


def test_get_node_endpoints_one_min():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([1, 2, 5, 4, 8, 8, 7, 8, -1])
    result_terminal_nodes = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_node_endpoints_alternative(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_two_mins():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([-1, 0, 5, 0, 0, 8, 7, 8, -1])
    result_terminal_nodes = np.array([0, 0, 8, 0, 0, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_node_endpoints_alternative(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_one_min_rectangular():

    num_of_cols = 2
    num_of_rows = 3
    downslope_neighbors = np.array([3, 3, 5, 5, 5, -1])
    result_terminal_nodes = np.array([5, 5, 5, 5, 5, 5])

    terminal_nodes = trap_analysis.get_node_endpoints_alternative(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_four_mins_rectangular():

    num_of_cols = 4
    num_of_rows = 3
    downslope_neighbors = np.array([1, 6, 6, 7, 5, 6, -1, -1, 9, 10, -1, -1])
    result_terminal_nodes = np.array([6, 6, 6, 7, 6, 6, 6, 7, 10, 10, 10, 11])

    terminal_nodes = trap_analysis.get_node_endpoints_alternative(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_indices_leading_to_endpoints():

    endpoints = np.array([7, 7, 7, 5, 5, 5, 7, 7, 7, 22, 5, 5, 13, 13, 13, 22,
                          22, 23, 13, 13, 13, 22, 22, 23, 13, 13, 13, 28, 28, 29])
    # unique_endpoints = np.array([5, 7, 13, 22, 23, 28, 29])
    result_indices_to_endpoints = {5: np.array([3, 4, 5, 10, 11]),
                                   7: np.array([0, 1, 2, 6, 7, 8]),
                                   13: np.array([12, 13, 14, 18, 19, 20, 24, 25, 26]),
                                   22: np.array([9, 15, 16, 21, 22]),
                                   23: np.array([17, 23]),
                                   28: np.array([27, 28]),
                                   29: np.array([29])}

    indices_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)

    for minimum in indices_to_endpoints:
        indices_to_endpoints[minimum] = np.sort(indices_to_endpoints[minimum])

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(indices_to_endpoints,
                                                                                 result_indices_to_endpoints)

    assert are_equal


def test_combine_all_minimums_set():

    minimum_indices = {5, 7, 13, 22, 23, 28, 29}
    neighbors = {5: {4, 10, 11}, 7: {0, 1, 2, 6, 8, 12, 13, 14}, 13: {6, 7, 8, 12, 14, 18, 19, 20},
                 22: {15, 16, 17, 21, 23, 27, 28, 29}, 23: {16, 17, 22, 28, 29},
                 28: {21, 22, 23, 27, 29}, 29: {22, 23, 28}}
    result_watersheds = {0: {5}, 1: {7, 13}, 2: {22, 23, 28, 29}}

    watersheds = trap_analysis.combine_all_minimums_set(minimum_indices, neighbors)

    assert watersheds == result_watersheds


def test_get_nodes_in_watersheds_set():

    watersheds = {0: {5}, 1: {7, 13}, 2: {22, 23, 28, 29}}
    indices_to_endpoints = {5: {3, 4, 5, 10, 11}, 7: {0, 1, 2, 6, 7, 8},
                            13: {12, 13, 14, 18, 19, 20, 24, 25, 26},
                            22: {9, 15, 16, 21, 22}, 23: {17, 23},
                            28: {27, 28}, 29: {29}}
    result_nodes_in_watershed = {0: {3, 4, 5, 10, 11}, 1: {0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26},
                                 2: {9, 15, 16, 17, 21, 22, 23, 27, 28, 29}}

    nodes_in_watershed = trap_analysis.get_nodes_in_watersheds_set(watersheds, indices_to_endpoints)

    assert nodes_in_watershed == result_nodes_in_watershed


def test_get_minimums_in_watersheds():

    num_of_cols = 6
    num_of_rows = 5
    minimum_indices = np.array([5, 7, 13, 22, 23, 28, 29])
    neighbors = np.array([[4, 10, 11, -1, -1, -1, -1, -1],
                         [0, 1, 2, 6, 8, 12, 13, 14],
                         [6, 7, 8, 12, 14, 18, 19, 20],
                         [15, 16, 17, 21, 23, 27, 28, 29],
                         [16, 17, 22, 28, 29, -1, -1, -1],
                         [21, 22, 23, 27, 29, -1, -1, -1],
                         [22, 23, 28, -1, -1, -1, -1, -1]])

    connections = [{5}, {7, 13}, {22, 23, 28, 29}]
    result_connections = trap_analysis.get_minimums_in_watersheds(minimum_indices, num_of_cols, num_of_rows)

    assert sorted(connections) == sorted(result_connections)


"""
def test_get_nodes_in_watersheds():

    num_of_cols = 6
    num_of_rows = 5
    total_nodes = num_of_cols * num_of_rows
    indices = np.array([5, 7, 13, 22, 23, 28, 29])
    neighbors = np.array([[4, 10, 11, -1, -1, -1, -1, -1],
                         [0, 1, 2, 6, 8, 12, 13, 14],
                         [6, 7, 8, 12, 14, 18, 19, 20],
                         [15, 16, 17, 21, 23, 27, 28, 29],
                         [16, 17, 22, 28, 29, -1, -1, -1],
                         [21, 22, 23, 27, 29, -1, -1, -1],
                         [22, 23, 28, -1, -1, -1, -1, -1]])
    endpoints = np.array([7, 7, 7, 5, 5, 5, 7, 7, 7, 22, 5, 5, 13, 13, 13, 22,
                          22, 23, 13, 13, 13, 22, 22, 23, 13, 13, 13, 28, 28, 29])
    combined_minimums = [{5}, {7, 13}, {22, 23, 28, 29}]
    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]

    result_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, combined_minimums)

    for i in range(len(result_watersheds)):
        result_watersheds[i] = np.sort(result_watersheds[i])

    are_equal = True

    print watersheds
    print result_watersheds

    if len(watersheds) != len(result_watersheds):
        are_equal = False
    else:
        for i in range(len(result_watersheds)):
            elements_not_equal = np.array_equal(watersheds[i], result_watersheds[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal
"""


def test_get_watersheds_basic():

    num_of_cols = 3
    num_of_rows = 3
    heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
    result_watersheds = [np.array([0, 1, 2, 3, 4, 6]), np.array([5, 7, 8])]

    watersheds = trap_analysis.get_watersheds(heights, num_of_cols, num_of_rows)

    # Sort the elements of each watershed as the compare method is more conservative.
    for i in range(len(watersheds)):
        watersheds[i] = np.sort(watersheds[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(watersheds, result_watersheds)

    assert are_equal


def test_get_watersheds_advanced():

    num_of_cols = 6
    num_of_rows = 5
    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    result_watersheds = [np.array([3, 4, 5, 10, 11]),
                         np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                         np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]

    watersheds = trap_analysis.get_watersheds(heights, num_of_cols, num_of_rows)

    # Sort the elements of each watershed as the compare method is more conservative.
    for i in range(len(watersheds)):
        watersheds[i] = np.sort(watersheds[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(watersheds, result_watersheds)

    assert are_equal


def test_get_nodes_in_watersheds():

    endpoints = np.array([7, 7, 7, 5, 5, 5, 7, 7, 7, 22, 5, 5, 13, 13, 13, 22, 22,
                          23, 13, 19, 19, 22, 22, 23, 25, 25, 25, 28, 28, 29])
    combined_minimums = [{5}, {7, 13, 19, 25}, {22, 23, 28, 29}]
    unique = np.array([5, 7, 13, 19, 22, 23, 25, 28, 29])
    counts = np.array([5, 6, 3, 2, 3, 5, 2, 2, 1])
    result_watersheds = [np.array([3, 4, 5, 10, 11]),
                         np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                         np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]

    watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, combined_minimums)

    # Sort the elements of each watershed as the compare method is more conservative.
    for i in range(len(watersheds)):
        watersheds[i] = np.sort(watersheds[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(watersheds, result_watersheds)

    assert are_equal


def test_get_boundary_nodes_in_watersheds():

    num_of_cols = 6
    num_of_rows = 5
    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]
    result_boundary_nodes = [np.array([3, 4, 5, 10, 11]),
                             np.array([0, 1, 2, 6, 8, 12, 14, 18, 20, 24, 25, 26]),
                             np.array([9, 15, 16, 17, 21, 23, 27, 28, 29])]

    boundary_nodes = trap_analysis.get_boundary_nodes_in_watersheds(watersheds, num_of_cols, num_of_rows)

    # Sort the elements of each boundary as the compare method is more conservative.
    for i in range(len(boundary_nodes)):
        boundary_nodes[i] = np.sort(boundary_nodes[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(boundary_nodes, result_boundary_nodes)

    assert are_equal


def test_get_boundary_nodes_in_watersheds_advanced():

    num_of_cols = 9
    num_of_rows = 9
    watersheds = [np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                            6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26]),
                  np.array([0, 36, 39, 30, 29, 28, 27, 22, 21, 20, 37, 18, 13, 19, 38,
                            1,  2, 11, 10, 9, 3,  4, 12, 5]),
                  np.array([77, 76, 75, 74, 73, 72, 68, 67, 50, 49, 64, 63, 45, 46,
                            47, 59, 58, 57, 56, 55, 54, 48, 65, 66])]

    result_boundary_nodes = [np.array([6, 7, 8, 14, 15, 17, 23, 26, 31, 32, 35, 40, 41, 42, 44, 51,
                                       53, 60, 62, 69, 71, 78, 79, 80]),
                             np.array([0, 1, 2, 3, 4, 5, 9, 13, 18, 21, 22, 27, 30, 36, 37, 38, 39]),
                             np.array([45, 46, 47, 48, 49, 50, 54, 59, 63, 68, 72, 73, 74, 75, 76, 77])]

    boundary_nodes = trap_analysis.get_boundary_nodes_in_watersheds(watersheds, num_of_cols, num_of_rows)

    # Sort the elements of each boundary as the compare method is more conservative.
    for i in range(len(boundary_nodes)):
        boundary_nodes[i] = np.sort(boundary_nodes[i])

    print boundary_nodes
    are_equal = compare_methods.compare_two_lists_of_arrays(boundary_nodes, result_boundary_nodes)

    assert are_equal


def test_get_boundary_pairs_in_watersheds():

    nx = 7
    ny = 7
    watersheds = [np.array([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22]),
                  np.array([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26, 27,
                            32, 33, 34, 39, 40, 41, 46, 47, 48]),
                  np.array([28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 45])]
    result_boundary_pairs = [np.array([np.array([2, 2, 9, 9, 9, 15, 16, 16, 16, 16, 21, 21, 22, 22, 22, 22]),
                                      np.array([3, 10, 3, 10, 17, 23, 10, 17, 23, 24, 28, 29, 23, 28, 29, 30])]),
                             np.array([np.array([3, 3, 10, 10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39, 39, 46, 46]),
                                      np.array([2, 9, 2, 9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38, 45, 38, 45])]),
                             np.array([np.array([28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38, 38, 45, 45]),
                                      np.array([21, 22, 21, 22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39, 46, 39, 46])])]

    boundary_pairs = trap_analysis.get_boundary_pairs_in_watersheds(watersheds, nx, ny)
    are_equal = compare_methods.compare_two_lists_of_arrays(boundary_pairs, result_boundary_pairs)

    assert are_equal


def test_get_boundary_pairs_in_watersheds_different_order():
    # Different order of the watersheds and the elements in the watersheds

    nx = 7
    ny = 7
    watersheds = [np.array([6, 5, 4, 3, 10, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26, 27,
                            32, 33, 34, 39, 40, 41, 46, 47, 48]),
                  np.array([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22]),
                  np.array([28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 45])]
    result_boundary_pairs = [np.array([np.array([3, 3, 10, 10, 10, 17, 17, 23, 23, 23, 23, 23, 23, 24, 24, 24, 25, 32, 32, 39, 39, 39, 46, 46]),
                                      np.array([2, 9, 2, 9, 16, 9, 16, 15, 16, 22, 29, 30, 31, 16, 30, 31, 31, 31, 38, 31, 38, 45, 38, 45])]),
                             np.array([np.array([2, 2, 9, 9, 9, 15, 16, 16, 16, 16, 21, 21, 22, 22, 22, 22]),
                                      np.array([3, 10, 3, 10, 17, 23, 10, 17, 23, 24, 28, 29, 23, 28, 29, 30])]),
                             np.array([np.array([28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 31, 31, 38, 38, 38, 45, 45]),
                                      np.array([21, 22, 21, 22, 23, 22, 23, 24, 23, 24, 25, 32, 39, 32, 39, 46, 39, 46])])]

    boundary_pairs = trap_analysis.get_boundary_pairs_in_watersheds(watersheds, nx, ny)
    are_equal = compare_methods.compare_two_lists_of_arrays(boundary_pairs, result_boundary_pairs)

    assert are_equal


def test_map_nodes_to_watersheds():

    number_of_nodes = 30
    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]
    result_watershed_indices = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0, 0,
                                         1, 1, 1, 2, 2, 2, 1, 1, 1,
                                         2, 2, 2, 1, 1, 1, 2, 2, 2])

    watershed_indices = trap_analysis.map_nodes_to_watersheds(watersheds, number_of_nodes)

    assert np.array_equal(result_watershed_indices, watershed_indices)


def test_get_external_nbrs_dict_for_watershed():

    nx = 3
    ny = 3
    heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
    watersheds = [np.array([0, 1, 2, 3, 4, 6]), np.array([5, 7, 8])]
    watershed_nr = 0

    a = 1/math.sqrt(200)
    b = 2/math.sqrt(200)
    total_dict = {0: (np.array([1, 3, 4]), np.array([-0.1, -0.1, -a])),
                  1: (np.array([0, 2, 3, 4, 5]), np.array([0.1, -0.1, 0, -0.1, -b])),
                  2: (np.array([1, 4, 5]), np.array([0.1, 0, -0.1])),
                  3: (np.array([0, 1, 4, 6, 7]), np.array([0.1, 0, -0.1, -0.1, -b])),
                  4: (np.array([0, 1, 2, 3, 5, 6, 7, 8]), np.array([b, 0.1, 0, 0.1, -0.1, 0, -0.1, a])),
                  5: (np.array([1, 2, 4, 7, 8]), np.array([b, 0.1, 0.1, 0, 0.2])),
                  6: (np.array([3, 4, 7]), np.array([0.1, 0, -0.1])),
                  7: (np.array([3, 4, 5, 6, 8]), np.array([b, 0.1, 0, 0.1, 0.2])),
                  8: (np.array([4, 5, 7]), np.array([-a, -0.2, -0.2]))}
    result_external_dict = {1: (np.array([5]), np.array([-b])),
                            2: (np.array([5]), np.array([-0.1])),
                            3: (np.array([7]), np.array([-b])),
                            4: (np.array([5, 7, 8]), np.array([-0.1, -0.1, a])),
                            6: (np.array([7]), np.array([-0.1]))}

    external_dict = trap_analysis.get_external_nbrs_dict_for_watershed(watershed_nr, total_dict, watersheds, nx, ny)

    print external_dict

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(
        external_dict, result_external_dict)

    assert are_equal


def test_get_all_boundary_pairs_for_watershed():

    a = 1/math.sqrt(200)
    b = 2/math.sqrt(200)
    external_dict = {1: (np.array([5]), np.array([-b])),
                     2: (np.array([5]), np.array([-0.1])),
                     3: (np.array([7]), np.array([-b])),
                     4: (np.array([5, 7, 8]), np.array([-0.1, -0.1, a])),
                     6: (np.array([7]), np.array([-0.1]))}
    result_boundary_pairs = [(2, 5), (1, 5), (4, 5), (4, 7), (4, 8), (3, 7), (6, 7)]

    boundary_pairs = trap_analysis.get_all_boundary_pairs_for_watershed(external_dict)

    # Sort both lists as the compare method is more conservative than necessary.
    result_boundary_pairs.sort()
    boundary_pairs.sort()

    are_equal = compare_methods.compare_two_lists_of_tuples(boundary_pairs, result_boundary_pairs)

    assert are_equal

"""
def test_get_all_boundary_pairs_for_watersheds():

    nx = 7
    ny = 7
    watersheds = [np.array([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22]),
                  np.array([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26, 27,
                            32, 33, 34, 39, 40, 41, 46, 47, 48]),
                  np.array([28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 45])]
    #boundary_nodes = [np.array([0, 1, 2, 7, 9, 14, 15, 16, 21, 22]),
    #                  np.array([3, 4, 5, 6, 10, 13, 17, 20, 23, 24, 25, 27, 32, 34, 39, 41, 46, 47, 48])]
    result_boundary_pairs = [(2, 3)]

    boundary_pairs = trap_analysis.get_all_boundary_pairs_for_watersheds(watersheds, boundary_nodes, nx, ny)

    result_boundary_pairs.sort()
    boundary_pairs.sort()

    are_equal = compare_methods.compare_two_lists_of_tuples(boundary_pairs, result_boundary_pairs)

    assert are_equal
"""

def test_get_min_of_max_of_boundary_pairs():

    nx = 3
    ny = 3
    a = 1/math.sqrt(200)
    b = 2/math.sqrt(200)
    heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
    watershed = np.array([0, 1, 2, 3, 4, 6])
    external_dict = {1: (np.array([5]), np.array([-b])),
                     2: (np.array([5]), np.array([-0.1])),
                     3: (np.array([7]), np.array([-b])),
                     4: (np.array([5, 7, 8]), np.array([-0.1, -0.1, a])),
                     6: (np.array([7]), np.array([-0.1]))}
    boundary_pairs = [(2, 5), (1, 5), (4, 5), (4, 7), (4, 8), (3, 7), (6, 7)]
    result_min_of_max = (2, (4, 8))

    min_of_max = trap_analysis.get_min_of_max_of_boundary_pairs(watershed, external_dict, boundary_pairs,
                                                                heights, nx, ny)
    are_equal = min_of_max == result_min_of_max

    assert are_equal


def test_get_lowest_height_on_landscape_boundary_for_watershed_basic():

    nx = 3
    ny = 3
    heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
    watershed = np.array([0, 1, 2, 3, 4, 6])
    result_lowest_height = 0

    lowest_height = trap_analysis.get_lowest_height_on_landscape_boundary_for_watershed(watershed, heights, nx, ny)

    assert lowest_height == result_lowest_height


def test_get_lowest_height_on_landscape_boundary_for_watershed_advanced():

    nx = 9
    ny = 9
    heights = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 4, 5, 8, 9, 9, 9, 10,
                        10, 4, 1, 6, 8, 7, 7, 7, 10, 10, 6, 6, 6, 8, 5, 5, 5, 10,
                        10, 8, 8, 8, 8, 4, 2, 4, 10, 10, 9, 9, 9, 9, 3, 3, 3, 10,
                        10, 6, 1, 1, 1, 5, 5, 5, 10, 10, 6, 1, 0, 1, 6, 6, 6, 10,
                        10, 10, 10, 10, 10, 10, 10, 10, 10])
    watersheds = [np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                            6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26]),
                  np.array([0, 36, 39, 30, 29, 28, 27, 22, 21, 20, 37, 18, 13, 19, 38,
                            1,  2, 11, 10, 9, 3,  4, 12, 5]),
                  np.array([77, 76, 75, 74, 73, 72, 68, 67, 50, 49, 64, 63, 45, 46,
                            47, 59, 58, 57, 56, 55, 54, 48, 65, 66])]
    watershed = np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                          6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26])
    result_lowest_height = 10

    lowest_height = trap_analysis.get_lowest_height_on_landscape_boundary_for_watershed(watershed, heights, nx, ny)

    assert lowest_height == result_lowest_height


def test_get_lowest_height_on_landscape_boundary_for_watershed_no_landscape_boundary():

    nx = 5
    ny = 5
    heights = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    watershed = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])
    result_lowest_height = -1

    lowest_height = trap_analysis.get_lowest_height_on_landscape_boundary_for_watershed(watershed, heights, nx, ny)

    assert lowest_height == result_lowest_height

"""
def test_get_lowest_landscape_boundary_for_watersheds():

    nx = 9
    ny = 9
    heights = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 4, 5, 8, 9, 9, 9, 10,
                        10, 4, 1, 6, 8, 7, 7, 7, 10, 10, 6, 6, 6, 8, 5, 5, 5, 10,
                        10, 8, 8, 8, 8, 4, 2, 4, 10, 10, 9, 9, 9, 9, 3, 3, 3, 10,
                        10, 6, 1, 1, 1, 5, 5, 5, 10, 10, 6, 1, 0, 1, 6, 6, 6, 10,
                        10, 10, 10, 10, 10, 10, 10, 10, 10])
    watersheds = [np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                            6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26]),
                  np.array([0, 36, 39, 30, 29, 28, 27, 22, 21, 20, 37, 18, 13, 19, 38,
                            1,  2, 11, 10, 9, 3,  4, 12, 5]),
                  np.array([77, 76, 75, 74, 73, 72, 68, 67, 50, 49, 64, 63, 45, 46,
                            47, 59, 58, 57, 56, 55, 54, 48, 65, 66])]
    watershed = np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                          6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26])
    result_lowest_heights = np.array([10, 10, 10])

    lowest_heights = trap_analysis.get_lowest_landscape_boundary_for_watersheds(watersheds, heights, nx, ny)

    assert lowest_heights == result_lowest_heights
"""


def test_get_lowest_landscape_boundary_for_watersheds():
    # Note: This is not a realistic example. The heights used in the test would not have resulted in these watersheds.

    nx = 4
    ny = 4
    watersheds = [np.array([0, 4, 8, 12, 13, 14, 15]),
                  np.array([1, 2]),
                  np.array([3, 7, 11]),
                  np.array([5, 6]),
                  np.array([9, 10])]
    heights = np.array([0, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8, 8, 7, 6, 5])

    result_lowest_landscape_boundary = np.array([0, 2, 1, -1, -1])

    lowest_landscape_boundary = trap_analysis.get_lowest_landscape_boundary_for_watersheds(watersheds, heights, nx, ny)
    are_equal = np.array_equal(lowest_landscape_boundary, result_lowest_landscape_boundary)

    assert are_equal


def test_get_lowest_landscape_boundary_for_watersheds_unordered():
    # Note: This is not a realistic example. The heights used in the test would not have resulted in these watersheds.
    # The ordering of the watershed is different here

    nx = 4
    ny = 4
    watersheds = [np.array([5, 6]),
                  np.array([3, 7, 11]),
                  np.array([9, 10]),
                  np.array([0, 4, 8, 12, 13, 14, 15]),
                  np.array([1, 2])]
    heights = np.array([0, 2, 3, 4, 4, 3, 2, 1, 5, 6, 7, 8, 8, 7, 6, 5])

    result_lowest_landscape_boundary = np.array([-1, 1, -1, 0, 2])

    lowest_landscape_boundary = trap_analysis.get_lowest_landscape_boundary_for_watersheds(watersheds, heights, nx, ny)
    are_equal = np.array_equal(lowest_landscape_boundary, result_lowest_landscape_boundary)

    assert are_equal


"""
def test_get_downslope_neighbors_for_spill_points():

    nx = 6
    ny = 5
    spill_points = np.array([5, 20, 23])
    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]

    result_downslope_neighbors = np.array([5, 27, 23])

    downslope_neighbors = trap_analysis.get_downslope_neighbors_for_spill_points(
            spill_points, heights, watersheds, nx, ny)

    assert np.array_equal(downslope_neighbors, result_downslope_neighbors)
"""

"""
def test_get_downslope_neighbors_for_spill_points_advanced():

    nx = 9
    ny = 9
    spill_points = np.array([42, 21, 50])
    heights = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 4, 5, 8, 9, 9, 9, 10,
                        10, 4, 1, 6, 8, 7, 7, 7, 10, 10, 6, 6, 6, 8, 5, 5, 5, 10,
                        10, 8, 8, 8, 8, 4, 2, 4, 10, 10, 9, 9, 9, 9, 3, 3, 3, 10,
                        10, 6, 1, 1, 1, 5, 5, 5, 10, 10, 6, 1, 0, 1, 6, 6, 6, 10,
                        10, 10, 10, 10, 10, 10, 10, 10, 10])
    watersheds = [np.array([79, 41, 42, 43, 69, 51, 52, 53, 71, 70, 60, 61, 62, 78, 44, 40, 34,
                            6, 7, 8, 14, 15, 16, 35, 23, 24, 25, 17, 80, 33, 32, 31, 26]),
                  np.array([0, 36, 39, 30, 29, 28, 27, 22, 21, 20, 37, 18, 13, 19, 38,
                            1,  2, 11, 10, 9, 3,  4, 12, 5]),
                  np.array([77, 76, 75, 74, 73, 72, 68, 67, 50, 49, 64, 63, 45, 46,
                            47, 59, 58, 57, 56, 55, 54, 48, 65, 66])]
    result_out_flow = np.array([42, 42, 31])

    out_flow = trap_analysis.get_downslope_neighbors_for_spill_points(spill_points, heights, watersheds,
                                                                                 nx, ny)
    assert np.array_equal(out_flow, result_out_flow)
"""

"""
def test_merge_watersheds_using_merged_indices():

    watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42]),
                  np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47]),
                  np.array([9, 10, 17, 18]),
                  np.array([11, 12]),
                  np.array([13, 14, 22]),
                  np.array([25, 33, 34]),
                  np.array([26, 27, 35]),
                  np.array([19, 20, 28, 36]),
                  np.array([21, 29, 30]),
                  np.array([37, 38])]
    merged_indices = [np.array([0, 2, 3, 5, 6, 7]), np.array([1, 4, 8, 9])]
    result_merged_watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42, 9, 10, 17, 18,
                                          11, 12, 25, 33, 34, 26, 27, 35, 19, 20, 28, 36]),
                                np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47, 13, 14, 22,
                                          21, 29, 30, 37, 38])]

    merged_watersheds = trap_analysis.merge_watersheds_using_merged_indices(watersheds, merged_indices)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_watersheds)):
        merged_watersheds[i] = np.sort(merged_watersheds[i])
        result_merged_watersheds[i] = np.sort(result_merged_watersheds[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_watersheds, result_merged_watersheds)

    assert are_equal

############################################################


def test_merge_indices_of_watersheds_graph():

    number_of_nodes = 48
    watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42]),
                  np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47]),
                  np.array([9, 10, 17, 18]),
                  np.array([11, 12]),
                  np.array([13, 14, 22]),
                  np.array([25, 33, 34]),
                  np.array([26, 27, 35]),
                  np.array([19, 20, 28, 36]),
                  np.array([21, 29, 30]),
                  np.array([37, 38])]
    in_flow = np.array([40, 15, 10, 10, 22, 26, 28, 12, 22, 29])
    out_flow = np.array([40, 15, 11, 20, 21, 24, 34, 27, 37, 46])
    result_merged_indices = [np.array([0, 2, 3, 5, 6, 7]), np.array([1, 4, 8, 9])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_with_loop():

    number_of_nodes = 48
    watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42]),
                  np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47]),
                  np.array([9, 10, 17, 18]),
                  np.array([11, 12]),
                  np.array([13, 14, 22]),
                  np.array([25, 33, 34]),
                  np.array([26, 27, 35]),
                  np.array([19, 20, 28, 36]),
                  np.array([21, 29, 30]),
                  np.array([37, 38])]
    in_flow = np.array([40, 15, 25, 10, 22, 26, 28, 12, 22, 29])
    out_flow = np.array([40, 15, 11, 20, 21, 17, 34, 27, 37, 46])
    result_merged_indices = [np.array([0]), np.array([1, 4, 8, 9]), np.array([2, 3, 5, 6, 7])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_upwards_river():

    number_of_nodes = 48
    watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42]),
                  np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47]),
                  np.array([9, 10, 17, 18]),
                  np.array([11, 12]),
                  np.array([13, 14, 22]),
                  np.array([25, 33, 34]),
                  np.array([26, 27, 35]),
                  np.array([19, 20, 28, 36]),
                  np.array([21, 29, 30]),
                  np.array([37, 38])]
    in_flow = np.array([40, 15, 25, 10, 30, 26, 28, 12, 37, 38])
    out_flow = np.array([40, 15, 11, 20, 7, 17, 34, 27, 22, 29])
    result_merged_indices = [np.array([0]), np.array([1, 4, 8, 9]), np.array([2, 3, 5, 6, 7])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_two_watersheds_spilling_in_same_ws():

    number_of_nodes = 48
    watersheds = [np.array([0, 1, 2, 3, 4, 5, 8, 16, 24, 32, 40, 41, 42]),
                  np.array([6, 7, 15, 23, 31, 39, 43, 44, 45, 46, 47]),
                  np.array([9, 10, 17, 18]),
                  np.array([11, 12]),
                  np.array([13, 14, 22]),
                  np.array([25, 33, 34]),
                  np.array([26, 27, 35]),
                  np.array([19, 20, 28, 36]),
                  np.array([21, 29, 30]),
                  np.array([37, 38])]
    in_flow = np.array([40, 15, 18, 12, 22, 26, 28, 12, 22, 29])
    out_flow = np.array([40, 15, 19, 20, 21, 24, 34, 27, 37, 46])
    result_merged_indices = [np.array([0, 2, 3, 5, 6, 7]), np.array([1, 4, 8, 9])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_already_combined():

    number_of_nodes = 20
    watersheds = [np.array([0, 1, 5, 6]),
                  np.array([2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19]),
                  np.array([10, 11, 15, 16])]

    in_flow = np.array([0, 19, 15])
    out_flow = np.array([0, 19, 15])
    result_merged_indices = [np.array([0]),
                             np.array([1]),
                             np.array([2])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_fill_upwards():

    number_of_nodes = 16
    watersheds = [np.array([0, 1, 4, 5]),
                  np.array([2, 3, 6, 7]),
                  np.array([8, 9, 12, 13]),
                  np.array([10, 11, 14, 15])]

    in_flow = np.array([9, 10, 9, 10])  # River_from
    out_flow = np.array([0, 3, 5, 6])  # River_to
    result_merged_indices = [np.array([0, 2]),
                             np.array([1, 3])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_fill_downwards():

    number_of_nodes = 16
    watersheds = [np.array([0, 1, 4, 5]),
                  np.array([2, 3, 6, 7]),
                  np.array([8, 9, 12, 13]),
                  np.array([10, 11, 14, 15])]

    in_flow = np.array([0, 3, 5, 6])
    out_flow = np.array([9, 10, 12, 15])
    result_merged_indices = [np.array([0, 2]),
                             np.array([1, 3])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_merge_indices_of_watersheds_graph_valley():

    number_of_nodes = 15
    watersheds = [np.array([0, 5, 6]),
                  np.array([1, 2]),
                  np.array([3, 4]),
                  np.array([7, 8]),
                  np.array([9, 14]),
                  np.array([10, 11]),
                  np.array([12, 13])]

    in_flow = np.array([1, 1, 8, 11, 9, 11, 9])  # River_from
    out_flow = np.array([5, 5, 4, 3, 13, 7, 13])  # River_to
    result_merged_indices = [np.array([0, 1]),
                             np.array([2, 3, 5]),
                             np.array([4, 6])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_graph(watersheds, number_of_nodes, in_flow, out_flow)

    # Sort both lists as the compare method is more conservative.
    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = compare_methods.compare_two_lists_of_arrays(merged_indices, result_merged_indices)

    assert are_equal


def test_do_spill_point_analysis():

    #heights = np.array([np.array([10, 10, 10, 10, 10, 10, 10, 10, 10]),
    #                    np.array([10, 3, 4, 5, 8, 9, 9, 9, 10]),
    #                    np.array([10, 4, 1, 6, 8, 7, 7, 7, 10]),
    #                    np.array([10, 6, 6, 6, 8, 5, 5, 5, 10]),
    #                    np.array([10, 8, 8, 8, 8, 4, 2, 4, 10]),
    #                    np.array([10, 9, 9, 9, 9, 3, 3, 3, 10]),
    #                    np.array([10, 6, 1, 1, 1, 5, 5, 5, 10]),
    #                    np.array([10, 6, 1, 0, 1, 6, 6, 6, 10]),
    #                    np.array([10, 10, 10, 10, 10, 10, 10, 10, 10])])

    heights = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 3, 4, 5, 8, 9, 9, 9, 10,
                        10, 4, 1, 6, 8, 7, 7, 7, 10, 10, 6, 6, 6, 8, 5, 5, 5, 10,
                        10, 8, 8, 8, 8, 4, 2, 4, 10, 10, 9, 9, 9, 9, 3, 3, 3, 10,
                        10, 6, 1, 1, 1, 5, 5, 5, 10, 10, 6, 1, 0, 1, 6, 6, 6, 10,
                        10, 10, 10, 10, 10, 10, 10, 10, 10])

    nx = 9
    ny = 9

    new_watersheds = trap_analysis.do_spill_point_analysis(heights, nx, ny)

    print new_watersheds

    assert new_watersheds is False
"""
