import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import trap_analysis
import numpy as np
import util
import networkx
from networkx.algorithms.components.connected import connected_components


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

    num_of_nodes_x = 6
    num_of_nodes_y = 5
    endpoints = np.array([7, 7, 7, 5, 5, 5, 7, 7, 7, 22, 5, 5, 13, 13, 13, 22,
                          22, 23, 13, 13, 13, 22, 22, 23, 13, 13, 13, 28, 28, 29])
    # unique_endpoints = np.array([5, 7, 13, 22, 23, 28, 29])
    result_indices = [np.array([3, 4, 5, 10, 11]),
                      np.array([0, 1, 2, 6, 7, 8]),
                      np.array([12, 13, 14, 18, 19, 20, 24, 25, 26]),
                      np.array([9, 15, 16, 21, 22]),
                      np.array([17, 23]),
                      np.array([27, 28]),
                      np.array([29])]

    unique, indices = trap_analysis.get_indices_leading_to_endpoints(endpoints)

    for i in range(len(indices)):
        indices[i] = np.sort(indices[i])

    are_equal = True

    if len(indices) != len(result_indices):
        are_equal = False
    else:
        for i in range(len(indices)):
            elements_not_equal = np.array_equal(indices[i], result_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_get_watersheds():

    minimum_indices = {5, 7, 13, 22, 23, 28, 29}
    neighbors = {5: {4, 10, 11}, 7: {0, 1, 2, 6, 8, 12, 13, 14}, 13: {6, 7, 8, 12, 14, 18, 19, 20},
                 22: {15, 16, 17, 21, 23, 27, 28, 29}, 23: {16, 17, 22, 28, 29},
                 28: {21, 22, 23, 27, 29}, 29: {22, 23, 28}}
    result_watersheds = {0: {5}, 1: {7, 13}, 2: {22, 23, 28, 29}}
    watersheds = trap_analysis.combine_all_minimums(minimum_indices, neighbors)

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


def test_combine_all_minimums_numpy():

    num_of_cols = 6
    num_of_rows = 5
    indices = np.array([5, 7, 13, 22, 23, 28, 29])
    neighbors = np.array([[4, 10, 11, -1, -1, -1, -1, -1],
                         [0, 1, 2, 6, 8, 12, 13, 14],
                         [6, 7, 8, 12, 14, 18, 19, 20],
                         [15, 16, 17, 21, 23, 27, 28, 29],
                         [16, 17, 22, 28, 29, -1, -1, -1],
                         [21, 22, 23, 27, 29, -1, -1, -1],
                         [22, 23, 28, -1, -1, -1, -1, -1]])

    connections = [{5}, {7, 13}, {22, 23, 28, 29}]
    result_connections = trap_analysis.combine_all_minimums_numpy(indices, num_of_cols, num_of_rows)

    assert sorted(connections) == sorted(result_connections)


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


def test_get_watersheds():

    num_of_cols = 6
    num_of_rows = 5
    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    nodes_in_watersheds = trap_analysis.get_watersheds(heights, num_of_cols, num_of_rows)

    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]

    for i in range(len(nodes_in_watersheds)):
        nodes_in_watersheds[i] = np.sort(nodes_in_watersheds[i])

    are_equal = True

    if len(watersheds) != len(nodes_in_watersheds):
        are_equal = False
    else:
        for i in range(len(nodes_in_watersheds)):
            elements_not_equal = np.array_equal(watersheds[i], nodes_in_watersheds[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(watersheds)):
        watersheds[i] = np.sort(watersheds[i])

    are_equal = True

    if len(watersheds) != len(result_watersheds):
        are_equal = False
    else:
        for i in range(len(watersheds)):
            elements_not_equal = np.array_equal(watersheds[i], result_watersheds[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal
