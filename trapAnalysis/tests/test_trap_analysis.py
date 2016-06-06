import sys
path = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/'
sys.path.insert(0, path + 'lib')
sys.path.insert(0, path + 'util')
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

    are_equal = True

    if len(indices_to_endpoints) != len(result_indices_to_endpoints):
        are_equal = False
    else:
        for key, value in indices_to_endpoints.iteritems():
            elements_not_equal = np.array_equal(value, indices_to_endpoints[key]) == False
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

    for i in range(len(boundary_nodes)):
        boundary_nodes[i] = np.sort(boundary_nodes[i])

    are_equal = True

    if len(boundary_nodes) != len(result_boundary_nodes):
        are_equal = False
    else:
        for i in range(len(watersheds)):
            elements_not_equal = np.array_equal(boundary_nodes[i], result_boundary_nodes[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_get_spill_points():

    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]
    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    boundary_nodes = [np.array([3, 4, 5, 10, 11]),
                      np.array([0, 1, 2, 6, 8, 12, 14, 18, 20, 24, 25, 26]),
                      np.array([9, 15, 16, 17, 21, 23, 27, 28, 29])]

    result_spill_points = np.array([5, 20, 23])

    spill_points = trap_analysis.get_spill_points(boundary_nodes, heights)

    assert np.array_equal(spill_points, result_spill_points)


def test_get_watershed_array():

    number_of_nodes = 30
    watersheds = [np.array([3, 4, 5, 10, 11]),
                  np.array([0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])]
    result_watershed_indices = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 0, 0,
                                         1, 1, 1, 2, 2, 2, 1, 1, 1,
                                         2, 2, 2, 1, 1, 1, 2, 2, 2])

    watershed_indices = trap_analysis.get_watershed_array(watersheds, number_of_nodes)

    assert np.array_equal(result_watershed_indices, watershed_indices)

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

    downslope_neighbors, in_flow = trap_analysis.get_downslope_neighbors_for_spill_points(
            spill_points, heights, watersheds, nx, ny)

    assert np.array_equal(downslope_neighbors, result_downslope_neighbors)


def test_merge_indices_of_watersheds_using_spill_points():

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
    downslope_neighbors = np.array([40, 15, 11, 20, 21, 24, 34, 27, 37, 46])
    result_merged_indices = [np.array([0, 2, 3, 5, 6, 7]), np.array([1, 4, 8, 9])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(watersheds, downslope_neighbors,
                                                                                  in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_with_loop():

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
    downslope_neighbors = np.array([40, 15, 11, 20, 21, 17, 34, 27, 37, 46])
    result_merged_indices = [np.array([0]), np.array([1, 4, 8, 9]), np.array([2, 3, 5, 6, 7])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(watersheds, downslope_neighbors,
                                                                                  in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_upwards_river():

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
    downslope_neighbors = np.array([40, 15, 11, 20, 7, 17, 34, 27, 22, 29])
    result_merged_indices = [np.array([0]), np.array([1, 4, 8, 9]), np.array([2, 3, 5, 6, 7])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, downslope_neighbors, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_two_watersheds_spilling_in_same_ws():

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

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_already_combined():

    number_of_nodes = 20
    watersheds = [np.array([0, 1, 5, 6]),
                  np.array([2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19]),
                  np.array([10, 11, 15, 16])]

    in_flow = np.array([0, 19, 15])
    out_flow = np.array([0, 19, 15])
    result_merged_indices = [np.array([0]),
                             np.array([1]),
                             np.array([2])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_fill_upwards():

    number_of_nodes = 16
    watersheds = [np.array([0, 1, 4, 5]),
                  np.array([2, 3, 6, 7]),
                  np.array([8, 9, 12, 13]),
                  np.array([10, 11, 14, 15])]

    in_flow = np.array([9, 10, 9, 10])  # River_from
    out_flow = np.array([0, 3, 5, 6])  # River_to
    result_merged_indices = [np.array([0, 2]),
                             np.array([1, 3])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_fill_downwards():

    number_of_nodes = 16
    watersheds = [np.array([0, 1, 4, 5]),
                  np.array([2, 3, 6, 7]),
                  np.array([8, 9, 12, 13]),
                  np.array([10, 11, 14, 15])]

    in_flow = np.array([0, 3, 5, 6])
    out_flow = np.array([9, 10, 12, 15])
    result_merged_indices = [np.array([0, 2]),
                             np.array([1, 3])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_valley():

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

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal


def test_merge_indices_of_watersheds_using_spill_points_valley():

    number_of_nodes = 15
    watersheds = [np.array([0, 5, 6]),
                  np.array([1, 2]),
                  np.array([3, 4]),
                  np.array([7, 8]),
                  np.array([9, 14]),
                  np.array([10, 11]),
                  np.array([12, 13])]

    in_flow = np.array([1, 3, 8, 11, 9, 5, 9])  # River_from
    out_flow = np.array([10, 5, 2, 3, 13, 7, 13])  # River_to
    result_merged_indices = [np.array([0, 1, 2, 3, 5]),
                             np.array([4, 6])]

    merged_indices = trap_analysis.merge_indices_of_watersheds_using_spill_points(
        watersheds, out_flow, in_flow, number_of_nodes)

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal
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

    for i in range(len(merged_watersheds)):
        merged_watersheds[i] = np.sort(merged_watersheds[i])
        result_merged_watersheds[i] = np.sort(result_merged_watersheds[i])
    are_equal = True

    if len(merged_watersheds) != len(result_merged_watersheds):
        are_equal = False
    else:
        for i in range(len(merged_watersheds)):
            elements_not_equal = np.array_equal(merged_watersheds[i], result_merged_watersheds[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

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

    for i in range(len(merged_indices)):
        merged_indices[i] = np.sort(merged_indices[i])

    are_equal = True

    if len(merged_indices) != len(result_merged_indices):
        are_equal = False
    else:
        for i in range(len(merged_indices)):
            elements_not_equal = np.array_equal(merged_indices[i], result_merged_indices[i]) == False
            if elements_not_equal:
                are_equal = False
                break

    assert are_equal
