import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import trap_analysis
import numpy as np
import util


def test_get_node_endpoints_one_min():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([1, 2, 5, 4, 8, 8, 7, 8, -1])
    result_terminal_nodes = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_two_mins():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([-1, 0, 5, 0, 0, 8, 7, 8, -1])
    result_terminal_nodes = np.array([0, 0, 8, 0, 0, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_one_min_rectangular():

    num_of_cols = 2
    num_of_rows = 3
    downslope_neighbors = np.array([3, 3, 5, 5, 5, -1])
    result_terminal_nodes = np.array([5, 5, 5, 5, 5, 5])

    terminal_nodes = trap_analysis.get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_node_endpoints_four_mins_rectangular():

    num_of_cols = 4
    num_of_rows = 3
    downslope_neighbors = np.array([1, 6, 6, 7, 5, 6, -1, -1, 9, 10, -1, -1])
    result_terminal_nodes = np.array([6, 6, 6, 7, 6, 6, 6, 7, 10, 10, 10, 11])

    terminal_nodes = trap_analysis.get_node_endpoints(num_of_cols, num_of_rows, downslope_neighbors)

    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_indices_leading_to_endpoints():

    endpoints = np.array([7, 7, 5, 5, 5, 5, 7, 7, 13, 22, 5, 5, 13, 13, 13, 22,
                          22, 23, 13, 13, 13, 22, 22, 23, 13, 13, 13, 28, 28, 29])
    # unique_endpoints = np.array([5, 7, 13, 22, 23, 28, 29])
    result_indices = [np.array([2, 3, 4, 5, 10, 11]),
                      np.array([0, 1, 6, 7]),
                      np.array([8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                      np.array([9, 15, 16, 21, 22]),
                      np.array([17, 23]),
                      np.array([27, 28]),
                      np.array([29])]

    unique, indices = trap_analysis.get_indices_leading_to_endpoints(endpoints)
    are_equal = True

    if len(result_indices) == len(indices):
        for i in range(len(indices)):
            list_element_is_equal = np.array_equal(indices[i], result_indices[i]) == True
            if list_element_is_equal:
                continue
            else:
                are_equal = False
                break
    else:
        are_equal = False

    assert are_equal is True


def test_get_watersheds():

    minimum_indices = set([5, 7, 13, 22, 23, 28, 29])
    neighbors = {5: set([4, 10, 11]), 7: set([0, 1, 2, 6, 8, 12, 13, 14]), 13: set([6, 7, 8, 12, 14, 18, 19, 20]),
                 22: set([15, 16, 17, 21, 23, 27, 28, 29]), 23: set([16, 17, 22, 28, 29]),
                 28: set([21, 22, 23, 27, 29]), 29: set([22, 23, 28])}
    result_watersheds = {0: set([5]), 1: set([7, 13]), 2: set([22, 23, 28, 29])}
    watersheds = trap_analysis.combine_all_minimums(minimum_indices, neighbors)
    # diff = set(watersheds.keys()) - set(result_watersheds.keys())

    assert watersheds == result_watersheds


def test_get_nodes_to_watersheds():

    watersheds = {0: set([5]), 1: set([7, 13]), 2: set([22, 23, 28, 29])}
    indices_to_endpoints = {5: set([2, 3, 4, 5, 10, 11]), 7: set([0, 1, 6, 7]),
                            13: set([8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                            22: set([9, 15, 16, 21, 22]), 23: set([17, 23]),
                            28: set([27, 28]), 29: set([29])}
    """
    [np.array([2, 3, 4, 5, 10, 11]),
                  np.array([0, 1, 6, 7]),
                  np.array([8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                  np.array([9, 15, 16, 21, 22]),
                  np.array([17, 23]),
                  np.array([27, 28]),
                  np.array([29])]
    """
    result_nodes_in_watershed = {0: set([2, 3, 4, 5, 10, 11]), 1: set([0, 1, 6, 7, 8, 12, 13, 14, 18, 19, 20, 24, 25, 26]),
                                 2: set([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])}

    nodes_in_watershed = trap_analysis.get_nodes_in_watersheds(watersheds, indices_to_endpoints)

    assert nodes_in_watershed == result_nodes_in_watershed
