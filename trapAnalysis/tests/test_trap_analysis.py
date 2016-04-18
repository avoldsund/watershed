import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import trap_analysis
import numpy as np


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

def test_combine_local_watersheds_three_mins_two_watersheds():

    num_of_cols = 3
    num_of_rows = 3
    terminal_nodes = np.array([0, 0, 8, 0, 0, 8, 7, 7, 8])


def test_get_indices_leading_to_endpoints():

    endpoints = np.array([7, 7, 5, 5, 5, 5, 7, 7, 13, 22, 5, 5, 13, 13, 13, 22,
                          22, 23, 13, 13, 13, 22, 22, 23, 13, 13, 13, 28, 28, 29])
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
