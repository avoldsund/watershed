import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import trap_analysis
import numpy as np


def test_get_downslope_minimum_one_min():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([1, 2, 5, 4, 8, 8, 7, 8, -1])
    result_terminal_nodes = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_downslope_minimums(num_of_cols, num_of_rows, downslope_neighbors)
    assert np.array_equal(terminal_nodes, result_terminal_nodes)


def test_get_downslope_minimum_two_mins():

    num_of_cols = 3
    num_of_rows = 3
    downslope_neighbors = np.array([-1, 0, 5, 0, 0, 8, 7, 8, -1])
    result_terminal_nodes = np.array([0, 0, 8, 0, 0, 8, 8, 8, 8])

    terminal_nodes = trap_analysis.get_downslope_minimums(num_of_cols, num_of_rows, downslope_neighbors)
    assert np.array_equal(terminal_nodes, result_terminal_nodes)
