import numpy as np
import sys
path = '/home/anderovo/Dropbox/watershed/trapAnalysis/'
sys.path.insert(0, path + 'lib')
import compare_methods
import rivers


def test_get_nr_of_upstream_nodes_basic():

    nx = 3
    ny = 3
    heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
    result_nr_of_upstream_nodes = np.array([5, 1, 0, 1, 0, 0, 0, 0, 2])

    nr_of_upstream_nodes = rivers.get_nr_of_upstream_nodes(heights, nx, ny)

    are_equal = np.array_equal(nr_of_upstream_nodes, result_nr_of_upstream_nodes)

    assert are_equal


def test_get_nr_of_upstream_nodes_advanced():

    nx = 6
    ny = 5
    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    result_nr_of_upstream_nodes = np.array([0, 0, 0, 0, 1, 4, 0, 5, 0, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 0, 3, 0, 4, 1,
                                            0, 1, 1, 0, 1, 0])

    nr_of_upstream_nodes = rivers.get_nr_of_upstream_nodes(heights, nx, ny)

    are_equal = np.array_equal(nr_of_upstream_nodes, result_nr_of_upstream_nodes)

    assert are_equal