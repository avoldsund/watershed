import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')

import util
import numpy as np


def test_get_node_index_square():

    number_of_nodes_x = 3
    number_of_nodes_y = 3

    x = np.arange(0, number_of_nodes_x, 1)
    y = np.arange(0, number_of_nodes_y, 1)
    xx, yy = np.meshgrid(x, y)
    indices = util.get_node_index(xx, yy, number_of_nodes_x, number_of_nodes_y)

    result_array = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])

    assert np.array_equal(indices.flatten(), result_array)


def test_get_node_index_rectangular():

    number_of_nodes_x = 4
    number_of_nodes_y = 3

    x = np.arange(0, number_of_nodes_x, 1)
    y = np.arange(0, number_of_nodes_y, 1)
    xx, yy = np.meshgrid(x, y)
    indices = util.get_node_index(xx, yy, number_of_nodes_x, number_of_nodes_y)

    result_array = np.array([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])

    assert np.array_equal(indices.flatten(), result_array)


def test_get_node_neighbors():

