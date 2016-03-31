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

    result_indices = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])

    assert np.array_equal(indices.flatten(), result_indices)


def test_get_node_index_rectangular():

    number_of_nodes_x = 4
    number_of_nodes_y = 3

    x = np.arange(0, number_of_nodes_x, 1)
    y = np.arange(0, number_of_nodes_y, 1)
    xx, yy = np.meshgrid(x, y)
    indices = util.get_node_index(xx, yy, number_of_nodes_x, number_of_nodes_y)

    result_indices = np.array([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])

    assert np.array_equal(indices.flatten(), result_indices)


def test_get_node_neighbors():

    num_of_nodes_x = 3
    num_of_nodes_y = 3

    node_index = 5
    result_neighbors = [1, 4, 5]
    neighbors = util.get_node_neighbors(node_index, num_of_nodes_x, num_of_nodes_y)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_node_row_and_col_from_index_wide():

    number_of_rows = 2
    number_of_cols = 3
    num_of_elements = number_of_rows * number_of_cols
    result_rows_and_cols = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # (r, c)

    rows_and_cols = []
    for i in range(0, num_of_elements):
        rows_and_cols.append(util.get_node_row_and_col_from_index(i, number_of_cols))

    assert cmp(result_rows_and_cols, rows_and_cols) == False


def test_get_node_row_and_col_from_index_skinny():

    number_of_rows = 3
    number_of_cols = 2
    num_of_elements = number_of_rows * number_of_cols
    result_rows_and_cols = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    rows_and_cols = []
    for i in range(0, num_of_elements):
        rows_and_cols.append(util.get_node_row_and_col_from_index(i, number_of_cols))

    assert cmp(result_rows_and_cols, rows_and_cols) == False
