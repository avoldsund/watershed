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


def test_get_row_and_col_from_index_wide():

    number_of_rows = 2
    number_of_cols = 3
    num_of_elements = number_of_rows * number_of_cols
    result_rows_and_cols = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]  # (r, c)

    rows_and_cols = []
    for i in range(0, num_of_elements):
        rows_and_cols.append(util.get_row_and_col_from_index(i, number_of_cols))

    assert cmp(result_rows_and_cols, rows_and_cols) == 0


def test_get_row_and_col_from_index_skinny():

    number_of_rows = 3
    number_of_cols = 2
    num_of_elements = number_of_rows * number_of_cols
    result_rows_and_cols = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    rows_and_cols = []
    for i in range(0, num_of_elements):
        rows_and_cols.append(util.get_row_and_col_from_index(i, number_of_cols))

    assert cmp(result_rows_and_cols, rows_and_cols) == 0


def test_get_index_from_row_and_col():

    number_of_cols = 2
    rows_and_cols = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    result_indices = [0, 1, 2, 3, 4, 5]

    indices = []
    for row_and_col in rows_and_cols:
        indices.append(util.get_index_from_row_and_col(row_and_col[0], row_and_col[1], number_of_cols))

    assert cmp(indices, result_indices) == 0


def test_get_node_neighbors_boundary_square():

    num_of_nodes_x = 3
    num_of_nodes_y = 3

    result_neighbors = [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5],
                        [0, 1, 4, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 7, 8],
                        [3, 4, 7], [3, 4, 5, 6, 8], [4, 5, 7]]

    neighbors = []
    for i in range(num_of_nodes_x * num_of_nodes_y):
        neighbors.append(util.get_node_neighbors_boundary(i, num_of_nodes_x, num_of_nodes_y))

    assert cmp(neighbors, result_neighbors) == 0


def test_get_node_neighbors_boundary_skinny():

    num_of_nodes_x = 2
    num_of_nodes_y = 3

    result_neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3, 4, 5],
                        [0, 1, 2, 4, 5], [2, 3, 5], [2, 3, 4]]

    neighbors = []
    for i in range(num_of_nodes_x * num_of_nodes_y):
        neighbors.append(util.get_node_neighbors_boundary(i, num_of_nodes_x, num_of_nodes_y))

    assert cmp(neighbors, result_neighbors) == 0


def test_is_boundary_node_square():

    num_of_cols = 3
    num_of_rows = 3

    indices = range(0, num_of_cols * num_of_rows)
    result_are_boundary_nodes = [True, True, True, True, False, True, True, True, True]

    boundary_nodes = []
    for index in indices:
        boundary_nodes.append(util.is_boundary_node(index, num_of_cols, num_of_rows))

    print boundary_nodes
    assert cmp(boundary_nodes, result_are_boundary_nodes) == 0


def test_is_boundary_node_wide():

    num_of_cols = 3
    num_of_rows = 2

    indices = range(0, num_of_cols * num_of_rows)
    result_are_boundary_nodes = [True, True, True, True, True, True]

    boundary_nodes = []
    for index in indices:
        boundary_nodes.append(util.is_boundary_node(index, num_of_cols, num_of_rows))

    print boundary_nodes
    assert cmp(boundary_nodes, result_are_boundary_nodes) == 0


def test_is_boundary_node_large():

    num_of_cols = 5
    num_of_rows = 5

    indices = range(0, num_of_cols * num_of_rows)
    result_are_boundary_nodes = [True, True, True, True, True,
                                 True, False, False, False, True,
                                 True, False, False, False, True,
                                 True, False, False, False, True,
                                 True, True, True, True, True]

    boundary_nodes = []
    for index in indices:
        boundary_nodes.append(util.is_boundary_node(index, num_of_cols, num_of_rows))

    print boundary_nodes
    assert cmp(boundary_nodes, result_are_boundary_nodes) == 0


def test_get_boundary_node_indices():

    num_of_cols = 4001
    num_of_rows = 4001
    result_boundary_indices = np.array([0, 1, 2, 6, 7, 8, 3, 5])

    boundary_indices = util.get_boundary_node_indices(num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)
