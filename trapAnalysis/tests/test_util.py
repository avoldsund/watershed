import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import util
import numpy as np
import math


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


def test_get_boundary_indices_square():

    num_of_cols = 3
    num_of_rows = 3
    result_boundary_indices = np.array([0, 1, 2, 3, 5, 6, 7, 8])

    boundary_indices = util.get_boundary_indices(num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)


def test_get_boundary_indices_rectangular():

    num_of_cols = 4
    num_of_rows = 3
    result_boundary_indices = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11])

    boundary_indices = util.get_boundary_indices(num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)


def test_get_interior_indices():

    num_of_cols = 3
    num_of_rows = 3

    result_interior_indices = np.array([4])
    interior_indices = util.get_interior_indices(num_of_cols, num_of_rows)

    assert np.array_equal(interior_indices, result_interior_indices)


#def test_get_downslope_neighbors_boundary():
#
#    num_of_cols = 3
#    num_of_rows = 3
#    heights = np.array([35, 15, 62, 14, 19, 101, 2, 27, 18])
#    result_indices = np.array([3, 3, 1, 6, 6, 8, -1, 6, 8])
#
#    indices = util.get_downslope_neighbors_boundary(num_of_cols, num_of_rows, heights)
#
#    assert np.array_equal(indices, result_indices)


def test_get_neighbors_interior():

    num_of_cols = 5
    num_of_rows = 5
    result_neighbors = np.array([[0, 1, 2, 5, 7, 10, 11, 12],
                                [1, 2, 3, 6, 8, 11, 12, 13],
                                [2, 3, 4, 7, 9, 12, 13, 14],
                                [5, 6, 7, 10, 12, 15, 16, 17],
                                [6, 7, 8, 11, 13, 16, 17, 18],
                                [7, 8, 9, 12, 14, 17, 18, 19],
                                [10, 11, 12, 15, 17, 20, 21, 22],
                                [11, 12, 13, 16, 18, 21, 22, 23],
                                [12, 13, 14, 17, 19, 22, 23, 24]], dtype=int)

    indices, neighbors = util.get_neighbors_interior(num_of_cols, num_of_rows)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_downslope_neighbors_interior():

    num_of_cols = 5
    num_of_rows = 5
    indices = np.array([6, 7, 8, 11, 12, 13, 16, 17, 18])
    neighbors = np.array([[0, 1, 2, 5, 7, 10, 11, 12],
                          [1, 2, 3, 6, 8, 11, 12, 13],
                          [2, 3, 4, 7, 9, 12, 13, 14],
                          [5, 6, 7, 10, 12, 15, 16, 17],
                          [6, 7, 8, 11, 13, 16, 17, 18],
                          [7, 8, 9, 12, 14, 17, 18, 19],
                          [10, 11, 12, 15, 17, 20, 21, 22],
                          [11, 12, 13, 16, 18, 21, 22, 23],
                          [12, 13, 14, 17, 19, 22, 23, 24]])
    heights = np.array([35, 15, 62, 29, 48, 14, 19, 101, 61, 27,
                        2, 27, 18, 149, 20, 31, 33, 36, 24, 22,
                        91, 93, 92, 21, 34])

    result_steepest_neighbors = np.array([10, 12, 9, 10, -1, 12, 10, 12, 12])
    steepest_neighbors = util.get_downslope_neighbors_interior(num_of_cols, num_of_rows, heights)

    assert np.array_equal(steepest_neighbors, result_steepest_neighbors)


def test_get_downslope_indices_corners():

    num_of_cols = 3
    num_of_rows = 3
    heights = np.array([35, 15, 62, 14, 19, 101, 2, 27, 18])
    # result_corners_downslope = np.array([3, 3, 1, 6, 6, 8, -1, 6, -1])
    result_corners_downslope = np.array([[0, 3], [2, 1], [6, -1], [8, -1]])

    corners_downslope = util.get_downslope_indices_corners(num_of_cols, num_of_rows, heights)

    assert np.array_equal(corners_downslope, result_corners_downslope)


def test_get_downslope_indices_sides():

    num_of_cols = 3
    num_of_rows = 3
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    heights = np.array([35, 15, 62, 14, 19, 101, 2, 27, 18])
    # result_corners_downslope = np.array([3, 3, 1, 6, 6, 8, -1, 6, -1])
    result_sides_downslope = np.array([[1, 3], [3, 6], [5, 8], [7, 6]])

    sides_downslope = util.get_downslope_indices_sides(num_of_cols, num_of_rows, heights)

    assert np.array_equal(sides_downslope, result_sides_downslope)


def test_get_downslope_neighbors_boundary():

    num_of_cols = 3
    num_of_rows = 3
    heights = np.array([35, 15, 62, 14, 19, 101, 2, 27, 18])

    result_downslope = np.array([[0, 3], [1, 3], [2, 1], [3, 6], [5, 8], [6, -1], [7, 6], [8, -1]])

    downslope = util.get_downslope_neighbors_boundary(num_of_cols, num_of_rows, heights)

    assert np.array_equal(downslope, result_downslope)

"""
def test_get_downslope_neighbors():

    num_of_cols = 3
    num_of_rows = 3
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    heights = np.array([10, 9, 7, 9, 5, 3, 7, 3, 1])

    result_downslope_neighbors = np.array([4, 5, 5, 7, 8, 8, 7, 8, -1])

    downslope_neighbors = util.get_downslope_neighbors(num_of_cols, num_of_rows, heights)

    assert np.array_equal(downslope_neighbors, result_downslope_neighbors)
"""