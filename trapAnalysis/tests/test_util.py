import sys
sys.path.insert(0, '/home/anderovo/Dropbox/watershed/trapAnalysis/lib')
import util
import compare_methods
import numpy as np
import math


def test_get_node_index_from_coords_square():

    number_of_nodes_x = 3
    number_of_nodes_y = 3

    x = np.arange(0, number_of_nodes_x, 1)
    y = np.arange(0, number_of_nodes_y, 1)
    xx, yy = np.meshgrid(x, y)
    indices = util.get_node_index_from_coords(xx, yy, number_of_nodes_x, number_of_nodes_y)

    result_indices = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])

    assert np.array_equal(indices.flatten(), result_indices)


def test_get_node_index_rectangular():

    number_of_nodes_x = 4
    number_of_nodes_y = 3

    x = np.arange(0, number_of_nodes_x, 1)
    y = np.arange(0, number_of_nodes_y, 1)
    xx, yy = np.meshgrid(x, y)
    indices = util.get_node_index_from_coords(xx, yy, number_of_nodes_x, number_of_nodes_y)

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


def test_get_row_and_col_from_indices():

    number_of_cols = 6
    indices = np.array([9, 15, 16, 17, 21, 22, 23, 27, 28, 29])
    result_row_col = np.array([[1, 3], [2, 3], [2, 4], [2, 5], [3, 3], [3, 4], [3, 5], [4, 3], [4, 4], [4, 5]])

    row_col = util.get_row_and_col_from_indices(indices, number_of_cols)

    assert np.array_equal(row_col, result_row_col)


def test_get_index_from_row_and_col():

    number_of_cols = 2
    rows_and_cols = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    result_indices = [0, 1, 2, 3, 4, 5]

    indices = []
    for row_and_col in rows_and_cols:
        indices.append(util.get_index_from_row_and_col(row_and_col[0], row_and_col[1], number_of_cols))

    assert cmp(indices, result_indices) == 0


def test_get_neighbors_boundary_square():

    num_of_nodes_x = 3
    num_of_nodes_y = 3

    result_neighbors = [[1, 3, 4], [0, 2, 3, 4, 5], [1, 4, 5],
                        [0, 1, 4, 6, 7], [0, 1, 2, 3, 5, 6, 7, 8], [1, 2, 4, 7, 8],
                        [3, 4, 7], [3, 4, 5, 6, 8], [4, 5, 7]]

    neighbors = []
    for i in range(num_of_nodes_x * num_of_nodes_y):
        neighbors.append(util.get_neighbors_boundary(i, num_of_nodes_x, num_of_nodes_y))

    assert cmp(neighbors, result_neighbors) == 0


def test_get_neighbors_boundary_skinny():

    num_of_nodes_x = 2
    num_of_nodes_y = 3

    result_neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3, 4, 5],
                        [0, 1, 2, 4, 5], [2, 3, 5], [2, 3, 4]]

    neighbors = []
    for i in range(num_of_nodes_x * num_of_nodes_y):
        neighbors.append(util.get_neighbors_boundary(i, num_of_nodes_x, num_of_nodes_y))

    assert cmp(neighbors, result_neighbors) == 0


def test_get_neighbors_for_indices_array():

    num_of_nodes_x = 6
    num_of_nodes_y = 5
    indices = np.array([5, 23, 28, 29])
    neighbors = np.array([[4, 10, 11, -1, -1, -1, -1, -1],
                         [16, 17, 22, 28, 29, -1, -1, -1],
                         [21, 22, 23, 27, 29, -1, -1, -1],
                         [22, 23, 28, -1, -1, -1, -1, -1]])

    result_neighbors = util.get_padded_neighbors(indices, num_of_nodes_x, num_of_nodes_y)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_neighbors_for_indices_array():

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

    result_neighbors = util.get_neighbors_for_indices_array(indices, num_of_cols, num_of_rows)

    assert np.array_equal(result_neighbors, neighbors)


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


def test_are_boundary_nodes():

    num_of_cols = 5
    num_of_rows = 5

    indices = np.arange(0, num_of_cols * num_of_rows, 1)
    result_boundary_indices = np.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24])
    boundary_indices = util.are_boundary_nodes(indices, num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)


def test_are_boundary_nodes_2():

    num_of_cols = 6
    num_of_rows = 5

    indices = np.array([5, 7, 13, 22, 23, 28, 29])
    result_boundary_indices = np.array([5, 23, 28, 29])
    boundary_indices = util.are_boundary_nodes(indices, num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)


def test_are_boundary_bool():

    num_of_cols = 6
    num_of_rows = 5

    indices = np.array([5, 7, 13, 22, 23, 28, 29])
    result_boundary_indices = np.array([True, False, False, False, True, True, True])
    boundary_indices = util.are_boundary_nodes_bool(indices, num_of_cols, num_of_rows)

    assert np.array_equal(boundary_indices, result_boundary_indices)



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


def test_get_all_neighbors_interior():

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

    indices, neighbors = util.get_all_neighbors_interior(num_of_cols, num_of_rows)

    assert np.array_equal(neighbors, result_neighbors)


def test_get_different_boundaries():

    nx = 5
    ny = 5

    result_corners = np.array([0, 4, 20, 24])
    result_top = np.array([1, 2, 3])
    result_right = np.array([9, 14, 19])
    result_bottom = np.array([21, 22, 23])
    result_left = np.array([5, 10, 15])

    corners, top, right, bottom, left = util.get_different_boundaries(nx, ny)

    are_equal = (np.array_equal(corners, result_corners) and np.array_equal(top, result_top) and
                 np.array_equal(right, result_right) and np.array_equal(bottom, result_bottom) and
                 np.array_equal(left, result_left))

    assert are_equal


def test_get_neighbors_derivatives_dictionary():

    nx = 3
    ny = 3
    heights = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1])
    b = 2/math.sqrt(200)
    d = 4/math.sqrt(200)

    result_neighbors_derivatives_dictionary = {0: (np.array([1, 3, 4]),
                                                   np.array([0.1, 0.3, d], dtype=float)),
                                               1: (np.array([0, 2, 3, 4, 5]),
                                                   np.array([-0.1, 0.1, b, 0.3, d], dtype=float)),
                                               2: (np.array([1, 4, 5]),
                                                   np.array([-0.1, b, 0.3], dtype=float)),
                                               3: (np.array([0, 1, 4, 6, 7]),
                                                   np.array([-0.3, -b, 0.1, 0.3, d], dtype=float)),
                                               4: (np.array([0, 1, 2, 3, 5, 6, 7, 8]),
                                                   np.array([-d, -0.3, -b, -0.1, 0.1, b, 0.3, d], dtype=float)),
                                               5: (np.array([1, 2, 4, 7, 8]),
                                                   np.array([-d, -0.3, -0.1, b, 0.3], dtype=float)),
                                               6: (np.array([3, 4, 7]),
                                                   np.array([-0.3, -b, 0.1], dtype=float)),
                                               7: (np.array([3, 4, 5, 6, 8]),
                                                   np.array([-d, -0.3, -b, -0.1, 0.1], dtype=float)),
                                               8: (np.array([4, 5, 7]),
                                                   np.array([-d, -0.3, -0.1], dtype=float))}

    neighbors_derivatives_dictionary = util.get_neighbors_derivatives_dictionary(heights, nx, ny)

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(
        neighbors_derivatives_dictionary, result_neighbors_derivatives_dictionary)

    assert are_equal


def test_get_downslope_indices_interior():

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

    result_steepest_neighbors = np.array([[6, 10], [7, 12], [8, 9], [11, 10], [12, -1],
                                         [13, 12], [16, 10], [17, 12], [18, 12]])
    steepest_neighbors = util.get_downslope_indices_interior(num_of_cols, num_of_rows, heights)

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


def test_get_downslope_indices_boundary():

    num_of_cols = 3
    num_of_rows = 3
    heights = np.array([35, 15, 62, 14, 19, 101, 2, 27, 18])

    result_downslope = np.array([[0, 3], [1, 3], [2, 1], [3, 6], [5, 8], [6, -1], [7, 6], [8, -1]])

    downslope = util.get_downslope_indices_boundary(num_of_cols, num_of_rows, heights)

    assert np.array_equal(downslope, result_downslope)


def test_get_downslope_neighbors():

    num_of_cols = 3
    num_of_rows = 3
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    heights = np.array([10, 9, 7, 9, 5, 3, 7, 3, 1])

    result_downslope_neighbors = np.array([4, 5, 5, 7, 8, 8, 7, 8, -1])

    downslope_neighbors = util.get_downslope_indices(num_of_cols, num_of_rows, heights)

    assert np.array_equal(downslope_neighbors, result_downslope_neighbors)


def test_get_downslope_neighbors_large():

    num_of_cols = 6
    num_of_rows = 5

    heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
    result_downslope_neighbors = np.array([7, 7, 7, 4, 5, -1, 7, -1, 7, 15, 5, 5, 13, -1, 13, 22, 22,
                                           23, 13, 13, 13, 22, -1, -1, 25, 26, 20, 28, -1, -1])

    downslope_neighbors = util.get_downslope_indices(num_of_cols, num_of_rows, heights)

    assert np.array_equal(downslope_neighbors, result_downslope_neighbors)
