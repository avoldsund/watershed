import numpy as np
import sys
path = '/home/anderovo/Dropbox/watershed/trapAnalysis/'
sys.path.insert(0, path + 'lib')
import compare_methods


def test_compare_two_dictionaries_where_values_are_arrays_one():
    #  Dictionaries have different lengths

    d1 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
    d2 = {0: np.array([1, 2, 3])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_two():
    # Dictionaries have different keys

    d1 = {0: np.array([1, 2, 3])}
    d2 = {1: np.array([1, 2, 3])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_three():
    # The values are in different orders

    d1 = {0: np.array([1, 2, 3])}
    d2 = {1: np.array([1, 3, 2])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_arrays_four():
    # The dictionaries are equal

    d1 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}
    d2 = {0: np.array([1, 2, 3]), 1: np.array([4, 5, 6])}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_arrays(d1, d2)

    assert are_equal


def test_compare_two_dictionaries_where_values_are_tuples_with_two_arrays_one():
    # Dictionaries have different lengths

    d1 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}
    d2 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3]))}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_tuples_with_two_arrays_two():
    # Dictionaries have different keys

    d1 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}
    d2 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 2: (np.array([4, 5, 6]), np.array([1, 2, 3]))}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_tuples_with_two_arrays_three():
    # The values are in different orders

    d1 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}
    d2 = {0: (np.array([1, 2, 3]), np.array([1, 3, 2])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(d1, d2)

    assert are_equal is False


def test_compare_two_dictionaries_where_values_are_tuples_with_two_arrays_four():
    # The dictionaries are equal

    d1 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}
    d2 = {0: (np.array([1, 2, 3]), np.array([1, 2, 3])), 1: (np.array([4, 5, 6]), np.array([1, 2, 3]))}

    are_equal = compare_methods.compare_two_dictionaries_where_values_are_tuples_with_two_arrays(d1, d2)

    assert are_equal


def test_compare_two_lists_of_arrays_one():
    # The lists have different lengths

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([1, 2, 3])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_two():
    # The lists have arrays in a different order

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([4, 5, 6]), np.array([1, 2, 3])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_three():
    # The lists have arrays with different orderings

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([1, 2, 3]), np.array([6, 5, 4])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_four():
    # The lists have arrays of different length

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6, 7])]
    l2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_arrays_five():
    # The lists are equal

    l1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    l2 = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    are_equal = compare_methods.compare_two_lists_of_arrays(l1, l2)

    assert are_equal


def test_compare_two_lists_of_tuples_one():
    # The lists have different lengths

    l1 = [(1, 2), (3, 4)]
    l2 = [(1, 2)]

    are_equal = compare_methods.compare_two_lists_of_tuples(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_tuples_two():
    # The lists have different tuples

    l1 = [(1, 2), (3, 4)]
    l2 = [(1, 2), (4, 3)]

    are_equal = compare_methods.compare_two_lists_of_tuples(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_tuples_three():
    # The tuples are in different orders

    l1 = [(1, 2), (3, 4)]
    l2 = [(3, 4), (1, 2)]

    are_equal = compare_methods.compare_two_lists_of_tuples(l1, l2)

    assert are_equal is False


def test_compare_two_lists_of_tuples_four():
    # The lists are equal

    l1 = [(1, 2), (3, 4)]
    l2 = [(1, 2), (3, 4)]

    are_equal = compare_methods.compare_two_lists_of_tuples(l1, l2)

    assert are_equal
