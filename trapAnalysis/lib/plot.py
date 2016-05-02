import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import util



def plot_landscape(landscape):
    """
    Plot the height of the landscape given a landscape object
    :param landscape: Landscape object with all data
    :return: Plot the landscape in the x-y-coordinate system
    """
    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::8], y_grid[0::8])
    z = landscape.arr[0::8, 0::8]

    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    plt.title('Height of the landscape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.rcParams.update({'font.size': 20})
    plt.show()


def plot_watersheds(nodes_in_watersheds, number_of_cols):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param number_of_cols: Number of grid points in the x-direction.
    :return: Plot watersheds
    """

    # Only plot watersheds with more than 10000 nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 5000]
    nr_of_large_watersheds = len(large_watersheds)

    print nr_of_large_watersheds
    colors = iter(cm.rainbow(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], number_of_cols)
        plt.plot(row_col[:, 1], row_col[:, 0], 'ro', color=next(colors))
    plt.gca().invert_yaxis()
    plt.show()


def plot_local_minimums(local_minimums_coordinates):
    # Do each 16th point

    plt.plot(local_minimums_coordinates[0::16, 0], local_minimums_coordinates[0::16, 1], 'ro')
    plt.show()

