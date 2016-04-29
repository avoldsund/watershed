import matplotlib.pyplot as plt
import numpy as np


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


def plot_local_minimums(local_minimums_coordinates):
    # Do each 16th point

    plt.plot(local_minimums_coordinates[0::16, 0], local_minimums_coordinates[0::16, 1], 'ro')
    plt.show()


def plot_watersheds(node_coordinates):

    plt.plot(node_coordinates[:, 0], node_coordinates[:, 1], 'ro')
    #plt.gca().invert_yaxis()
    plt.show()
