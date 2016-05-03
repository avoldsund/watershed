import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import util
import mpl_toolkits.mplot3d.axes3d as axes3d


def plot_landscape_2d(landscape, ds):
    """
    Plot the height of the landscape in 2 dimensions given a landscape object
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor to only plot every ds point
    :return: Plot the landscape in the x-y-coordinate system
    """
    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    plt.title('Height of the landscape')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.rcParams.update({'font.size': 20})
    plt.show()


def plot_landscape_3d(landscape, ds):
    """
    Plot the height of the landscape in 3 dimensions given a landscape object
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor to only plot every ds point
    :return: Plot the landscape in the x-y-z-coordinate system
    """

    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('terrain'))
    plt.show()


def plot_watersheds(nodes_in_watersheds, landscape):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param number_of_cols: Number of grid points in the x-direction.
    :return: Plot watersheds
    """

    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::8], y_grid[0::8])
    z = landscape.arr[0::8, 0::8]

    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    plt.title('The largest watersheds in the landscape')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.rcParams.update({'font.size': 10})

    # Only plot watersheds with more than 10000 nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 4000]
    nr_of_large_watersheds = len(large_watersheds)

    colors = iter(cm.flag(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.plot(landscape.x_min + row_col[0::32, 1] * landscape.step_size,
                 landscape.y_max - row_col[0::32, 0] * landscape.step_size, 'ro', color=next(colors), alpha=0.15)
    plt.show()


def plot_watersheds_3d(nodes_in_watersheds, landscape):

    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x_landscape, y_landscape = np.meshgrid(x_grid[0::4], y_grid[0::4])
    z_landscape = landscape.arr[0::4, 0::4]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x_landscape, y_landscape, z_landscape, cmap=plt.get_cmap('terrain'), zorder=0)

    # Only plot watersheds with more than 10000 nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 10000]
    nr_of_large_watersheds = len(large_watersheds)

    colors = iter(cm.flag(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(len(large_watersheds)):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        x = landscape.x_min + row_col[0::10, 1] * landscape.step_size
        y = landscape.y_max - row_col[0::10, 0] * landscape.step_size
        z = landscape.arr[row_col[0::10, 0], row_col[0::10, 1]]
        print z
        ax.scatter(x, y, z, c=next(colors), depthshade=True, zorder=1)

    plt.show()

def plot_local_minimums(local_minimums_coordinates):
    # Do each 16th point

    plt.plot(local_minimums_coordinates[0::16, 0], local_minimums_coordinates[0::16, 1], 'ro')
    plt.show()

