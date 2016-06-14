import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import util
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import axes3d
import matplotlib.colors as colors
import matplotlib.cm as cmx


def get_cmap(N):
    # http://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def plot_landscape_2d(landscape, ds):
    """
    Plot the height of the landscape in 2 dimensions given a landscape object
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor for only plotting every ds point
    :return: Plot the landscape in the x-y-coordinate system
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Decide color map and number of contour levels
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Title and labels
    plt.rcParams.update({'font.size': 14})
    plt.title('The landscape')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_landscape_3d(landscape, ds):
    """
    Plot the height of the landscape in 3 dimensions, given a landscape object.
    :param landscape: Landscape object with all data.
    :param ds: Downsampling factor for only plotting every ds point.
    :return: Plot the landscape in the x-y-z-coordinate system.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Plot (x, y, z) in 3D
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('terrain'))

    plt.show()


def plot_watersheds_2d(nodes_in_watersheds, landscape, ws_above_n_nodes, ds):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds in 2D. Using the
    standard method.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param landscape: Landscape object with all data.
    :param ds: Downsampling factor for only plotting every ds point.
    :return: Plot watersheds in 2D using the standard method.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Plotting the terrain in the background
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot watersheds with more than n nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > ws_above_n_nodes]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    cmap = get_cmap(len(large_watersheds))
    # Plotting all watersheds except the 10 largest
    for i in range(nr_of_large_watersheds - 5):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    color_list = ['gold', 'darkgreen', 'darkorange', 'darkorchid', 'dodgerblue']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # Plot the 10 largest watersheds indigo colored
    for i in range(nr_of_large_watersheds - 5, nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    plt.rcParams.update({'font.size': 14})
    print ws_above_n_nodes
    if ws_above_n_nodes == 0:
        plt.title('All watersheds in the landscape')
    else:
        plt.title('All watersheds with over %s nodes in the landscape'%str(ws_above_n_nodes))
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_watersheds_and_spill_points_2d(nodes_in_watersheds, in_flow, out_flow, landscape, ds):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds in 2D. Using the
    standard method.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param landscape: Landscape object with all data.
    :param ds: Downsampling factor for only plotting every ds point.
    :return: Plot watersheds in 2D using the standard method.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Plotting the terrain in the background
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot watersheds with more than n nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 100]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # Plotting all watersheds except the 10 largest
    for i in range(nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    row_col_in = util.get_row_and_col_from_indices(in_flow, landscape.num_of_nodes_x)
    row_col_out = util.get_row_and_col_from_indices(out_flow, landscape.num_of_nodes_x)

    plt.scatter(landscape.x_min + row_col_in[:, 1] * landscape.step_size,
                landscape.y_max - row_col_in[:, 0] * landscape.step_size,
                color='aqua', s=25, lw=0, alpha=1)
    plt.scatter(landscape.x_min + row_col_out[:, 1] * landscape.step_size,
                landscape.y_max - row_col_out[:, 0] * landscape.step_size,
                color='salmon', s=20, lw=0, alpha=1)

    # Plot the 10 largest watersheds indigo colored
    #for i in range(nr_of_large_watersheds-10, nr_of_large_watersheds):
    #    row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
    #    plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
    #                landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
    #                color='indigo', s=30, lw=0, alpha=0.7)

    plt.rcParams.update({'font.size': 14})
    plt.title('All watersheds with over 100 nodes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_watersheds_3d(nodes_in_watersheds, landscape, ds):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds in 3D. Using the
    standard method.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param landscape: Landscape object with all data.
    :param ds: Downsampling factor for only plotting every ds point.
    :return: Plot watersheds in 3D using the standard method.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x_landscape, y_landscape = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z_landscape = landscape.arr[0::ds, 0::ds]

    # Plot the landscape in 3D
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x_landscape, y_landscape, z_landscape, cmap=plt.get_cmap('terrain'), zorder=0)

    # Plot all watersheds with over n nodes
    large_watersheds = ([watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 5000])
    nr_of_large_watersheds = len(large_watersheds)

    colors = iter(cm.flag(np.linspace(0, 1, nr_of_large_watersheds)))

    # Plot all watersheds on top of the terrain
    for i in range(len(large_watersheds)):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        x = landscape.x_min + row_col[0::ds, 1] * landscape.step_size
        y = landscape.y_max - row_col[0::ds, 0] * landscape.step_size
        z = landscape.arr[row_col[0::ds, 0], row_col[0::ds, 1]]

        ax.scatter(x, y, z, c=next(colors), s=30, lw=0, zorder=1)

    plt.show()


def plot_watersheds_2d_alternative(nodes_in_watersheds, landscape, ds):
    """
    Plot all watersheds after having used information about lakes, rivers and marshes.
    :param nodes_in_watersheds: All watersheds.
    :param landscape: The landscape object.
    :param ds: Downsampling factor.
    :return: Plot all watersheds with information about lakes, rivers and marshes.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Plot terrain in the background
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot watersheds with more than n nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 10]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    # The list of colors it cycles through
    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # Plot all watersheds except the 10 largest
    for i in range(nr_of_large_watersheds - 10):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=25, lw=0, alpha=0.8)

    # Colors for the 10 largest watersheds
    colors = iter(['palevioletred', 'deeppink', 'pink', 'violet', 'aquamarine',
                   'darkorchid', 'purple', 'lawngreen', 'seagreen', 'indigo'])

    # Plot the 10 largest watersheds using different colors
    for i in range(nr_of_large_watersheds-10, nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(colors), s=25, lw=0, alpha=0.8)

    # Set title and labels
    plt.rcParams.update({'font.size': 14})
    plt.title('All watersheds with over 100 nodes, using information about lakes, rivers and marshes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_combined_minimums(combined_minimums, landscape):
    """
    General plotting of minimums. Can decide how large the combinations must be.
    :param combined_minimums: All the combinations of minimums in the landscape.
    :param landscape: The landscape object.
    :return: Plot all combined minimums with over n minimum nodes.
    """

    # Turning sets into arrays, and removing all combinations with less than n nodes
    combined_minimums = [np.array(list(comb)) for comb in combined_minimums]
    combined_minimums = [comb_min for comb_min in combined_minimums if len(comb_min) > 10]

    # Plotting each combination (lake)
    for lake in combined_minimums:
        row_col = util.get_row_and_col_from_indices(lake, landscape.num_of_nodes_x)
        plt.scatter(row_col[:, 1], row_col[:, 0], s=1, facecolor='0.5', lw=0)

    # Fixing labels and axis
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()

    plt.show()


def plot_lakes_rivers_marshes(landscape, lakes, rivers, small_rivers, marshes):
    """
    Plot all lakes, rivers and marshes from the data files in different colors.
    :param landscape: The landscape object.
    :param lakes: The indices for all lakes.
    :param rivers: The indices for all rivers.
    :param small_rivers: The indices for all small rivers.
    :param marshes: The indices for all marshes.
    :return: Plot the lakes, rivers and marshes.
    """

    # Areas, legends and colors to cycle through.
    areas = [lakes, rivers, small_rivers, marshes]
    legends = ['Lakes', 'Large rivers', 'Small rivers', 'Marshes']
    colors = ['darkblue', 'royalblue', 'lightskyblue', 'black']

    for i in range(len(areas)):
        row_col = util.get_row_and_col_from_indices(areas[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::, 0] * landscape.step_size,
                    color=colors[i], s=5, lw=0, alpha=1, label=legends[i])

    # Making the legend
    lakes_patch = mpatches.Patch(color='darkblue', label='Lakes')
    rivers_patch = mpatches.Patch(color='royalblue', label='Large rivers')
    small_rivers_patch = mpatches.Patch(color='lightskyblue', label='Small rivers')
    marshes_patch = mpatches.Patch(color='black', label='Marshes')
    plt.legend(handles=[lakes_patch, rivers_patch, small_rivers_patch, marshes_patch])

    # Title and labels
    plt.rcParams.update({'font.size': 14})
    plt.title('Lakes, rivers and marshes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_watersheds_and_lakes_rivers_marshes(landscape, nodes_in_watersheds, ds, lakes, rivers, small_rivers, marshes):
    """
    Plot all watersheds, as well as lakes, rivers and marshes from the data files in different colors.
    :param landscape: The landscape object.
    :param lakes: The indices for all lakes.
    :param rivers: The indices for all rivers.
    :param small_rivers: The indices for all small rivers.
    :param marshes: The indices for all marshes.
    :return: Plot the lakes, rivers and marshes.
    """

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    # Plot terrain in the background
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot watersheds with more than n nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 100]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    # The list of colors it cycles through
    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # Plot all watersheds except the 10 largest
    for i in range(nr_of_large_watersheds - 10):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=25, lw=0, alpha=0.8)

    # Colors for the 10 largest watersheds
    colors = iter(['palevioletred', 'deeppink', 'pink', 'violet', 'aquamarine',
                   'darkorchid', 'purple', 'lawngreen', 'seagreen', 'indigo'])

    # Plot the 10 largest watersheds using different colors
    for i in range(nr_of_large_watersheds-10, nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(colors), s=25, lw=0, alpha=0.8)

    # Areas, legends and colors to cycle through.
    areas = [lakes, rivers, small_rivers, marshes]
    legends = ['Lakes', 'Large rivers', 'Small rivers', 'Marshes']
    colors = ['darkblue', 'royalblue', 'lightskyblue', 'black']

    for i in range(len(areas)):
        row_col = util.get_row_and_col_from_indices(areas[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::, 0] * landscape.step_size,
                    color=colors[i], s=5, lw=0, alpha=1, label=legends[i])

    # Making the legend for the lakes, rivers and marshes
    lakes_patch = mpatches.Patch(color='darkblue', label='Lakes')
    rivers_patch = mpatches.Patch(color='royalblue', label='Large rivers')
    small_rivers_patch = mpatches.Patch(color='lightskyblue', label='Small rivers')
    marshes_patch = mpatches.Patch(color='black', label='Marshes')
    plt.legend(handles=[lakes_patch, rivers_patch, small_rivers_patch, marshes_patch])

    # Title and labels
    plt.rcParams.update({'font.size': 14})
    plt.title('All watersheds with over 100 nodes, using information about lakes, rivers and marshes')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_landscape_2d_without_landscape_object(heights2d, heights1d, xmin, xmax, ymin, ymax, nx, ny, ds):

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymax, ymin, ny)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = heights2d

    # Decide color map and number of contour levels
    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(heights1d), max(heights1d), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Title and labels
    plt.rcParams.update({'font.size': 14})
    plt.title('The landscape')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


def plot_landscape_3d_without_landscape_object(heights2d, xmin, xmax, ymin, ymax, nx, ny, ds):

    # Construct the (x, y)-coordinate system
    x_grid = np.linspace(xmin, xmax, nx)
    y_grid = np.linspace(ymax, ymin, ny)
    fig = plt.figure()
    X, Y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])

    #ax = fig.add_subplot(111, projection='3d')

    #ax.plot_wireframe(X, Y, heights2d, rstride=1, cstride=1)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, heights2d, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    cmap = plt.get_cmap('terrain')
    fig.colorbar(surf)
    plt.show()
    # Plot (x, y, z) in 3D
    #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #ax.plot_surface(x, y, heights2d, cmap=plt.get_cmap('terrain'))

    plt.show()
