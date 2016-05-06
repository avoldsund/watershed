import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import util
import matplotlib.patches as mpatches

def plot_landscape_2d(landscape, ds):
    """
    Plot the height of the landscape in 2 dimensions given a landscape object
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor for only plotting every ds point
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

    plt.rcParams.update({'font.size': 14})

    plt.title('The landscape')
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.savefig('landscape.eps', format='eps', bbox_inches='tight') #dpi='fig.dpi'
    plt.show()


def plot_landscape_3d(landscape, ds):
    """
    Plot the height of the landscape in 3 dimensions given a landscape object
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor for only plotting every ds point
    :return: Plot the landscape in the x-y-z-coordinate system
    """

    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x, y, z, cmap=plt.get_cmap('terrain'))
    plt.show()


def plot_watersheds_2d(nodes_in_watersheds, landscape, ds):
    """
    Plot all or some watersheds in the landscape using different colors for different watersheds.
    :param nodes_in_watersheds: List of arrays. Each array have all indices in the watershed.
    :param landscape: Landscape object with all data
    :param ds: Downsampling factor for only plotting every ds point
    :return: Plot watersheds
    """

    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]
    print y

    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    plt.rcParams.update({'font.size': 12})

    # Only plot watersheds with more than 10000 nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 100]
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # colors = iter(cm.prism(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(nr_of_large_watersheds - 10):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=30, lw=0, alpha=0.7)

    for i in range(nr_of_large_watersheds-10, nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color='indigo', s=30, lw=0, alpha=0.7)

    plt.rcParams.update({'font.size': 14})
    plt.title('All watersheds with over 100 nodes')
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.savefig('watersheds2dStandard.eps', format='eps', dpi=1000, bbox_inches='tight')

    plt.show()


def plot_watersheds_3d(nodes_in_watersheds, landscape, ds):

    x_grid = np.linspace(landscape.x_min + landscape.step_size, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min + landscape.step_size, landscape.num_of_nodes_y)
    x_landscape, y_landscape = np.meshgrid(x_grid[0::2], y_grid[0::2])
    z_landscape = landscape.arr[0::2, 0::2]

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(x_landscape, y_landscape, z_landscape, cmap=plt.get_cmap('terrain'), zorder=0)

    large_watersheds = ([watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 5000])
    nr_of_large_watersheds = len(large_watersheds)

    colors = iter(cm.flag(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(len(large_watersheds)):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        x = landscape.x_min + row_col[0::ds, 1] * landscape.step_size
        y = landscape.y_max - row_col[0::ds, 0] * landscape.step_size
        z = landscape.arr[row_col[0::ds, 0], row_col[0::ds, 1]]

        ax.scatter(x, y, z, c=next(colors), s=30, lw=0, zorder=1)

    plt.show()


def plot_local_minimums(local_minimums_coordinates):
    # Do each 16th point

    plt.plot(local_minimums_coordinates[0::16, 0], local_minimums_coordinates[0::16, 1], 'ro')
    plt.show()


def plot_combined_minimums(combined_minimums, landscape):

    combined_minimums = [np.array(list(comb)) for comb in combined_minimums]
    combined_minimums = [comb_min for comb_min in combined_minimums if len(comb_min) > 10]

    for lake in combined_minimums:
        row_col = util.get_row_and_col_from_indices(lake, landscape.num_of_nodes_x)
        plt.scatter(row_col[:, 1], row_col[:, 0], s=1, facecolor='0.5', lw=0)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    plt.show()


def plot_watersheds_add_info(nodes_in_watersheds, landscape, ds):

    x_grid = np.linspace(landscape.x_min, landscape.x_max, landscape.num_of_nodes_x)
    y_grid = np.linspace(landscape.y_max, landscape.y_min, landscape.num_of_nodes_y)
    x, y = np.meshgrid(x_grid[0::ds], y_grid[0::ds])
    z = landscape.arr[0::ds, 0::ds]

    cmap = plt.get_cmap('terrain')
    v = np.linspace(min(landscape.coordinates[:, 2]), max(landscape.coordinates[:, 2]), 100, endpoint=True)
    plt.contourf(x, y, z, v, cmap=cmap)
    plt.colorbar(label='Height', spacing='uniform')

    # Only plot watersheds with more than n nodes
    large_watersheds = [watershed for watershed in nodes_in_watersheds
                        if len(watershed) > 10]
    small_sheds = [ws for ws in nodes_in_watersheds if len(ws) <= 10]
    #print 'Number of watersheds with less than 500 nodes: ', len(small_sheds)
    large_watersheds.sort(key=len)
    nr_of_large_watersheds = len(large_watersheds)
    #print 'Nr of ws over 100', nr_of_large_watersheds

    color_list = ['red', 'green', 'blue', 'yellow']
    color_list = iter(color_list * (nr_of_large_watersheds/3))

    # colors = iter(cm.prism(np.linspace(0, 1, nr_of_large_watersheds)))

    for i in range(nr_of_large_watersheds - 10):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(color_list), s=25, lw=0, alpha=0.5)

    colors = iter(['lime', 'deeppink', 'chartreuse', 'fuchsia', 'aquamarine',
                   'darkorchid', 'lawngreen', 'purple', 'darkgreen', 'indigo'])
    for i in range(nr_of_large_watersheds-10, nr_of_large_watersheds):
        row_col = util.get_row_and_col_from_indices(large_watersheds[i], landscape.num_of_nodes_x)
        plt.scatter(landscape.x_min + row_col[0::ds, 1] * landscape.step_size,
                    landscape.y_max - row_col[0::ds, 0] * landscape.step_size,
                    color=next(colors), s=25, lw=0, alpha=0.5)

    plt.rcParams.update({'font.size': 14})
    plt.title('All watersheds with over 10 nodes, using information about lakes, rivers and marshes')
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.savefig('watersheds2dInformation.eps', format='eps', dpi=1000, bbox_inches='tight')
    plt.show()


def plot_lakes_rivers_marshes(landscape, lakes, rivers, small_rivers, marshes):
    """
    Plot all lakes, rivers and marshes in different colors
    :param landscape: The landscape object
    :param lakes: The indices for all lakes
    :param rivers: The indices for all rivers
    :param small_rivers: The indices for all small rivers
    :param marshes: The indices for all marshes
    :return:
    """

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

    plt.rcParams.update({'font.size': 14})
    plt.title('Lakes, rivers and marshes')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
