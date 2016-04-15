import matplotlib.pyplot as plt
import numpy as np

def plot_landscape(grid):
    # Create colormap
    c = plt.get_cmap('terrain')
    c.set_bad(color='k', alpha=0.0)

    # Plot
    plt.imshow(arr[:, 0:1000], cmap=c)
    plt.show()


def plot_local_minimums(local_minimums_coordinates):
    # Do each 16th point

    plt.plot(local_minimums_coordinates[0::16, 0], local_minimums_coordinates[0::16, 1], 'ro')
    plt.show()

