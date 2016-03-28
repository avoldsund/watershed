import matplotlib.pyplot as plt

def plot_landscape(grid):
    # Create colormap
    c = plt.get_cmap('terrain')
    c.set_bad(color='k', alpha=0.0)

    # Plot
    plt.imshow(arr[:, 0:1000], cmap=c)
    plt.show()
