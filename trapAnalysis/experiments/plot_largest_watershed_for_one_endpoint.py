import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import load_geotiff
import util
import numpy as np
import plot
import trap_analysis

"""
Plot the n nodes with most nodes terminating there
"""

file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

n_largest_watersheds = 20
landscape = load_geotiff.get_landscape(file_name)
downslope_neighbors = util.get_downslope_neighbors(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                   landscape.coordinates[0:landscape.total_number_of_nodes, 2])
endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)

unique, counts = np.unique(endpoints, return_counts=True)
unique_counts = np.column_stack((unique, counts))
unique_counts = unique_counts[np.argsort(unique_counts[:, 1])]

indices = np.where(endpoints == unique_counts[-1, 0])[0]

for i in range(1, n_largest_watersheds):
    indices = np.concatenate((indices, np.where(endpoints == unique_counts[-i - 1, 0])[0]))

coordinates = landscape.coordinates[indices]

plot.plot_watersheds(coordinates)
