import sys

# Linux:
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')

import load_geotiff
import util
import numpy as np
import plot
import trap_analysis

# Linux:
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)

downslope_neighbors = util.get_downslope_neighbors(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                   landscape.coordinates[0:landscape.total_number_of_nodes, 2])
endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)

unique, counts = np.unique(endpoints, return_counts=True)
#print 'hellooo'
#freq = itemfreq(endpoints)  # unique, counts
#unique = freq[:, 0]
#counts = freq[:, 1]
#print 'bye'
index_in_unique = np.argmax(counts)
all_nodes_lead_to = unique[index_in_unique]


indices = np.where(endpoints == all_nodes_lead_to)

coordinates = landscape.coordinates[indices]
plot.plot_local_minimums(coordinates)

