import sys
import numpy as np
import time
import cPickle
import networkx
from networkx.algorithms.components.connected import connected_components
from tempfile import TemporaryFile

sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import util
import trap_analysis
import time

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'


"""
 Save the watershed using pickle and numpy save.
"""

landscape = load_geotiff.get_landscape(file_name)

# Get downslope neighbors. Save to file.
downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                 landscape.coordinates[:, 2])
downslopeNeighbors = TemporaryFile()
np.save('downslopeNeighbors', downslope_neighbors)

# Get endpoints. Save to file.
endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
endPoints = TemporaryFile()
np.save('endPoints', endpoints)

# Get minimums in each watershed. Save to file.
minimum_indices = np.where(downslope_neighbors == -1)[0]
minimums_in_each_watershed = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                    landscape.num_of_nodes_y))
cPickle.dump(minimums_in_each_watershed, open('minimumsInEachWatershed.pkl', 'wb'))

# Get indices leading to endpoints. Save to file.
indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)
cPickle.dump(indices_leading_to_endpoints, open('indicesLeadingToEndpoints.pkl', 'wb'))

# Get the minimums in each watershed. Save to file.
minimums_in_watersheds = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                                                         landscape.num_of_nodes_y))
cPickle.dump(minimums_in_watersheds, open('minimumsInWatersheds.pkl', 'wb'))

# Get the nodes in the watersheds. Save to file.
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_watersheds)
cPickle.dump(nodes_in_watersheds, open('nodesInWatersheds.pkl', 'wb'))
