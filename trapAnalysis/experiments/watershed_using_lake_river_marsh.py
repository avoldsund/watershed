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
import plot

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
heights_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
lakes_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_innsjo.tiff'
rivers_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_elvbekk.tiff'
smallRivers_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_elvmidtlinje.tiff'

"""
 Make watershed using information about which points in the landscape that are lakes and rivers.
"""

landscape = load_geotiff.get_landscape(heights_file)
"""
lakes_and_rivers = load_geotiff.get_lake_river_information(landscape, lakes_file, rivers_file, smallRivers_file)
lakes_and_rivers = lakes_and_rivers.flatten()

print lakes_and_rivers

downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                 landscape.coordinates[:, 2])
downslope_neighbors[lakes_and_rivers] = -1
print 'Done making additional downslope neighbors -1'

endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
print 'Done endpoints'
minimum_indices = np.where(downslope_neighbors == -1)[0]
minimums_in_each_watershed = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                    landscape.num_of_nodes_y))
indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)
minimums_in_watersheds = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                                                         landscape.num_of_nodes_y))
print 'All before nodes in watersheds'
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_watersheds)
print 'after nodes_in_watersheds'
"""
#cPickle.dump(nodes_in_watersheds, open('nodesInWatersheds.pkl', 'wb'))

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatersheds_addInfo.pkl', 'rb'))

small_sheds = [ws for ws in nodes_in_watersheds if len(ws) <= 100]
print 'Nr of watersheds: ', len(nodes_in_watersheds)
print 'Nr of watersheds below 100 nodes: ', len(small_sheds)
plot.plot_watersheds_add_info(nodes_in_watersheds, landscape, 4)


"""
endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
minimum_indices = np.where(downslope_neighbors == -1)[0]
minimums_in_each_watershed = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                    landscape.num_of_nodes_y))
indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)
minimums_in_watersheds = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                                                         landscape.num_of_nodes_y))
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_watersheds)
"""
