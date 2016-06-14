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
file_name = saved_files + 'anders_hoh.tiff'


"""
 Save the watershed using pickle and numpy save.
"""

landscape = load_geotiff.get_landscape_tyrifjorden(file_name)

# Get downslope neighbors
#downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
#                                                 landscape.coordinates[:, 2])
#
## Get endpoints
#endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
#
## Get minimums in each watershed
#minimum_indices = np.where(downslope_neighbors == -1)[0]
#minimums_in_each_watershed = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
#                                    landscape.num_of_nodes_y))
#
## Get indices leading to endpoints
#indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)

# Get the nodes in the watersheds. Save to file.
#nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)
#cPickle.dump(nodes_in_watersheds, open('nodesInWatershedsStandard.pkl', 'wb'))

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsStandard.pkl', 'rb'))

#start = time.time()
boundary_nodes = trap_analysis.get_boundary_nodes_in_watersheds(nodes_in_watersheds, landscape.num_of_nodes_x,
                                                                landscape.num_of_nodes_y)
#
cPickle.dump(boundary_nodes, open('boundaryNodesInWatersheds.pkl', 'wb'))

#boundary_nodes = cPickle.load(open(saved_files + 'boundaryNodesInWatersheds.pkl', 'rb'))
#
spill_points = trap_analysis.get_spill_points(boundary_nodes, landscape.coordinates[:, 2])
#
#
print 'done spill_points'
out_flow, in_flow = trap_analysis.get_downslope_neighbors_for_spill_points(
    spill_points, landscape.coordinates[:, 2], nodes_in_watersheds, landscape.num_of_nodes_x, landscape.num_of_nodes_y)
print 'Done out_flow, in_flow'
merged_indices = trap_analysis.merge_indices_of_watersheds_graph(
    nodes_in_watersheds, landscape.total_number_of_nodes, in_flow, out_flow)

print 'Number of watersheds: ', len(merged_indices)
small_ws = [ws for ws in merged_indices if len(ws) == 1]
print 'Number of alone watersheds: ', len(small_ws)

#start = time.time()
#new_watersheds = trap_analysis.merge_watersheds_using_merged_indices(nodes_in_watersheds, merged_indices_of_watersheds)
#end = time.time()
#print 'Time to merged watersheds using merged indices: ', end-start

new_watersheds = []

for i in range(len(merged_indices)):
    merged_ws = np.concatenate([nodes_in_watersheds[j] for j in merged_indices[i]])
    new_watersheds.append(merged_ws)

#cPickle.dump(nodes_in_watersheds, open('watershedsUsingSpillPointAnalysis.pkl', 'wb'))
#start = time.time()
#new_watersheds = cPickle.load(open(saved_files + 'watershedsUsingSpillPointAnalysis.pkl', 'rb'))
#end = time.time()
#print 'Number of watersheds: ', len(new_watersheds)
#small_ws = [ws for ws in new_watersheds if len(ws) == 1]
#print 'Number of alone watersheds: ', len(small_ws)

#print 'Time for loading new watersheds: ', end - start

# plot.plot_watersheds_2d(new_watersheds, landscape, 4)
# plot.plot_watersheds_and_spill_points_2d(new_watersheds, in_flow, out_flow, landscape, 4)
plot.plot_watersheds_2d(new_watersheds, landscape, 8)
#plot.plot_watersheds_and_spill_points_2d(nodes_in_watersheds, in_flow, out_flow, landscape, 4)
