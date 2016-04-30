import sys

# Linux:
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')
# Windows:
# sys.path.insert(0, "C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib")

import load_geotiff
import util
import numpy as np
import trap_analysis
import time
import cPickle

from tempfile import TemporaryFile
saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'

# Linux:
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
# Windows:
# file_name = 'C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib\\anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)


#downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
#                                                 landscape.coordinates[:, 2])
#downslopeNeighbors = TemporaryFile()  # downslopeNeighbors is the outfile
#np.save('downslopeNeighbors', downslope_neighbors)
downslope_neighbors = np.load(saved_files + 'downslopeNeighbors.npy')

#endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
#endPoints = TemporaryFile()
#np.save('endPoints', endpoints)
endpoints = np.load(saved_files + 'endPoints.npy')

minimum_indices = np.where(downslope_neighbors == -1)[0]

#minimums_in_each_watershed = sorted(trap_analysis.combine_all_minimums_numpy(minimum_indices, landscape.num_of_nodes_x,
#                                    landscape.num_of_nodes_y)) # 43 seconds

#cPickle.dump(minimums_in_each_watershed, open('minimumsInEachWatershed.p', 'wb'))

minimums_in_each_watershed = cPickle.load(open(saved_files + 'minimumsInEachWatershed.pkl', 'rb'))

indices_leading_to_endpoints = cPickle.load(open(saved_files + 'indicesLeadingToEndpoints.pkl', 'rb'))
# Test to improve speed
#print 'Start of method: '
#start = time.time()
#indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints, landscape.total_number_of_nodes)
#end = time.time()
#print end-start

#cPickle.dump(indices_leading_to_endpoints, open('indicesLeadingToEndpoints.pkl', 'wb'))
print 'Done loading files'
print 'Start of getting nodes in watersheds: '
start = time.time()

nodes_in_watersheds = []
minimum_indices = indices_leading_to_endpoints[0]
for i in range(len(minimums_in_each_watershed)):
    ws = []
    for minimum in minimums_in_each_watershed[i]:
        row_index = np.where(minimum_indices == minimum)[0]
        nodes_to_minimum = indices_leading_to_endpoints[1][row_index].tolist()
        ws.extend(nodes_to_minimum)
    nodes_in_watersheds.append(sorted(ws))

end = time.time()
print end-start

#cPickle.dump(indices_leading_to_endpoints, open('indicesLeadingToEndpoints.p', 'wb'))

