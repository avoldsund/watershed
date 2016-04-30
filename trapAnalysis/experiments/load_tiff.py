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
#endpoints = TemporaryFile()
#np.save('endpoints', endpoints)
endpoints = np.load(saved_files + 'endpoints.npy')

minimum_indices = np.where(downslope_neighbors == -1)[0]

#minimums_in_each_watershed = sorted(trap_analysis.combine_all_minimums_numpy(minimum_indices, landscape.num_of_nodes_x,
#                                    landscape.num_of_nodes_y)) # 43 seconds

#cPickle.dump(minimums_in_each_watershed, open('minimumsInEachWatershed.p', 'wb'))

minimums_in_each_watershed = cPickle.load(open(saved_files + 'minimumsInEachWatershed.pkl', 'rb'))

start = time.time()
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)
end = time.time()
print end - start
