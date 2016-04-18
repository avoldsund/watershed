import sys

# Linux:
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
# Windows:
# sys.path.insert(0, "C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib")

import load_geotiff
import util
import numpy as np
import trap_analysis
import time

# Linux:
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
# Windows:
# file_name = 'C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib\\anders_hoh.tiff'

#landscape = load_geotiff.get_landscape(file_name)

#print 'Done importing landscape'
#nx = landscape.num_of_nodes_x
#ny = landscape.num_of_nodes_y  # 82 seconds for /8, 225 seconds for /4 OR IMPLEMENT NEW ONE WITH 31.6 /1!!!!!!!
#total = nx * ny

#downslope_neighbors = util.get_downslope_indices(nx, ny, landscape.coordinates[0:total, 2])

#start = time.time()
#terminal_nodes = trap_analysis.get_node_endpoints(nx, ny, downslope_neighbors)
#end = time.time()
#time_taken = end - start
#print time_taken


num_of_cols = 4000
num_of_rows = 4000

indices = np.arange(0, num_of_rows * num_of_cols, 1)

boundary_nodes = []
start = time.time()
for i in range(len(indices)):
    boundary_nodes.append(util.is_boundary_node(indices[i], num_of_cols, num_of_rows))
end = time.time()
print end - start
print indices[boundary_nodes]

start = time.time()

boundary_indices = util.are_boundary_nodes(indices, num_of_cols, num_of_rows)
end = time.time()
print end - start
print boundary_indices
