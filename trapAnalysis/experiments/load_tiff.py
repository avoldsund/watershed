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

landscape = load_geotiff.get_landscape(file_name)
print 'Done importing landscape'
nx = landscape.num_of_nodes_x
ny = landscape.num_of_nodes_y/4
total = nx * ny
print nx, ny, total
downslope_neighbors = util.get_downslope_neighbors(nx, ny, landscape.coordinates[0:total, 2])
print downslope_neighbors
print 'Done neighbors: '
print 'Start finding terminal nodes: '
start = time.time()
terminal_nodes = trap_analysis.get_downslope_minimums(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
end = time.time()
time_taken = end - start
print time_taken
print terminal_nodes