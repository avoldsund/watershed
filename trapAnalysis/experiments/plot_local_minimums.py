import sys

# Linux:
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')

import load_geotiff
import util
import numpy as np
import plot

# Linux:
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)

downslope_neighbors = util.get_downslope_neighbors(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                   landscape.coordinates[0:landscape.total_number_of_nodes, 2])

indices_minimum = (np.where(downslope_neighbors == -1))[0]
local_minimums_coordinates = landscape.coordinates[indices_minimum]
plot.plot_local_minimums(local_minimums_coordinates)
