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
heights_file = saved_files + 'anders_hoh.tiff'
lakes_file = saved_files + 'anders_innsjo.tiff'
rivers_file = saved_files + 'anders_elvbekk.tiff'
small_rivers_file = saved_files + 'anders_elvmidtlinje.tiff'
marshes_file = saved_files + 'anders_myr.tiff'


# Test to see if create landscape works if one doesn't need to remove last row and first column:
# landscape = load_geotiff.get_landscape(heights_file)
# print landscape.x_min, landscape.x_max, landscape.y_min, landscape.y_max
# print landscape.num_of_nodes_x, landscape.num_of_nodes_y
# print landscape.arr, np.shape(landscape.arr)

#  Test to see how modify landscape works:
# landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)
# print landscape.x_min, landscape.x_max, landscape.y_min, landscape.y_max
# print landscape.num_of_nodes_x, landscape.num_of_nodes_y
# print landscape.arr, np.shape(landscape.arr)

# Test to see if fit_data_in_landscape works:
# landscape = load_geotiff.get_landscape(heights_file)
# marshes = load_geotiff.fit_data_in_landscape(landscape, marshes_file)
# print marshes.flatten()
# print np.shape(marshes)
# print np.sum(marshes.flatten())

# Test to see if get_lake_river_marsh_information works:
# landscape = load_geotiff.get_landscape(heights_file)
# lakes, rivers, small_rivers, marshes =
# load_geotiff.get_lake_river_marsh_information(landscape, lakes_file, rivers_file, small_rivers_file, marshes_file)

# Test to see if fit_data_in_landscape_tyrifjorden works:
# landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)
# marshes = load_geotiff.fit_data_in_landscape_tyrifjorden(landscape, marshes_file)
# print np.shape(marshes)
# print len(np.where(marshes.flatten() == 1)[0])
# print marshes

# Test to see if get_lake_river_marsh_information_tyrifjorden works:
# landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)
# lakes, rivers, small_rivers, marshes = load_geotiff.get_lake_river_marsh_information_tyrifjorden(
#     landscape, lakes_file, rivers_file, small_rivers_file, marshes_file)
