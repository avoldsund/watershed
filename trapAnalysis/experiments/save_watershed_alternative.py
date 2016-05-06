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
small_rivers_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_elvmidtlinje.tiff'
marshes_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_myr.tiff'

"""
 Save watershed using information about which points in the landscape that are lakes and rivers.
"""

# Construct landscape object
landscape = load_geotiff.get_landscape(heights_file)

# Get information about lakes, rivers and marshes
lake_river_marsh = load_geotiff.get_lake_river_marsh_information(
    landscape, lakes_file, rivers_file, small_rivers_file, marshes_file)
lakes_and_rivers = lake_river_marsh.flatten()

# Get downslope neighbors and modify it with the previous information
downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                                 landscape.coordinates[:, 2])
downslope_neighbors[lakes_and_rivers] = -1

# Get endpoints
endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)

# Get the indices of all minimums, and find minimums in each watershed
minimum_indices = np.where(downslope_neighbors == -1)[0]
minimums_in_each_watershed = sorted(trap_analysis.get_minimums_in_watersheds(minimum_indices, landscape.num_of_nodes_x,
                                    landscape.num_of_nodes_y))

# Get indices ending up in each minimum
indices_leading_to_endpoints = trap_analysis.get_indices_leading_to_endpoints(endpoints)

# Get all watersheds
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_each_watershed)

cPickle.dump(nodes_in_watersheds, open('watershed_using_lake_river_marsh.pkl', 'wb'))
