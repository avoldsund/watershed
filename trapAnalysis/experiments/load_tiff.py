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

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)

"""
 Test creation of the watershed. Load all files instead of redoing it every time.
"""

downslope_neighbors = np.load(saved_files + 'downslopeNeighbors.npy')
endpoints = np.load(saved_files + 'endPoints.npy')

minimum_indices = np.where(downslope_neighbors == -1)[0]

minimums_in_each_watershed = cPickle.load(open(saved_files + 'minimumsInEachWatershed.pkl', 'rb'))
indices_leading_to_endpoints = cPickle.load(open(saved_files + 'indicesLeadingToEndpoints.pkl', 'rb'))
minimums_in_watersheds = cPickle.load(open(saved_files + 'minimumsInWatersheds.pkl', 'rb'))

print 'Done loading files. '
nodes_in_watersheds = trap_analysis.get_nodes_in_watersheds(endpoints, minimums_in_watersheds)
cPickle.dump(nodes_in_watersheds, open('nodesInWatersheds.pkl', 'wb'))
