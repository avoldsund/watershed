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

"""
 Make watershed using information about which points in the landscape that are lakes and rivers.
"""

landscape = load_geotiff.get_landscape(heights_file)

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatersheds_addInfo.pkl', 'rb'))

plot.plot_watersheds_add_info(nodes_in_watersheds, landscape, 1)
