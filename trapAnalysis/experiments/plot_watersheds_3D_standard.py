import sys
import numpy as np
import cPickle
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot
import util

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)

"""
 Plot the watersheds in 3d with the landscape below
"""

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatersheds.pkl', 'rb'))

plot.plot_watersheds_3d(nodes_in_watersheds, landscape, 4)
