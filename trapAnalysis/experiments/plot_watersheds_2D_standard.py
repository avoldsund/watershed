import sys
import numpy as np
import cPickle
sys.path.insert(0, '/home/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot
import util

saved_files = '/home/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

"""
 Plot the watersheds in 2d
"""

landscape = load_geotiff.get_landscape_tyrifjorden(file_name)

nodes_in_watersheds, nr_of_iterations = cPickle.load(open(saved_files + 'watershedsComplete.pkl', 'rb'))
ws_with_above_n_nodes = 0

plot.plot_watersheds_2d(nodes_in_watersheds, landscape, ws_with_above_n_nodes, 4)
