import sys
import numpy as np
import cPickle
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot
import util

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

"""
 Plot the watersheds in 3d with the landscape below
"""

landscape = load_geotiff.get_landscape_tyrifjorden(file_name)

nodes_in_watersheds, nr_of_iterations = cPickle.load(open(saved_files + 'watershedsComplete.pkl', 'rb'))

plot.plot_watersheds_3d(nodes_in_watersheds, landscape, 8)
