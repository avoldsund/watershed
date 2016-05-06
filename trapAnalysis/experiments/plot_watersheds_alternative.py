import sys
import cPickle

sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
heights_file = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

"""
 Make watershed using information about which points in the landscape that are lakes and rivers.
"""

landscape = load_geotiff.get_landscape(heights_file)

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsInclMarshes.pkl', 'rb'))

plot.plot_watersheds_add_info(nodes_in_watersheds, landscape, 4)
