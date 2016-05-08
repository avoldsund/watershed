import sys
import cPickle

sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
heights_file = saved_files + 'anders_hoh.tiff'

"""
 Make watershed using information about which points in the landscape that are lakes and rivers.
"""

landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsAlternative.pkl', 'rb'))
print 'Total number of watersheds: ', len(nodes_in_watersheds)
small_watersheds = ([watershed for watershed in nodes_in_watersheds if len(watershed) <= 100])
print len(small_watersheds)

plot.plot_watersheds_2d_alternative(nodes_in_watersheds, landscape, 4)
