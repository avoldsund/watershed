import sys
import cPickle
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot
import util

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

"""
 Plot all combined minimums in the landscape.
"""

landscape = load_geotiff.get_landscape(file_name)
combined_minimums = cPickle.load(open(saved_files + 'minimumsInWatersheds.pkl', 'rb'))

plot.plot_combined_minimums(combined_minimums, landscape)
