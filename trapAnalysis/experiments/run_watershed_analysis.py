import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import load_geotiff
import trap_analysis
import time
import cPickle
import cProfile

"""
    Do the complete watershed analysis: Time spent: 553 seconds ~ 10 minutes
"""
pr = cProfile.Profile()

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
heights_file = saved_files + 'anders_hoh.tiff'

landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)

pr.enable()
watersheds = trap_analysis.calculate_watersheds(landscape)
pr.disable()
pr.print_stats(sort='cumulative')

#cPickle.dump(watersheds, open('watershedsComplete.pkl', 'wb'))
