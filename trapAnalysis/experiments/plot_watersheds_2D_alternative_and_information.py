import sys
import cPickle

sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles')

import load_geotiff
import plot
import numpy as np

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
heights_file = saved_files + 'anders_hoh.tiff'
lakes_file = saved_files + 'anders_innsjo.tiff'
rivers_file = saved_files + 'anders_elvbekk.tiff'
small_rivers_file = saved_files + 'anders_elvmidtlinje.tiff'
marshes_file = saved_files + 'anders_myr.tiff'
"""
 Make watershed using information about which points in the landscape that are lakes and rivers.
"""

landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)
ds = 1
nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsAlternative.pkl', 'rb'))

lakes, rivers, small_rivers, marshes = load_geotiff.get_lake_river_marsh_information_tyrifjorden(
    landscape, lakes_file, rivers_file, small_rivers_file, marshes_file)

lakes = lakes.flatten().astype(int)
rivers = rivers.flatten().astype(int)
small_rivers = small_rivers.flatten().astype(int)
marshes = marshes.flatten().astype(int)

lakes = np.where(lakes == 1)[0]
rivers = np.where(rivers == 1)[0]
small_rivers = np.where(small_rivers == 1)[0]
marshes = np.where(marshes == 1)[0]


plot.plot_watersheds_and_lakes_rivers_marshes(landscape, nodes_in_watersheds, ds,
                                              lakes, rivers, small_rivers, marshes)
