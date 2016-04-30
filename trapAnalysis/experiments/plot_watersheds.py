import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/util')
import load_geotiff
import trap_analysis

"""
Get all watersheds by combining the minimums and the nodes having them as endpoints
"""

file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'

landscape = load_geotiff.get_landscape(file_name)
watersheds = trap_analysis.get_watersheds(landscape.coordinates[:, 2], landscape.num_of_nodes_x,
                                          landscape.num_of_nodes_y)

print sorted(watersheds)