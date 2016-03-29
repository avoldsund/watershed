import sys

# Linux:
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
# Windows:
#sys.path.insert(0, "C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib")

import load_geotiff

# Linux:
file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
# Windows:
#file_name = 'C:\Users\Anders O. Voldsund\Dropbox\watershed\\trapAnalysis\lib\\anders_hoh.tiff'

data_set = load_geotiff.load_data_set(file_name)
arr = load_geotiff.get_array_from_band(data_set)
landscape = load_geotiff.set_landscape(data_set)
load_geotiff.construct_landscape_grid(arr, landscape)
