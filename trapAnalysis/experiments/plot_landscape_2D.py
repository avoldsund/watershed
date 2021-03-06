import sys
sys.path.insert(0, '/home/anderovo/Dropbox/watershed/trapAnalysis/lib')
import load_geotiff
import plot

"""
Plot the landscape using the landscape object holding (x, y, z)-data in 2 dimensions
"""

file_name = '/home/anderovo/Dropbox/watershedLargeFiles/anders_hoh.tiff'

landscape = load_geotiff.get_landscape_tyrifjorden(file_name)
downsampling_factor = 1

plot.plot_landscape_2d(landscape, downsampling_factor)
