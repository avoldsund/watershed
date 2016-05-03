import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import load_geotiff
import plot

"""
Plot the landscape using the landscape object holding (x, y, z)-data in 2 dimensions
"""

file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
landscape = load_geotiff.get_landscape(file_name)
downsampling_factor = 8

plot.plot_landscape_2d(landscape, downsampling_factor)
