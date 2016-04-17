import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import load_geotiff
import plot

"""
Plot the landscape using the landscape object holding (x, y, z)-data
"""

file_name = '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib/anders_hoh.tiff'
landscape = load_geotiff.get_landscape(file_name)

plot.plot_landscape(landscape)
