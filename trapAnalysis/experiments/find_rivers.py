import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import load_geotiff
import util
import trap_analysis
import rivers
import numpy as np
saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

#landscape = load_geotiff.get_landscape_tyrifjorden(file_name)
# Get downslope neighbors
#downslope_neighbors = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
#                                                 landscape.coordinates[:, 2])
# Get endpoints
#endpoints = trap_analysis.get_node_endpoints(landscape.num_of_nodes_x, landscape.num_of_nodes_y, downslope_neighbors)
#nr_of_upstream_nodes = rivers.get_rivers(endpoints)


#nx = 3
#ny = 3
#heights = np.array([0, 1, 2, 1, 2, 3, 2, 3, 1])
#indices = np.arange(0, len(heights), 1)

nx = 6
ny = 5
heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])
indices = np.arange(0, len(heights), 1)
downslope_indices = util.get_downslope_indices(nx, ny, heights)
#print downslope_indices
#
#minimas = np.where(downslope_indices == -1)[0]
#print minimas
#u, count = np.unique(downslope_indices, return_counts=True)
#u = u[1:]
#count = count[1:]
#print u, count
#
#counter = np.zeros(len(heights), dtype=int)
#
#counter[u] = count
#non_min = np.setdiff1d(indices, minimas)
#print non_min
#index = non_min[np.where(counter[non_min] != 0)[0]]
#counter[downslope_indices[index]] += sum(counter[index])
#
#print counter

u, count = np.unique(downslope_indices, return_counts=True)



