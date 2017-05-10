import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import util
import numpy as np
import networkx
import matplotlib.pyplot as plt
import load_geotiff

#downslope_indices = np.array([-1, 0, 1, 0, 0, 8, 3, 8, -1])
#indices = np.arange(0, len(downslope_indices), 1)

#nx = 6
#ny = 5
#heights = np.array([5, 7, 8, 7, 6, 0, 7, 2, 10, 10, 7, 6, 7, 2, 4, 5, 5, 4, 7, 7, 3.9, 4, 0, 0, 6, 5, 4, 4, 0, 0])

saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
file_name = saved_files + 'anders_hoh.tiff'

landscape = load_geotiff.get_landscape_tyrifjorden(file_name)
print 'before downslopes'
downslope_indices = util.get_downslope_indices(landscape.num_of_nodes_x, landscape.num_of_nodes_y,
                                               landscape.coordinates[:, 2])
print 'after downslopes'
indices = np.arange(0, landscape.total_number_of_nodes, 1)
G = networkx.DiGraph()

print 'ello ello friend'
G.add_nodes_from(indices)

print 'hei'
for i in indices:
    if downslope_indices[i] != -1:
        G.add_edges_from([(i, downslope_indices[i])])

for i in range(len(landscape.total_number_of_nodes)):
    if len(G.predecessors(i)) == 0 and len(G.successors(i)) == 0:
        print i

