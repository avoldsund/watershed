import sys
sys.path.insert(0, '/home/shomea/a/anderovo/Dropbox/watershed/trapAnalysis/lib')
import util
import numpy as np
import trap_analysis
import cPickle
saved_files = '/home/shomea/a/anderovo/Dropbox/watershedLargeFiles/'
import load_geotiff
import time
import cProfile
import plot

pr = cProfile.Profile()


heights_file = saved_files + 'anders_hoh.tiff'

landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)

#watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsStandard.pkl', 'rb'))
new_watersheds = cPickle.load(open(saved_files + 'watershedsAfterMergingSubTraps.pkl', 'rb'))

#start = time.time()
#new_watersheds, nr_of_iterations = trap_analysis.merge_sub_traps(
#    watersheds, landscape.coordinates[:, 2], landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#print 'There are ', len(new_watersheds), ' watersheds after doing ', nr_of_iterations, ' iterations.'
#end = time.time()-
#print end-start

"""
lakes_file = saved_files + 'anders_innsjo.tiff'
rivers_file = saved_files + 'anders_elvbekk.tiff'
small_rivers_file = saved_files + 'anders_elvmidtlinje.tiff'
marshes_file = saved_files + 'anders_myr.tiff'

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
ds = 1"""
#new_watersheds = cPickle.load(open(saved_files + 'watershedsAfterMergingSubTraps.pkl', 'rb'))
# plot.plot_watersheds_and_lakes_rivers_marshes(landscape, new_watersheds, ds,
#                                              lakes, rivers, small_rivers, marshes)
#cPickle.dump(new_watersheds, open('watershedsAfterMergingSubTraps.pkl', 'wb'))

plot.plot_watersheds_2d(new_watersheds, landscape, 1)

#new_watersheds_second_iteration = trap_analysis.merge_sub_traps(
#    new_watersheds_first_iteration, landscape.coordinates[:, 2], landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#print len(new_watersheds_second_iteration)
#
#new_watersheds_third_iteration = trap_analysis.merge_sub_traps(
#    new_watersheds_second_iteration, landscape.coordinates[:, 2], landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#print len(new_watersheds_third_iteration)
#
#new_watersheds_fourth_iteration = trap_analysis.merge_sub_traps(
#    new_watersheds_third_iteration, landscape.coordinates[:, 2], landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#print len(new_watersheds_fourth_iteration)
#
#new_watersheds_fifth_iteration = trap_analysis.merge_sub_traps(
#    new_watersheds_fourth_iteration, landscape.coordinates[:, 2], landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#print len(new_watersheds_fifth_iteration)


#pr.enable()
#pr.disable()
#pr.print_stats(sort='cumulative')
"""print 'Before boundary pairs'
boundary_pairs = trap_analysis.get_boundary_pairs_in_watersheds(
    watersheds, landscape.num_of_nodes_x, landscape.num_of_nodes_y)
print 'Before min of max'
min_of_max_in_each_watershed, max_heights_of_pairs = \
    trap_analysis.get_min_height_of_max_of_all_pairs(boundary_pairs, landscape.coordinates[:, 2])
print 'Before spill_pairs'
spill_pairs = trap_analysis.get_spill_pair_indices(max_heights_of_pairs, min_of_max_in_each_watershed)
print 'Before steepest spill pairs'
steepest_spill_pairs = trap_analysis.get_steepest_spill_pair(boundary_pairs, spill_pairs)
print 'Before merged indices'
merged_indices = trap_analysis.merge_watersheds_based_on_steepest_pairs(
    steepest_spill_pairs, watersheds, landscape.num_of_nodes_x, landscape.num_of_nodes_y)

new_watersheds = trap_analysis.merge_watersheds_using_merged_indices(watersheds, merged_indices)
plot.plot_watersheds_2d(new_watersheds, landscape, 4)

print len(merged_indices)
print sum([i for i in range(len(merged_indices)) if len(merged_indices[i]) == 1])
#start = time.time()
"""
#end = time.time()
#print end-start


#print boundary_pairs
#boundary_nodes, boundary_pairs = trap_analysis.get_boundary_nodes_in_watersheds(
#    watersheds, landscape.num_of_nodes_x, landscape.num_of_nodes_y)

#boundary_nodes = trap_analysis.get_boundary_nodes_in_watersheds(watersheds,
#                                                                landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#trap_analysis.map_nodes_to_watersheds(watersheds, landscape.num_of_nodes_x * landscape.num_of_nodes_y)

#print boundary_nodes

#merged = trap_analysis.merge_watersheds_using_boundary_pairs(
#    nodes_in_watersheds, landscape.coordinates[:, 2], nbrs_der_dict, landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#ws_indices = ([el for el in range(len(lowest)) if lowest[el] != -1])




"""

#nx = 7
#ny = 7
#watersheds = [np.array([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22]),
#              np.array([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26,
#                        27, 32, 33, 34, 39, 40, 41, 46, 47, 48]),
#              np.array([28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 45])]
#




nx = 7
ny = 7
heights = np.array([10, 10, 10, 10, 10, 10, 10, 10, 1, 8, 7, 7, 7, 10, 10, 6, 8, 5, 5, 5, 10,
                    10, 8, 8, 4, 2, 4, 10, 10, 9, 9, 3, 3, 3, 10, 10, 0, 1, 5, 5, 5, 10, 10, 10, 10, 10, 10, 10, 10])

nbrs_der_dict = util.get_neighbors_derivatives_dictionary(heights, nx, ny)

watersheds = [np.array([0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22]),
              np.array([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20, 23, 24, 25, 26,
                        27, 32, 33, 34, 39, 40, 41, 46, 47, 48]),
              np.array([28, 29, 30, 31, 35, 36, 37, 38, 42, 43, 44, 45])]

merged = trap_analysis.merge_watersheds_using_boundary_pairs(watersheds, heights, nbrs_der_dict, nx, ny)
"""

"""
# The real example
heights_file = saved_files + 'anders_hoh.tiff'

landscape = load_geotiff.get_landscape_tyrifjorden(heights_file)
#cPickle.dump(landscape, open('landscape.pkl', 'wb'))

nodes_in_watersheds = cPickle.load(open(saved_files + 'nodesInWatershedsStandard.pkl', 'rb'))

start = time.time()
nbrs_der_dict = util.get_neighbors_derivatives_dictionary(landscape.coordinates[:, 2],
                                                          landscape.num_of_nodes_x, landscape.num_of_nodes_y)
#cPickle.dump(nbrs_der_dict, open('nbrsDerDict.pkl', 'wb'))
end = time.time()
print end-start

start = time.time()
merged = trap_analysis.merge_watersheds_using_boundary_pairs(
    nodes_in_watersheds, landscape.coordinates[:, 2], nbrs_der_dict, landscape.num_of_nodes_x, landscape.num_of_nodes_y)
end = time.time()
print end-start
"""
