# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:35:07 2017

@author: h501zgrl
"""
import numpy as np


array = np.array([[[ 1.,  1.,  0.],
                   [ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]],
                  [[1.,  0.,  0.],
                   [ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]],
                  [[ 1.,  0.,  0.],
                   [ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]]])
                   
dire = np.array(([1, 0, 0], [0, 1, 0], [0, 0, 1])) # Possible directions
results = np.zeros((array.shape[0], len(dire)))  # Maximum runs will be 3
#results2 = np.zeros(shape(results))                               

# Find all indices
indices = np.argwhere(array == 1) 
# Loop over each direction                 
for idire, dire in enumerate(dire):
    results[0, idire] = len(indices) # Count all 1 (conection 0)
    indices_to_compare = indices 
    print('dir:',dire, 'run', 0)
    print(indices_to_compare)
     
    for irun in range(1, array.shape[0]):
        print('dir:',dire, 'run', irun)
        indices_displaced = indices + dire*irun             
        aset = set([tuple(x) for x in indices_displaced])
        bset = set([tuple(x) for x in indices_to_compare])
        indices_to_compare = (np.array([x for x in aset & bset]))
        print(indices_to_compare)
        print('counts:',len(indices_to_compare))
        results[irun, idire] = len(indices_to_compare)
        
    print('\n Bruts:\n',results)
    
    print('\n clin')
    
    for iindx in np.arange(array.shape[0]-2,-1,-1):
#        idx0 = array.shape[0]-1
#        results[indx, idire] -= np.sum(results[np.arange(idx0,indx,-1), idire]
#                                       *np.arange(idx0,indx,-1))
        for jindx in np.arange(iindx+1, array.shape[0]):
            print(iindx,jindx,  results[jindx, idire])
            results[iindx, idire] -=  results[jindx, idire] *(jindx-iindx+1)         
            
            
   #     print(results[np.arange(idx0,indx,-1), idire])
    #    print(np.arange(idx0,indx,-1))
print('\n',results)



#import networkx
#
#def array_to_graph(bool_array, allowed_steps):
#     """
#     Arguments:
#     ----------
#     bool_array    -- boolean array
#     allowed_steps -- list of allowed steps; e.g. given a 2D boolean array,
#                      [(0, 1), (1, 1)] signifies that from coming from element
#                      (i, j) only elements (i, j+1) and (i+1, j+1) are reachable
#
#     Returns:
#     --------
#     g               -- networkx.DiGraph() instance
#     position_to_idx -- dict mapping (i, j) position to node idx
#     idx_to_position -- dict mapping node idx to (i, j) position
#     """
#
#     # ensure that bool_array is boolean
#     assert bool_array.dtype == np.bool, "Input array has to be boolean!"
#
#     # map array indices to node indices and vice versa
#     node_idx = range(np.sum(bool_array))
#     node_positions = zip(*np.where(bool_array))
#     position_to_idx = dict(zip(node_positions, node_idx))
#
#     # create graph
#     g = networkx.DiGraph()
#     for source in node_positions:
#         for step in allowed_steps: # try to step in all allowed directions
#             target = tuple(np.array(source) + np.array(step))
#             if target in position_to_idx:
#                 g.add_edge(position_to_idx[source], position_to_idx[target])
#
#     idx_to_position = dict(zip(node_idx, node_positions))
#
#     return g, idx_to_position, position_to_idx
#
#def get_connected_component_sizes(g):
#     component_generator = networkx.connected_components(g)
#     component_sizes = [len(component) for component in component_generator]
#     return component_sizes
#
#def test():
#     array = np.array([[[ 1.,  1.,  0.],
#                        [ 0.,  0.,  0.],
#                        [ 0.,  0.,  0.]],
#                       [[1.,  0.,  0.],
#                        [ 0.,  0.,  0.],
#                        [ 0.,  0.,  0.]],
#                       [[ 1.,  0.,  0.],
#                        [ 0.,  0.,  0.],
#                        [ 0.,  0.,  0.]]], dtype=np.bool)
#
#     directions = [(1,0,0), (0,1,0), (0,0,1)]
#
#     for direction in directions:
#         graph, _, _ = array_to_graph(array, [direction])
#         sizes = get_connected_component_sizes(graph.to_undirected())
#         counts = np.bincount(sizes)
#         print('b',sizes)
#         print(direction)
#         print('a')
#         for ii, c in enumerate(counts):
#             if ii > 1:
#                 print ("Runs of size")
#
#     return
#test()
