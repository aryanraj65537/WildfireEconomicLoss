from simulator import main, createNewForest, MAP_HEIGHT, MAP_WIDTH
import matplotlib.pyplot as plt
import networkx as nx
import random
import itertools
from collections import deque

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):

        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

rishicodefunky = createNewForest()
rishisigma = list(itertools.islice({(i, j) for i in range(MAP_WIDTH) for j in range(MAP_HEIGHT)}, 120))
output = main(60, rishicodefunky, rishisigma, rishicodefunky[0])
pos = hierarchy_pos(output[1], (MAP_WIDTH//2, MAP_HEIGHT//2))
nx.draw(output[1], pos=pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.title('NetworkX Graph')
plt.show()
pos = hierarchy_pos(output[2], output[3][(MAP_WIDTH//2, MAP_HEIGHT//2)])
nx.draw(output[2], pos=pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.title('NetworkX Graph')
plt.show()