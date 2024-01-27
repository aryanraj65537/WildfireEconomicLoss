from simulator import main, createNewForest
import matplotlib.pyplot as plt
import networkx as nx

output = main(20, createNewForest(), [])
print(output[0])
nx.draw(output[1], with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
plt.title('NetworkX Graph')
plt.show()