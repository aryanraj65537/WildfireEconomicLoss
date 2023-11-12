import random
import ee
ee.Authenticate()
ee.Initialize()
wildfire_lat = 39.819
wildfire_lon = -121.419
node_dist = .0009
scheduler = {
    0 : {(wildfire_lon, wildfire_lat)}
}
burned = set()
last_task = 0
cur_task = 0
def calc_prob(node1, node2):
    return random.uniform(.1, .3)
def calc_time(node1, node2):
    return random.randint(1, 3)
while cur_task <= last_task:
    if cur_task in scheduler:
        for node in scheduler[cur_task]:
            for lon_dist in range(-1, 2):
                for lat_dist in range(-1, 2):
                    if lon_dist == 0 and lat_dist == 0:
                        continue
                    next_node = (node[0]-lon_dist*node_dist, node[1]-lat_dist*node_dist)
                    if next_node in burned:
                        continue
                    if random.random() > calc_prob(node, next_node):
                        continue
                    next_time = calc_time(node, next_node)+cur_task
                    last_task = max(last_task, next_time)
                    if next_time in scheduler:
                        scheduler[next_time].add(next_node)
                    else:
                        scheduler[next_time] = {next_node}
            burned.add(node)
    cur_task += 1
print(scheduler)
print(burned)
