import random

# Initialize the scheduler and burned sets
scheduler = {
    0: {(0, 0)}
}

# Controlled burn
burned = set([(i, 2) for i in range(-30, 30)])

# Task tracking
last_task = 0
cur_task = 0

def calc_prob(node1, node2):
    return random.uniform(.25, .75)

def calc_time(node1, node2):
    return random.randint(1, 3)

# Main simulation loop
while cur_task <= last_task:
    if cur_task in scheduler:
        for node in scheduler[cur_task]:
            for lon_dist in range(-1, 2):
                for lat_dist in range(-1, 2):
                    if lon_dist == 0 and lat_dist == 0:
                        continue
                    next_node = (node[0] - lon_dist, node[1] - lat_dist)
                    if next_node in burned:
                        continue
                    if random.random() > calc_prob(node, next_node):
                        continue
                    next_time = calc_time(node, next_node) + cur_task
                    last_task = max(last_task, next_time)
                    if next_time in scheduler:
                        scheduler[next_time].add(next_node)
                    else:
                        scheduler[next_time] = {next_node}
            burned.add(node)
    cur_task += 1

# Output the results
print(scheduler)
print(burned)
