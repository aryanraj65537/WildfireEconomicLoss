import random
import matplotlib.pyplot as plt

# Set your wildfire parameters
wildfire_lat = 0
wildfire_lon = 0
node_dist = 1
scheduler = {
    0: {(wildfire_lon, wildfire_lat)}
}
burned = set()
last_task = 0
cur_task = 0
grid_size = 20  # Define the size of the grid
grid = [[0 for _ in range(2 * grid_size + 1)] for _ in range(2 * grid_size + 1)]

def calc_prob(node1, node2):
    return random.uniform(0.1, 0.3)

def calc_time(node1, node2):
    return random.randint(1, 3)

plt.ion()  # Turn on interactive mode for matplotlib

while cur_task <= last_task:
    if cur_task in scheduler:
        for node in scheduler[cur_task]:
            x, y = node[0] + grid_size, node[1] + grid_size  # Shift the coordinates to fit in the grid
            grid[x][y] = 1  # Mark the cell as burnt

            for lon_dist in range(-1, 2):
                for lat_dist in range(-1, 2):
                    if lon_dist == 0 and lat_dist == 0:
                        continue
                    next_node = (node[0] - lon_dist * node_dist, node[1] - lat_dist * node_dist)
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

            # Create a grid plot and update it
            plt.imshow(grid, cmap='hot', interpolation='nearest', origin='lower')
            plt.pause(0.001)  # Pause for a short time to update the plot

    cur_task += 1
print(scheduler)
print(burned)
# Keep the plot window open until the user closes it
plt.ioff()
plt.show()
