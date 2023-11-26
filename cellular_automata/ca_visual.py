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

# Function to calculate the probability of spread
def calc_prob(node1, node2):
    if node2[0] > node1[0] and node2[0] < 5:
        return 1
    return 0

# Function to calculate the time to spread
def calc_time(node1, node2):
    return 1

# Event handler for keyboard input
stop_requested = False
def on_key(event):
    global stop_requested
    if event.key == 'q':
        stop_requested = True

# Turn on interactive mode for matplotlib
plt.ion()
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

while cur_task <= last_task and not stop_requested:
    if cur_task in scheduler:
        for node in scheduler[cur_task]:
            x, y = node[0] + grid_size, node[1] + grid_size
            grid[x][y] = 1

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

            ax.imshow(grid, cmap='hot', interpolation='nearest', origin='lower')
            plt.pause(0.001)

    cur_task += 1

# Disconnect the event handler and close the plot
fig.canvas.mpl_disconnect(fig.canvas.mpl_connect('key_press_event', on_key))
plt.ioff()
plt.show()
