import matplotlib.pyplot as plt
import numpy as np
import random
import ee

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize()
# Define the wildfire location and node distance
wildfire_lat = 39.819
wildfire_lon = -121.419
node_dist = .0009

# Initialize the scheduler and burned sets
scheduler = {
    0: {(wildfire_lon, wildfire_lat)}
}
burned = set()

# Task tracking
last_task = 0
cur_task = 0

image_elevation = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
image_temperature = ee.ImageCollection("MODIS/061/MOD11A1").select('LST_Day_1km')
image_biomass = ee.ImageCollection("WCMC/biomass_carbon_density/v1_0").select('carbon_tonnes_per_ha')

def dataset_value(dataset, lon, lat):
    point = ee.Geometry.Point([lon, lat])
    if isinstance(dataset, ee.imagecollection.ImageCollection):
        dataset = dataset.mean()
    value = dataset.reduceRegion(
        reducer=ee.Reducer.first(), 
        geometry=point, 
        scale=30
    ).getInfo()
    return value
# Visualization function
def visualize_fire(scheduler, burned, cur_task):
    # Create a grid for visualization
    x_min, x_max = -121.43, -121.41
    y_min, y_max = 39.81, 39.83
    grid_x, grid_y = np.mgrid[x_min:x_max:node_dist, y_min:y_max:node_dist]
    grid = np.zeros(grid_x.shape, dtype=int)  # 0 for unburned (white)

    # Update the grid for burned and burning nodes
    for burned_node in burned:
        grid[int((burned_node[0] - x_min) / node_dist), int((burned_node[1] - y_min) / node_dist)] = 2  # 2 for burned (black)

    if cur_task in scheduler:
        for node in scheduler[cur_task]:
            grid[int((node[0] - x_min) / node_dist), int((node[1] - y_min) / node_dist)] = 1  # 1 for burning (red)

    # Plotting
    plt.imshow(grid.T, cmap='hot', origin='lower', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar(label='Node State (0: Unburned, 1: Burning, 2: Burned)')
    plt.title(f'Wildfire Spread Simulation at Step {cur_task}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# Probability and time calculation functions (placeholders for your logic)
def calc_prob(node1, node2):
    # Retrieve dataset values
    elevation1 = dataset_value(image_elevation, *node1)['elevation']
    elevation2 = dataset_value(image_elevation, *node2)['elevation']
    temp1 = dataset_value(image_temperature, *node1)['LST_Day_1km']
    temp2 = dataset_value(image_temperature, *node2)['LST_Day_1km']
    biomass1 = dataset_value(image_biomass, *node1)['carbon_tonnes_per_ha']
    biomass2 = dataset_value(image_biomass, *node2)['carbon_tonnes_per_ha']
    #wind_speed1 = dataset_value(image_wind, *node1)['YOUR_WIND_SPEED_BAND']  # Placeholder
    #wind_speed2 = dataset_value(image_wind, *node2)['YOUR_WIND_SPEED_BAND']  # Placeholder
    #moisture1 = dataset_value(image_moisture, *node1)['YOUR_MOISTURE_BAND']  # Placeholder
    #moisture2 = dataset_value(image_moisture, *node2)['YOUR_MOISTURE_BAND']  # Placeholder

    # Calculate factors influenced by elevation, temperature, biomass, wind, and moisture
    elevation_factor = abs(elevation1 - elevation2)
    temperature_factor = abs(temp1 - temp2)
    biomass_factor = abs(biomass1 - biomass2)
    #wind_factor = abs(wind_speed1 - wind_speed2)
    #moisture_factor = abs(moisture1 - moisture2)

    # Simplified probability calculation
    pn = 0.5  # Nominal fire spread probability (assumed)
    alpha_wh = 1 + elevation_factor * 0.1  # Modified for elevation and wind
    em = 1 + temperature_factor * 0.01 - biomass_factor * 0.01 # Modified for temperature, biomass, and moisture
    
    pij = (1 - (1 - pn) ** alpha_wh) * em
    return pij

def calc_time(node1, node2):
    # Retrieve dataset values
    elevation1 = dataset_value(image_elevation, *node1)['elevation']
    elevation2 = dataset_value(image_elevation, *node2)['elevation']
    temp1 = dataset_value(image_temperature, *node1)['LST_Day_1km']
    temp2 = dataset_value(image_temperature, *node2)['LST_Day_1km']
    biomass1 = dataset_value(image_biomass, *node1)['carbon_tonnes_per_ha']
    biomass2 = dataset_value(image_biomass, *node2)['carbon_tonnes_per_ha']
    #moisture1 = dataset_value(image_moisture, *node1)['YOUR_MOISTURE_BAND']  # Placeholder
    #moisture2 = dataset_value(image_moisture, *node2)['YOUR_MOISTURE_BAND']  # Placeholder

    # Calculate factors influenced by elevation, temperature, biomass, and moisture
    elevation_factor = abs(elevation1 - elevation2)
    temperature_factor = abs(temp1 - temp2)
    biomass_factor = abs(biomass1 - biomass2)
    #moisture_factor = abs(moisture1 - moisture2)

    # Simplified time calculation
    d = 1  # Distance between cells (assumed)
    vprop_base = 1  # Base Rate of Spread (assumed)
    # Adjust Rate of Spread based on factors
    vprop = vprop_base + elevation_factor * 0.05 - temperature_factor * 0.01 + biomass_factor * 0.02
    fm = 1# + moisture_factor * 0.01  # Modified for moisture
    
    delta_t = d / (vprop * fm)
    return delta_t


# Main simulation loop
while cur_task <= last_task:
    visualize_fire(scheduler, burned, cur_task)  # Visualize the current state

    if cur_task in scheduler:
        # Simulation logic
        for node in scheduler[cur_task]:
            # Calculate spread to adjacent nodes
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

        cur_task += 1

visualize_fire(scheduler, burned, cur_task)
