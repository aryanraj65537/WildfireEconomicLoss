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

# Define the datasets
image_elevation = ee.Image("NASA/NASADEM_HGT/001").select('elevation')
image_temperature = ee.ImageCollection("MODIS/061/MOD11A1").select('LST_Day_1km')
image_biomass = ee.ImageCollection("WCMC/biomass_carbon_density/v1_0").select('carbon_tonnes_per_ha')
image_winddir = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').select('th')  #homogenous
image_windvel = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').select('vs') #homogenous
image_moisture = ee.ImageCollection("YOUR_MOISTURE_DATASET").select('YOUR_MOISTURE_BANDS')  # Placeholder for moisture dataset

# Function to retrieve dataset values
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

def calc_prob(node1, node2):
    biomass2 = dataset_value(image_biomass, *node2)['carbon_tonnes_per_ha']
    elevdiff = dataset_value(image_elevation, *node1)['elevation']-dataset_value(image_elevation, *node2)['elevation']
    # Simplified probability calculation
    pn = biomass2/891  # Nominal fire spread probability (assumed)
    curnode_dist = node_dist
    if node2[0] != node1[0] and node2[1] != node1[1]:
        curnode_dist *= 2**.5
    slope = elevdiff / (curnode_dist)
    alpha_h = 2**slope
    alpha_w = 
    alpha_wh = alpha_w*alpha_h  # Modified for elevation and wind
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
    if cur_task in scheduler:
        for node in scheduler[cur_task]:
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

# Output the results
print(scheduler)
print(burned)
