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
def calc_prob(node1, node2):
    # Extracting elevation, temperature, and biomass for both nodes
    elevation1 = dataset_value(image_elevation, *node1)['elevation']
    elevation2 = dataset_value(image_elevation, *node2)['elevation']
    temp1 = dataset_value(image_temperature, *node1)['LST_Day_1km']
    temp2 = dataset_value(image_temperature, *node2)['LST_Day_1km']
    biomass1 = dataset_value(image_biomass, *node1)['carbon_tonnes_per_ha']
    biomass2 = dataset_value(image_biomass, *node2)['carbon_tonnes_per_ha']

    # Probability factors
    elevation_factor = 1.0 + (elevation2 - elevation1) / 100  # Higher elevation difference increases probability
    temperature_factor = 1.0 + (temp2 - temp1) / 300  # Higher temperature increases probability
    biomass_factor = 1.0 + (biomass2 - biomass1) / 10  # Higher biomass increases probability

    # Combining factors (simplified)
    probability = 0.1 * elevation_factor * temperature_factor * biomass_factor
    probability = min(max(probability, 0.1), 0.3)  # Clamping the probability between 0.1 and 0.3

    return probability
def calc_time(node1, node2):
    # Using similar factors as in calc_prob
    elevation1 = dataset_value(image_elevation, *node1)['elevation']
    elevation2 = dataset_value(image_elevation, *node2)['elevation']
    temp1 = dataset_value(image_temperature, *node1)['LST_Day_1km']
    temp2 = dataset_value(image_temperature, *node2)['LST_Day_1km']
    biomass1 = dataset_value(image_biomass, *node1)['carbon_tonnes_per_ha']
    biomass2 = dataset_value(image_biomass, *node2)['carbon_tonnes_per_ha']

    # Time factors
    elevation_time_factor = 1.0 + (elevation2 - elevation1) / 100  # Higher elevation difference increases spread time
    temperature_time_factor = 1.0 - (temp2 - temp1) / 300  # Higher temperature decreases spread time
    biomass_time_factor = 1.0 + (biomass2 - biomass1) / 10  # Higher biomass increases spread time

    # Combining factors (simplified)
    time = 1 * elevation_time_factor * temperature_time_factor * biomass_time_factor
    time = max(time, 1)  # Ensuring time is at least 1

    return time
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
