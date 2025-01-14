![PyPI](https://img.shields.io/pypi/v/odc)
![Downloads](https://img.shields.io/pypi/dm/odc)

# ODC

The "Observatorio de Ciudades" (ODC) is a platform which seeks to present itself as an innovation platform focused on helping to collect, process, analyze and visualize data related to urban dynamics. 

This package streamlines spatial analysis processes by integrating various libraries and developing first-party functions. It is designed as a low-code solution for spatial analysis.

## Satrting a new project
------------

    git clone This repo

------------

# Proximity
------------

## Basic usage example
------------

```python
pip install odc
pip install osmnx
```

```python
import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import odc 
from shapely.geometry import Polygon
```

```python

# Step 1: Define an area of interest (AOI)
print("\n--- Defining Area of Interest (AOI) ---")
# Load the boundary GeoJSON file and ensure it has a valid CRS
aoi_gdf = gpd.read_file("../data/raw/bbox_{}.geojson")
if aoi_gdf.crs is None:
    aoi_gdf = aoi_gdf.set_crs("EPSG:4326")  # Assign CRS if missing

# Step 2: Create the network using graph_from_place
print("\n--- Creating OSMnx Network ---")
# Generate a walking network graph for New York City using OSMnx
G_poly = ox.graph.graph_from_polygon(polygon, network_type="network_type")

# Convert the graph to GeoDataFrames for nodes and edges
nodes, edges = ox.graph_to_gdfs(G)
print(f"Graph created with {len(nodes)} nodes and {len(edges)} edges.")

# Optional Step: Generate example POIs for testing (if POIs file is unavailable)
def generate_example_pois(n_pois=10):
    """Generate random points of interest (POIs) within the bounding box."""
    import random
    lats = [random.uniform(south, north) for _ in range(n_pois)]
    lons = [random.uniform(west, east) for _ in range(n_pois)]
    geometries = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    return gpd.GeoDataFrame({"name": [f"POI_{i}" for i in range(n_pois)]}, geometry=geometries, crs="EPSG:4326")

# The following line use example POIs
pois = generate_example_pois()
print(f"Generated {len(pois)} example POIs.")

# Step 3: Load points of interest (POIs)
# Uncomment if loading from a file instead
# pois = gpd.read_file("../data/raw/pois.geojson")
if pois.crs is None:
    pois = pois.set_crs("EPSG:4326")  # Ensure POIs have a valid CRS

# Step 4: Analyze time to nearest POI using a custom function
print("\n--- Calculating Time to POIs ---")
walking_speed = 5  # Walking speed in km/h
proximity_measure = "time_min"
pois_name = "example_poi"  # Update if necessary for real POIs

# Add missing logic for calculating time_min if necessary
if proximity_measure == "length":
    edges['time_min'] = (edges['length'] * 60) / (walking_speed * 1000)
else:
    # Fill missing time_min values if prox_measure is not length
    no_time = len(edges.loc[edges['time_min'].isna()]) if 'time_min' in edges.columns else len(edges)
    edges['time_min'] = edges.get('time_min', pd.Series(index=edges.index))  # Create if missing
    edges['time_min'].fillna((edges['length'] * 60) / (walking_speed * 1000), inplace=True)
    print(f"Calculated time for {no_time} edges with missing time data.")

# Run pois_time function
nodes_with_time = odc.pois_time(
    G = G,
    nodes = nodes,
    edges = edges,
    pois = pois,
    poi_name = pois_name,
    prox_measure = proximity_measure,
    walking_speed = walking_speed,
    count_pois = (True, 10),
    projected_crs = "EPSG:6372",
    preprocessed_nearest = (False, "")
)

# Use the dynamically named column
proximity_column = f"time_{pois_name}"  # e.g., "time_example_poi"

# Verify the column exists
if proximity_column not in nodes_with_time.columns:
    raise KeyError(f"The column '{proximity_column}' is missing from the nodes_with_time output.")

# Step 5: Save results to a file
print("\n--- Saving Results ---")
output_path = f"../data/processed/nodes_with_{pois_name}_time.geojson"
nodes_with_time.to_file(output_path, driver="GeoJSON")
print(f"Results saved to {output_path}")

```

------------
    
# Raster Analysis
------------

```
├── odc/                 
│   ├── data.py        <- Scripts to download or generate data
│   ├── raster.py      <- Principal functions for raster data treatment and analysis functions
│   ├── utils.py       <- Utilities and auxiliary functions
│
```

## Basic usage example
------------

```python
# Import the module and required libraries
# Example: loading a shapefile that represents the geometry of a city or region.
from raster import odc.download_raster_from_pc
import geopandas as gpd
import odc
```

```python
# Load your shapefile or GeoJSON file
gdf = gpd.read_file("path/to/your/shapefile.shp")

# Define analysis parameters
params = {
    "index_analysis": "NDVI",
    "city": "ExampleCity",
    "freq": "MS",
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "tmp_dir": "/path/to/tmp",
    band_name_dict = {
    "nir": [False],  # If the GSD (resolution) of this band is different, set to True.
    "red": [False],  # If the GSD (resolution) of this band is different, set to True.
    "eq": ['(nir-red)/(nir+red)'],  # Equation to calculate an index, such as NDVI.
    }


# Download and process raster data
df_result = odc.download_raster_from_pc(
    gdf=gdf, # GeoDataFrame
    index_analysis=params["index_analysis"], # Analysis index (ndvi, ndmi)
    city=params["city"], # Name of the city or region
    freq=params["freq"], # Analysis frequency ("MS" for monthly, "D" for daily)
    start_date=params["start_date"], # Start date (YYYY-MM-DD)
    end_date=params["end_date"], # Finish date
    tmp_dir=params["tmp_dir"], # Temporal dictionary for saving the processed data
    band_name_dict=params["band_name_dict"] # Dictionary with names of spectral bands
)

# Inspect the output
print(df_result.head())
```
