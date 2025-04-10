![PyPI](https://img.shields.io/pypi/v/odc)
![Downloads](https://img.shields.io/pypi/dm/odc)

# OdC

The Observatory of Cities (OdC) is an urban science laboratory which seeks to present itself as an innovation platform focused on helping to collect, process, analyze and visualize spatial data related to urban dynamics. 

This package streamlines spatial analysis processes by integrating various libraries and developing first-party functions. It is designed as a low-code solution for spatial analysis.

------------

# Installing the package

```python
pip install odc
```
------------
# Basic usage example

## Proximity analysis


```python
import geopandas as gpd
import odc
```

```python

# Step 1: Define an area of interest (AOI)
print("\n--- Defining Area of Interest (AOI) ---")
# Load the boundary GeoJSON file in EPSG:4326 and ensure it has a valid CRS
aoi_gdf = gpd.read_file("../path/to/polygon/{}.geojson")
if aoi_gdf.crs is None:
    aoi_gdf = aoi_gdf.set_crs("EPSG:4326")  # Assign CRS if missing

# Step 2: Download the network using odc wrapper function
print("\n--- Creating OSMnx Network ---")
# Download osmnx network 
G, nodes, edges = odc.download_osmnx_network(aoi_gdf, how='from_bbox')

print(f"Downloaded {len(nodes)} nodes and {len(edges)} edges.")

# Step 3: Load points of interest (POIs) in EPSG:4326 and ensure it has a valid CRS
pois = gpd.read_file("../path/to/pois/{}.geojson")

if pois.crs is None:
    pois = pois.set_crs("EPSG:4326")  # Ensure POIs have a valid CRS

# Step 4: Analyze walking time to nearest and total number
# of POI accessible within a given time using a pois_time function
print("\n--- Calculating Time to POIs ---")
walking_speed = 5  # Walking speed in km/h
proximity_measure = "time_min"
pois_name = "example_poi"  # Update if necessary for real POIs
count_pois = (True, 10) # Count the number of pois within a certain distance or time

# Run pois_time function
nodes_with_time = odc.pois_time(
    G = G,
    nodes = nodes,
    edges = edges,
    pois = pois,
    poi_name = pois_name,
    prox_measure = proximity_measure,
    walking_speed = walking_speed,
    count_pois = count_pois,
    projected_crs = "EPSG:6372",
)

# Step 5: Save results to a file
print("\n--- Saving Results ---")
output_path = f"../data/to/output/nodes_with_{pois_name}_time.geojson"
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

## Raster image analysis
------------

```python
# Import the module and required libraries
# Example: loading a shapefile that represents the geometry of a city or region.
from odc.raster import download_raster_from_pc
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
