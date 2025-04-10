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

# Proximity analysis
## Basic usage example

```python
import geopandas as gpd
import odc
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
```

```python
# Example: Define an area of interest, download an OpenStreetMap network, analyze walking times to points of interest, and save the results as a GeoJSON file.
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

```python
# Step 6: Graph visualization
# Load the AOI, nodes with time, and POIs GeoJSON files (mocked paths for example)
aoi_path = "../path/to/polygon/aoi.geojson"
pois_path = "../path/to/pois/pois.geojson"
nodes_with_time_path = "../data/to/output/nodes_with_example_poi_time.geojson"

# Read the GeoJSON files
aoi_gdf = gpd.read_file(aoi_path)
pois_gdf = gpd.read_file(pois_path)
nodes_with_time_gdf = gpd.read_file(nodes_with_time_path)

# Plot the data
fig, ax = plt.subplots(figsize=(12, 8))
aoi_gdf.boundary.plot(ax=ax, color="black", linewidth=1, label="AOI Boundary")
nodes_with_time_gdf.plot(
    ax=ax,
    column="time_min",  # Assuming "time_min" is the time to POIs column
    cmap="viridis",
    legend=True,
    markersize=5,
    label="Nodes (Time to POI)"
)
pois_gdf.plot(ax=ax, color="red", markersize=20, label="POIs")

# Add labels and legend
plt.title("Walking Time to Points of Interest", fontsize=14)
plt.xlabel("Longitude", fontsize=12)
plt.ylabel("Latitude", fontsize=12)
plt.legend(loc="upper left", fontsize=10)
plt.grid(alpha=0.3)
plt.show()
```

------------
# Raster Analysis

## Basic usage example


```python

# Import the module and required libraries
# Example: loading a shapefile that represents the geometry of a city or region.
from odc.raster import download_raster_from_pc
import geopandas as gpd
import odc
import matplotlib.pyplot as plt
import seaborn sns
```

```python
# Step 1: Define the area of interest (AOI) as a GeoDataFrame
hex_gdf = gpd.read_file("path/to/{}.geojson")  # GeoJSON file containing the hexagonal grid

# Step 2: Set the parameters for raster data download and analysis
index_analysis = "NDVI"  # The index to be analyzed
city = "ExampleCity"  # The name of the city or area to analyze
freq = "M"  # Frequency of the analysis ("M" = monthly)
start_date = "YYYY-MM-DD"  # Start date for the analysis
end_date = "YYYY-MM-DD"  # End date for the analysis
tmp_dir = "/temporary/path"  # Temporary directory where raster data will be saved
satellite = "sentinel-2-l2a"  # Satellite used to download imagery
projection_crs = "EPSG:6372"  # Projection for processing the data
compute_unavailable_dates = True  # Whether to compute missing months

# Dictionary with spectral band names (example for Sentinel-2)
band_name_dict= {
        "nir": [False],  # Near-infrared band
        "red": [False],  # Red band
        "eq": ["(nir-red)/(nir+red)"],  # NDVI formula
    }

# Step 3: Download and process the raster data
print("Downloading and processing raster data...")
df_len = download_raster_from_pc(
    gdf=hex_gdf,
    index_analysis=index_analysis,
    city=city,
    freq=freq,
    start_date=start_date,
    end_date=end_date,
    tmp_dir=tmp_dir,
    band_name_dict=band_name_dict,
    satellite=satellite,
    projection_crs=projection_crs,
    compute_unavailable_dates=compute_unavailable_dates
)

# Step 4: Visualize the summary of downloaded data
print("Summary of downloaded data:")
print(df_len.head())
```

```python
# Step 6: Graph visualization
# Visualization: Distribution of Data IDs over Months
plt.figure(figsize=(10, 6))
sns.barplot(x="month", y="data_id", data=df_len, palette="viridis")
plt.title("Data ID Distribution Across Months", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Data ID", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualization: Index Values
plt.figure(figsize=(10, 6))
sns.heatmap(df.pivot_table(index="year", columns="month", values="interpolate"),
            annot=True, cmap="coolwarm", cbar=True, linewidths=0.5)
plt.title("Index Values (Heatmap)", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Year", fontsize=12)
plt.show()
```
