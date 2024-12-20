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

```
pip install odc
```

```python

    import osmnx as ox
    import geopandas as gpd
    import numpy as np
    import odc
    
    # Example: Using functions to download and process an OSMnx network, and analyze POIs
    def main():
        # Step 1: Define an area of interest (AOI)
        # Create a GeoDataFrame representing the boundary of the city (for example, using a polygon)
        polygon = gpd.GeoSeries.from_file("../data/raw/city_boundary.geojson").unary_union
        city = "ExampleCity"
    
        # Step 2: Create the network using the create_osmnx_network function
        print("\n--- Creating OSMnx Network ---")
        G, nodes, edges = odc.create_osmnx_network(
            aoi=gpd.GeoDataFrame(geometry=[polygon]),
            how="from_polygon",
            network_type="walk"
        )
    
        # Step 3: Download or retrieve the graph
        print("\n--- Downloading Graph ---")
        graph = odc.download_graph(polygon, city, network_type="walk", save=True)
    
        # Step 4: Define points of interest (POIs)
        # Load POIs as a GeoDataFrame (example file with POIs for the city)
        pois = gpd.read_file("../data/raw/pois.geojson")
    
        # Step 5: Analyze time to nearest POI using pois_time
        print("\n--- Calculating Time to POIs ---")
        walking_speed = walking_speed  # in km/hr
        proximity_measure = "time_min"
        pois_name = "amenity_shop"
    
        nodes_with_time = odc.pois_time(
            G=graph,
            nodes=nodes,
            edges=edges,
            pois=pois,
            pois_name=pois_name,
            prox_measure=proximity_measure,
            walking_speed=walking_speed,
            count_pois=(True, 10),  # Count POIs within a 10-minute range
            projected_crs="EPSG:6372",
            preprocessed_nearest=(False, "")
        )
    
        # Save results to a file for further analysis
        print("\n--- Saving Results ---")
        nodes_with_time.to_file(f"../data/processed/nodes_with_{poi_name}_time.geojson", driver="GeoJSON")
    
    
    if __name__ == "__main__":
        main()

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
