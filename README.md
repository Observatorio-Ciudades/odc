![PyPI](https://img.shields.io/pypi/v/odc)
![Downloads](https://img.shields.io/pypi/dm/odc)

# ODC

The "Observatorio de Ciudades" (ODC) is a platform which seeks to present itself as an innovation platform focused on helping to collect, process, analyze and visualize data related to urban dynamics. 

This package streamlines spatial analysis processes by integrating various libraries and developing first-party functions. It is designed as a low-code solution for spatial analysis.

## Satrting a new project
------------

    git clone This repo
    
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
```
pip install odc
```

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
