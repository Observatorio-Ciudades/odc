import csv
import json
import os
from io import StringIO

import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
import h3
from shapely.geometry import Polygon, MultiLineString, Point, LineString

import shutil

from . import utils



def download_graph(polygon, city, network_type="walk", save=True):
    """
    Download a graph from a bounding box, and saves it to disk

    Arguments:
            polygon (polygon): polygon to use as boundary to download the network
            city (str): string with the name of the city
            network_type (str): String with the type of network to download (drive, walk, bike, all_private, all) for more details see OSMnx documentation
            save (bool): Save the graph to disk or not (default: {True})

    Returns:
            nx.MultiDiGraph
    """
    try:
        G = ox.load_graphml(
            "../data/raw/network_{}_{}.graphml".format(city, network_type)
        )
        utils.log(f"{city} retrived graph")
        return G
    except:
        utils.log(f"{city} graph not in data/raw")
        G = ox.graph_from_polygon(
            polygon,
            network_type=network_type,
            simplify=True,
            retain_all=False,
            truncate_by_edge=False,
        )
        utils.log("downloaded")
        if save:
            ox.save_graphml(
                G, filename="raw/network_{}_{}.graphml".format(city, network_type)
            )
        return G


def df_to_geodf(df, x, y, crs):
    """
    Create a GeoDataFrame from a pandas DataFrame
    Arguments:
            df (pandas.DataFrame): pandas data frame with lat, lon or x, y, columns
            x (str): Name of the column that contains the x or Longitud values
            y (str): Name of the column that contains the y or Latitud values
            crs (dict): Coordinate reference system to use
    Returns:
            geopandas.GeoDataFrame: GeoDataFrame with Points as geometry
    """
    df["y"] = df[y].astype(float)
    df["x"] = df[x].astype(float)
    geometry = [Point(xy) for xy in zip(df.x, df.y)]
    return gpd.GeoDataFrame(df, crs=crs, geometry=geometry)



def convert_type(df, data_dict):
    """
    Converts columns from DataFrame to specified data type

    Arguments:
        df (pandas.DataFrame): DataFrame containing all columns
        data_dict (dict): Dictionary with the desiered data type as a
                                key {string, integer, float} and a list of columns.
                                For example: {'string':[column1,column2],'integer':[column3,column4]}
    Returns:
        df (pandas.DataFrame): DataFrame with converted data types for columns
    """
    for d in data_dict:
        if d == "string":
            for c in data_dict[d]:
                df[c] = df[c].astype("str")
        else:
            for c in data_dict[d]:
                df[c] = pd.to_numeric(df[c], downcast=d, errors="ignore")

    return df


def delete_files_from_folder(delete_dir):
    """
    The delete_files_from_folder function deletes all files from a given directory.
    Arguments:
    delete_dir (str): Specify the directory where the files are to be deleted
    """

    for filename in os.listdir(delete_dir):
        file_path = os.path.join(delete_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            utils.log('Failed to delete %s. Reason: %s' % (file_path, e))



def download_osmnx_network(aoi, how='from_polygon', network_type='all_private'):
    """Download OSMnx graph, nodes and edges according to a GeoDataFrame area of interest.
       Based on Script07-download_osmnx.py located in database.

    Args:
        aoi (geopandas.GeoDataFrame): GeoDataFrame polygon boundary for the area of interest.
        how (str, optional): Defines the OSMnx function to be used. "from_polygon" will call osmnx.graph.graph_from_polygon,
                                 while "from_bbox" will call osmnx.features.features_from_bbox. No other choices are accepted in this function,
                                 for more details see OSMnx documentation.
        network_type (str, optional): String with the type of network to download (drive, walk, bike, all_private, all) for more details see OSMnx documentation.
                                        Defaults to 'all_private'.

    Returns:
        G (networkx.MultiDiGraph): Graph with edges and nodes within boundaries
		nodes (geopandas.GeoDataFrame): GeoDataFrame for nodes within boundaries
		edges (geopandas.GeoDataFrame): GeoDataFrame for edges within boundaries
    """

    # Set crs of area of interest
    aoi = aoi.to_crs("EPSG:4326")

    if how == 'from_bbox':
        # Read area of interest as a polygon geometry
        poly = aoi.geometry
        # Extracts coordinates from polygon as DataFrame
        coord_val = poly.bounds
        # Gets coordinates for bounding box
        n = coord_val.maxy.max()
        s = coord_val.miny.min()
        e = coord_val.maxx.max()
        w = coord_val.minx.min()
        utils.log(f"Extracted min and max coordinates from the municipality. Polygon N:{round(n,5)}, S:{round(s,5)}, E{round(e,5)}, W{round(w,5)}.")

        # Downloads OSMnx graph from bounding box
        G = ox.graph_from_bbox(n, s, e, w,
                               network_type=network_type,
                               simplify=True,
                               retain_all=False,
                               truncate_by_edge=False)
        utils.log("Created OSMnx graph from bounding box.")

    elif how == 'from_polygon':
        # Downloads OSMnx graph from bounding box
        G = ox.graph_from_polygon(aoi.unary_union,
                                  network_type=network_type,
                                  simplify=True,
                                  retain_all=False,
                                  truncate_by_edge=False)
        utils.log("Created OSMnx graph from bounding polygon.")

    else:
        utils.log
        ("Invalid argument 'how'.")

    #Transforms graph to nodes and edges Geodataframe
    nodes, edges = ox.graph_to_gdfs(G)
    #Resets index to access osmid as a column
    nodes.reset_index(inplace=True)
    #Resets index to acces u and v as columns
    edges.reset_index(inplace=True)
    print(f"Converted OSMnx graph to {len(nodes)} nodes and {len(edges)} edges GeoDataFrame.")

    # Defines columns of interest for nodes and edges
    nodes_columns = ["osmid", "x", "y", "street_count", "geometry"]
    edges_columns = [
        "osmid",
        "v",
        "u",
        "key",
        "oneway",
        "lanes",
        "name",
        "highway",
        "maxspeed",
        "length",
        "geometry",
        "bridge",
        "ref",
        "junction",
        "tunnel",
        "access",
        "width",
        "service",
    ]
    # if column doesn't exist it creates it as nan
    for c in nodes_columns:
        if c not in nodes.columns:
            nodes[c] = np.nan
            print(f"Added column {c} for nodes.")
    for c in edges_columns:
        if c not in edges.columns:
            edges[c] = np.nan
            print(f"Added column {c} for edges.")
    # Filters GeoDataFrames for relevant columns
    nodes = nodes[nodes_columns]
    edges = edges[edges_columns]
    print("Filtered columns.")

    # Converts columns with lists to strings to allow saving to local and further processes.
    for col in nodes.columns:
        if any(isinstance(val, list) for val in nodes[col]):
            nodes[col] = nodes[col].astype('string')
            print(f"Column: {col} in nodes gdf, has a list in it, the column data was converted to string.")
    for col in edges.columns:
        if any(isinstance(val, list) for val in edges[col]):
            edges[col] = edges[col].astype('string')
            print(f"Column: {col} in nodes gdf, has a list in it, the column data was converted to string.")

    # Final format
    nodes = nodes.set_crs("EPSG:4326")
    edges = edges.set_crs("EPSG:4326")
    nodes = nodes.set_index('osmid')
    edges = edges.set_index(["u", "v", "key"])

    return G, nodes, edges


def create_hexgrid(polygon, hex_res, geometry_col='geometry'):
    """
	Takes in a geopandas geodataframe, the desired resolution, the specified geometry column and some map parameters to create a hexagon grid (and potentially plot the hexgrid

	Arguments:
		polygon (geopandas.GeoDataFrame): geoDataFrame to be used
		hex_res (int): Resolution to use

	Keyword Arguments:
		geometry_col (str): column in the geoDataFrame that contains the geometry (default: {'geometry'})

	Returns:
		all_polys (geopandas.GeoDataFrame): geoDataFrame with the hexbins according to resolution and EPSG:4326
	"""

    #multiploygon to polygon
    polygons = polygon[geometry_col].explode(index_parts=True)

    polygons = polygons.reset_index(drop=True)

    all_polys = gpd.GeoDataFrame()

    for p in range(len(polygons)):

        #create hex grid from GeoDataFrame
        #for i in range(len(polygons[p])):
        dict_poly = polygons[p].__geo_interface__
        hexs = h3.polyfill(dict_poly, hex_res, geo_json_conformant = True)
        polygonise = lambda hex_id: Polygon(
                                    h3.h3_to_geo_boundary(
                                        hex_id, geo_json=True)
                                        )

        poly_tmp = gpd.GeoSeries(list(map(polygonise, hexs)), \
                                                index=hexs, \
                                                crs="EPSG:4326" \
                                                )
        gdf_tmp = gpd.GeoDataFrame(poly_tmp.reset_index()).rename(columns={'index':f'hex_id_{hex_res}',0:geometry_col})

        all_polys = pd.concat([all_polys, gdf_tmp],
        ignore_index = True, axis = 0)

    all_polys = all_polys.drop_duplicates()
    all_polys = all_polys.set_geometry('geometry')
    all_polys.set_crs("EPSG:4326")

    return all_polys
