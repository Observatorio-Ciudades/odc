################################################################################
# Module: Data
# General data management functions
# updated: 21/10/2025
################################################################################

import shapely
import geopandas as gpd
import osmnx as ox
import pandas as pd
import numpy as np
import h3
import networkx as nx

from pathlib import Path
from typing import Union, List, Dict, Optional

import shutil

from . import utils


def convert_column_types(
    df: pd.DataFrame,
    type_mapping: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Convert DataFrame columns to specified data types.

    Parameters:
        df: Input DataFrame
        type_mapping: Dictionary mapping data types to column lists
                     Supported types: 'string', 'integer', 'float', 'boolean', 'datetime'
                     Example: {'string': ['col1', 'col2'], 'integer': ['col3']}

    Returns:
        DataFrame with converted column types

    Raises:
        ValueError: If unsupported data type is specified
        KeyError: If specified column doesn't exist in DataFrame
    """

    if df is None or df.empty:
        raise ValueError("Input DataFrame cannot be None or empty")

    if not isinstance(type_mapping, dict):
        raise ValueError("type_mapping must be a dictionary")

    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    supported_types = {'string', 'integer', 'float', 'boolean', 'datetime'}

    for data_type, columns in type_mapping.items():
        if data_type not in supported_types:
            raise ValueError(f"Unsupported data type: {data_type}. "
                           f"Supported types: {supported_types}")

        if not isinstance(columns, list):
            raise ValueError(f"Column list for type '{data_type}' must be a list")

        for col in columns:
            if col not in df_copy.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Convert columns based on type
        for col in columns:
            try:
                if data_type == "string":
                    df_copy[col] = df_copy[col].astype("str")
                elif data_type == "boolean":
                    df_copy[col] = df_copy[col].astype("bool")
                elif data_type == "datetime":
                    df_copy[col] = pd.to_datetime(df_copy[col])
                else:  # integer or float
                    df_copy[col] = pd.to_numeric(df[col])
                    if data_type.lower() == 'integer':
                        df_copy[col] = df_copy[col].astype('int32')
                    else:
                        df_copy[col] =  df_copy[col].astype('float64')

                utils.log(f"Converted column '{col}' to {data_type}")

            except Exception as e:
                utils.log(f"Failed to convert column '{col}' to {data_type}: {e}")
                raise

    utils.log(f"Successfully converted columns for {len(type_mapping)} data types")
    return df_copy


def clear_directory(
    directory_path: Union[str, Path],
) -> Dict[str, int]:
    """
    Delete all files and subdirectories from a given directory.

    Parameters:
        directory_path: Path to the directory to clear

    Returns:
        Dictionary with counts of files and directories processed

    Raises:
        ValueError: If directory doesn't exist
    """

    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    stats = {'files': 0, 'directories': 0, 'errors': 0}

    try:
        for item in directory_path.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                    stats['files'] += 1
                    utils.log(f"Deleted file: {item}")
                elif item.is_dir():
                    shutil.rmtree(item)
                    stats['directories'] += 1
                    utils.log(f"Deleted directory: {item}")

            except Exception as e:
                stats['errors'] += 1
                utils.log(f"Failed to delete {item}: {e}")

    except PermissionError as e:
        utils.log(f"Permission denied accessing directory: {e}")
        raise

    return stats


def download_osm_network(
    area_of_interest: gpd.GeoDataFrame,
    method: str = 'from_polygon',
    network_type: str = 'all_private',
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Download OSM network data and return graph, nodes, and edges.

    Parameters:
        area_of_interest: GeoDataFrame defining the area boundary in EPSG:4326
        method: Method for downloading ('from_polygon' or 'from_bbox')
        network_type: Type of network to download
                     ('drive', 'walk', 'bike', 'all_private', 'all')

    Returns:
        Tuple containing:
        - NetworkX MultiDiGraph with network topology
        - GeoDataFrame with network nodes
        - GeoDataFrame with network edges

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If network download fails
    """

    # Validate inputs
    if area_of_interest is None or area_of_interest.empty:
        raise ValueError("Area of interest cannot be None or empty")

    valid_methods = {'from_polygon', 'from_bbox'}
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")

    valid_network_types = {'drive', 'walk', 'bike', 'all_private', 'all'}
    if network_type not in valid_network_types:
        raise ValueError(f"Invalid network_type '{network_type}'. "
                        f"Must be one of {valid_network_types}")
    
    # Convert to EPSG:4326 for graph download
    if area_of_interest.crs != "EPSG:4326":
        area_of_interest = area_of_interest.to_crs("EPSG:4326")

    try:
        if method == 'from_bbox':
            # Extract bounding box coordinates
            bounds = area_of_interest.total_bounds
            west, south, east, north = bounds

            utils.log(f"Downloading network from bbox: N={north:.5f}, S={south:.5f}, "
                       f"E={east:.5f}, W={west:.5f}")

            graph = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
                simplify=True,
                retain_all=False,
                truncate_by_edge=False
            )

        else:  # from_polygon
            utils.log("Downloading network from polygon boundary")

            graph = ox.graph_from_polygon(
                area_of_interest.union_all(),
                network_type=network_type,
                simplify=True,
                retain_all=False,
                truncate_by_edge=False
            )

        utils.log(f"Successfully downloaded {network_type} network")

    except Exception as e:
        utils.log(f"Failed to download network: {e}")
        raise RuntimeError(f"Network download failed: {e}")

    # Convert graph to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)
    nodes = nodes.reset_index()
    edges = edges.reset_index()

    utils.log(f"Converted graph to {len(nodes)} nodes and {len(edges)} edges")

    # Define required columns
    required_node_columns = ["osmid", "x", "y", "street_count", "geometry"]
    required_edge_columns = [
        "osmid", "u", "v", "key", "oneway", "lanes", "name", "highway",
        "maxspeed", "length", "geometry", "bridge", "ref", "junction",
        "tunnel", "access", "width", "service"
    ]

    # Add missing columns
    for col in required_node_columns:
        if col not in nodes.columns:
            nodes[col] = np.nan
            utils.log(f"Added missing column '{col}' to nodes")

    for col in required_edge_columns:
        if col not in edges.columns:
            edges[col] = np.nan
            utils.log(f"Added missing column '{col}' to edges")

    # Filter to required columns
    nodes = nodes[required_node_columns]
    edges = edges[required_edge_columns]

    # Handle list columns by converting to strings
    for gdf, name in [(nodes, 'nodes'), (edges, 'edges')]:
        for col in gdf.columns:
            if col == 'geometry':
                continue
            if any(isinstance(val, list) for val in gdf[col].dropna()):
                gdf[col] = gdf[col].astype('string')
                utils.log(f"Converted list column '{col}' to string in {name}")

    # Set final format
    nodes = nodes.set_crs('EPSG:4326').set_index('osmid')
    edges = edges.set_crs('EPSG:4326').set_index(["u", "v", "key"])

    utils.log("Network processing completed successfully")

    return graph, nodes, edges


def create_hexagonal_grid(
    geometry: gpd.GeoDataFrame,
    resolution: int,
    geometry_column: str = 'geometry'
) -> gpd.GeoDataFrame:
    """
    Create a hexagonal grid covering the input geometry using H3 hexagons.

    Parameters:
        geometry: GeoDataFrame containing the area to be covered
        resolution: H3 resolution level (0-15, higher = smaller hexagons)

    Returns:
        GeoDataFrame with hexagonal grid covering the input geometry

    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If hexagon creation fails
    """

    # Validate inputs
    if geometry is None or geometry.empty:
        raise ValueError("Input geometry cannot be None or empty")

    # Ensure WGS84 CRS for H3 compatibility
    if geometry.crs != "EPSG:4326":
        geometry = geometry.to_crs("EPSG:4326")
        utils.log("Converted geometry to EPSG:4326 for H3 compatibility")

    # Ensure resolution is an integer between 0 and 15
    if not isinstance(resolution, int) or not 0 <= resolution <= 15:
        raise ValueError(f"resolution must be an integer between 0 and 15, got {resolution}")

    # Handle MultiPolygon by exploding to individual polygons
    exploded_geoms = geometry[geometry_column].explode(ignore_index=True)
    exploded_geoms = exploded_geoms.reset_index(drop=True)
    utils.log(exploded_geoms)

    # Collect all hexagons
    all_hexagons = []

    for idx, geom in enumerate(exploded_geoms):
        try:
            # Convert to GeoJSON format for H3
            geom_dict = geom.__geo_interface__
            geom_dict = h3.geo_to_h3shape(geom_dict)
            utils.log(geom_dict)

            # Get hexagon IDs covering this polygon
            hex_ids = h3.polygon_to_cells(geom_dict, resolution)
            utils.log(f"Generated {len(hex_ids)} hexagons for polygon {idx}")

            # Convert hex IDs to polygons
            for hex_id in hex_ids:
                hex_boundary = h3.cell_to_boundary(hex_id)
                hex_boundary = [(lon,lat) for lat, lon in hex_boundary]
                hex_polygon = shapely.Polygon(hex_boundary)
                all_hexagons.append({
                    f'hex_id_{resolution}': hex_id,
                    'geometry': hex_polygon
                })

        except Exception as e:
            utils.log(f"Failed to create hexagons for polygon {idx}: {e}")
            raise RuntimeError(f"Hexagon creation failed: {e}")

    if not all_hexagons:
        raise RuntimeError("No hexagons were created")

    # Create GeoDataFrame
    try:
        result_gdf = gpd.GeoDataFrame(all_hexagons, crs="EPSG:4326")
        result_gdf = result_gdf.set_geometry('geometry')

        # Remove duplicate hexagons
        result_gdf = result_gdf.drop_duplicates(subset=[f'hex_id_{resolution}'])

        return result_gdf

    except Exception as e:
        utils.log(f"Failed to create final GeoDataFrame: {e}")
        raise RuntimeError(f"GeoDataFrame creation failed: {e}")
