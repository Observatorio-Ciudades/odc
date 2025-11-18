################################################################################
# Module: Network
# Set of network processing and creation functions
# updated: 03/10/2025
################################################################################

from typing import Callable, List, Optional, Tuple, Union

import geopandas as gpd
import igraph as ig
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from geopandas import GeoDataFrame
from networkx import MultiDiGraph
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.neighbors import BallTree

from .utils import *


def nearest_nodes(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    X: Union[float, List[float]],
    Y: Union[float, List[float]],
    return_distance: bool = False,
) -> Union[int, List[int], Tuple[Union[int, List[int]], Union[float, List[float]]]]:
    """
    Find the nearest node to a point or to each of several points.

    Adapted from OSMnx to search for nodes without requiring a Graph object.
    Uses k-d tree for projected coordinates or ball tree for unprojected coordinates
    to efficiently find the closest network nodes to given coordinate points.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph containing CRS information for coordinate system detection.
        Used to determine whether to use euclidean or haversine distance.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame containing network nodes with geometry column.
        Must have 'x' and 'y' coordinate columns for unprojected data.
    X : float or list of float
        Points' x (longitude) coordinates in same CRS as graph.
        Cannot contain null values.
    Y : float or list of float
        Points' y (latitude) coordinates in same CRS as graph.
        Cannot contain null values.
    return_distance : bool, default False
        Whether to return distances between points and nearest nodes.

    Returns
    -------
    int, list of int, or tuple
        If return_dist=False: Nearest node IDs (int for single point, list for multiple).
        If return_dist=True: Tuple of (node_ids, distances) with same structure.

    Raises
    ------
    ValueError
        If X or Y contain null values.
    ImportError
        If scipy is not installed (for projected graphs) or
        scikit-learn is not installed (for unprojected graphs).
    """

    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        # make coordinates arrays if user passed non-iterable values
        is_scalar = True
        X = np.array([X])
        Y = np.array([Y])

    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")

    if is_projected(G.graph["crs"]):
        # if projected, use k-d tree for euclidean nearest-neighbor search
        if cKDTree is None:  # pragma: no cover
            raise ImportError("scipy must be installed to search a projected graph")
        dist, pos = cKDTree(nodes).query(np.array([X, Y]).T, k=1)
        nn = nodes.index[pos]

    else:
        # if unprojected, use ball tree for haversine nearest-neighbor search
        if BallTree is None:  # pragma: no cover
            raise ImportError(
                "scikit-learn must be installed to search an unprojected graph"
            )
        # haversine requires lat, lng coords in radians
        nodes_rad = np.deg2rad(nodes[["y", "x"]])
        points_rad = np.deg2rad(np.array([Y, X]).T)
        dist, pos = BallTree(nodes_rad, metric="haversine").query(points_rad, k=1)
        EARTH_RADIUS_M = 6_371_000
        dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters
        nn = nodes.index[pos[:, 0]]

    # convert results to correct types for return
    nn = nn.tolist()
    dist = dist.tolist()
    if is_scalar:
        nn = nn[0]
        dist = dist[0]

    if return_distance:
        return nn, dist
    else:
        return nn


def find_nearest_point_to_node(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    gdf: GeoDataFrame,
    return_distance: bool = False,
) -> GeoDataFrame:
    """
    Find the nearest graph nodes to points in a GeoDataFrame.

    For each point geometry in the input GeoDataFrame, identifies the closest
    network node and adds this information as a new column. Optionally includes
    the distance to the nearest node.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph created with OSMnx containing CRS information.
        Used for coordinate system detection in distance calculations.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes, must have 'osmid' as index.
        Contains the candidate nodes for nearest neighbor search.
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with Point geometries to find nearest nodes for.
        All geometries must be Point type.
    return_distance : bool, default False
        Whether to include distance to nearest node in output.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of input GeoDataFrame with additional columns:
        - 'osmid': ID of nearest network node
        - 'distance_node': Distance to nearest node (if return_distance=True)
    """

    gdf = gdf.copy()

    if "osmid" in nodes.columns:
        nodes = nodes.set_index("osmid")

    osmnx_tuple = nearest_nodes(
        G,
        nodes,
        gdf.geometry.x.values,
        gdf.geometry.y.values,
        return_distance=return_distance,
    )

    if return_distance:
        gdf["osmid"] = osmnx_tuple[0]
        gdf["distance_node"] = osmnx_tuple[1]
    else:
        gdf["osmid"] = osmnx_tuple
    return gdf


def to_igraph(
    nodes: GeoDataFrame, edges: GeoDataFrame, weight: str = "length"
) -> Tuple[ig.Graph, np.ndarray, dict]:
    """
    Convert NetworkX-style graph data to igraph format for efficient calculations.

    Transforms node and edge GeoDataFrames into an igraph.Graph object with
    corresponding weight array and node mapping dictionary. This conversion
    enables faster shortest path calculations using igraph's optimized algorithms.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes having 'osmid' column/index.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges having 'u' and 'v' columns.
        Must contain the specified weight column.
    weight : str, default 'length'
        Name of edge attribute to use as weights in shortest path calculations.
        Common values: 'length', 'time_min', 'travel_time'.

    Returns
    -------
    tuple of (igraph.Graph, numpy.ndarray, dict)
        - graph: igraph.Graph object with same topology as input
        - weights: Array of edge weights for shortest path calculations
        - node_mapping: Dictionary mapping original osmid to igraph node indices
    """

    if "osmid" in nodes.columns:
        nodes.set_index(["osmid"], inplace=True)

    if ("u" in edges.columns) and ("v" in edges.columns):
        edges.set_index(["u", "v"], inplace=True)

    node_mapping = dict(zip(nodes.index.values, range(len(nodes))))
    g = ig.Graph(
        len(nodes),
        [(node_mapping[i[0]], node_mapping[i[1]]) for i in edges.index.values],
    )
    weights = np.array([float(e) for e in edges[weight]])

    return g, weights, node_mapping


def get_seeds(gdf: GeoDataFrame, node_mapping: dict, column_name: str) -> np.ndarray:
    """
    Generate seed array for Voronoi diagram calculation from mapped nodes.

    Converts node identifiers from a GeoDataFrame into igraph node indices
    using the provided mapping. These seeds serve as generator points for
    network-based Voronoi tessellation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing points of interest with nearest node assignments.
        Must contain the specified column with node identifiers.
    node_mapping : dict
        Dictionary mapping original node IDs to igraph node indices.
        Typically created by the to_igraph function.
    column_name : str
        Name of column containing node identifiers to convert to seeds.
        Usually 'osmid' or 'nearest'.

    Returns
    -------
    numpy.ndarray
        Array of igraph node indices corresponding to seed locations.
        Used as input for Voronoi diagram calculations.
    """

    # Get the seed to calculate shortest paths
    return np.array(list([node_mapping[i] for i in gdf[column_name]]))


def voronoi_cpu(g: ig.Graph, weights: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """
    Calculate network-based Voronoi diagram using CPU-optimized algorithms.

    Computes Voronoi tessellation on a graph where each node is assigned to
    the nearest seed (generator point) based on shortest path distances.
    Uses Dijkstra's algorithm for accurate distance calculations on weighted networks.

    Parameters
    ----------
    g : igraph.Graph
        Graph object representing the network topology.
        Must be connected for meaningful results.
    weights : numpy.ndarray
        Array of edge weights for shortest path calculations.
        Length must match number of edges in graph.
    seeds : numpy.ndarray
        Array of node indices serving as Voronoi generators.
        All indices must be valid nodes in the graph.

    Returns
    -------
    numpy.ndarray
        Array of length equal to number of nodes, where each value indicates
        which seed (by index in seeds array) the corresponding node belongs to.
    """

    return seeds[
        np.array(g.distances(seeds, weights=weights, algorithm="dijkstra")).argmin(
            axis=0
        )
    ]


def get_distances(
    g: ig.Graph,
    seeds: np.ndarray,
    weights: np.ndarray,
    voronoi_assignment: np.ndarray,
    get_nearest_poi: bool = False,
    count_pois: Tuple[bool, float] = (False, 0),
) -> Union[List[float], Tuple[List[float], List[int]], Tuple[List[float], List[int]]]:
    """
    Calculate shortest path distances from each node to its assigned seed.

    Computes the minimum travel distance/time from each network node to its
    nearest point of interest (seed). Optionally provides additional metrics
    like nearest POI identification and proximity counts.

    Parameters
    ----------
    g : igraph.Graph
        Graph object for shortest path calculations.
    seeds : numpy.ndarray
        Array of seed node indices (points of interest).
    weights : numpy.ndarray
        Edge weights for distance/time calculations.
    voronoi_assignment : numpy.ndarray
        Assignment of each node to its nearest seed (from voronoi_cpu).
    get_nearest_poi : bool, default False
        Whether to return indices of nearest POIs for each node.
    count_pois : tuple of (bool, float), default (False, 0)
        Whether to count POIs within specified distance threshold.
        Format: (enable_counting, distance_threshold).

    Returns
    -------
    list or tuple of lists
        - If only distances: List of minimum distances to nearest seeds
        - If get_nearest_poi=True: Tuple of (distances, nearest_poi_indices)
        - If count_pois[0]=True: Includes count of nearby POIs
    """

    shortest_paths = np.array(g.distances(seeds, weights=weights, algorithm="dijkstra"))
    distances = [np.min(shortest_paths[:, i]) for i in range(len(voronoi_assignment))]

    if get_nearest_poi:
        nearest_poi_idx = [
            np.argmin(shortest_paths[:, i]) for i in range(len(voronoi_assignment))
        ]

    if count_pois[0]:
        near_count = [
            len(np.where(shortest_paths[:, i] <= count_pois[1])[0])
            for i in range(len(voronoi_assignment))
        ]

    # Function output options
    if get_nearest_poi and count_pois[0]:
        return distances, nearest_poi_idx, near_count
    elif get_nearest_poi:
        return distances, nearest_poi_idx
    elif count_pois[0]:
        return distances, near_count
    else:
        return distances


def calculate_distance_nearest_poi(
    gdf_f: GeoDataFrame,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    poi_name: str,
    column_name: str,
    weight: str = "length",
    get_nearest_poi: Tuple[bool, str] = (False, "poi_id_column"),
    count_pois: Tuple[bool, float] = (False, 0),
    max_distance: Tuple[float, str] = (0, "distance_node"),
) -> GeoDataFrame:
    """
    Calculate network distances from all nodes to nearest points of interest.

    Performs comprehensive proximity analysis by computing shortest path distances
    from every network node to the closest POI. Supports various analysis options
    including POI counting within thresholds and nearest POI identification.

    Parameters
    ----------
    gdf_f : geopandas.GeoDataFrame
        GeoDataFrame containing points of interest with assigned nearest nodes.
        Must have the column specified in column_name.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes for analysis.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges containing weight information.
    poi_name : str
        Name identifier for the poi type (e.g., 'pharmacy', 'school').
        Used in output column naming.
    column_name : str
        Column name containing nearest node assignments for POIs.
        Typically 'osmid' from find_nearest_point_to_node function.
    weight : str, default 'length'
        Edge attribute to use for distance calculations.
        Common values: 'length', 'time_min', 'travel_time'.
    get_nearest_poi : tuple of (bool, str), default (False, 'poi_id_column')
        Whether to identify nearest POI and which column contains POI IDs.
    count_pois : tuple of (bool, float), default (False, 0)
        Whether to count POIs within distance threshold and the threshold value.
    max_distance : tuple of (float, str), default (0, 'distance_node')
        Maximum distance filter for POI inclusion and column name.

    Returns
    -------
    geopandas.GeoDataFrame
        Nodes GeoDataFrame with additional columns:
        - f'dist_{poi_name}': Distance to nearest amenity
        - f'nearest_{poi_name}': ID of nearest amenity (if requested)
        - f'{poi_name}_{threshold}min': Count of amenities within threshold (if requested)
    """

    # --- Required processing
    nodes = nodes.copy()
    edges = edges.copy()
    if max_distance[0] > 0:
        gdf_f = gdf_f.loc[gdf_f[max_distance[1]] <= max_distance[0]]
    g, weights, node_mapping = to_igraph(
        nodes, edges, weight=weight
    )  # convert to igraph to run the calculations
    seeds = get_seeds(gdf_f, node_mapping, column_name)
    voronoi_assignment = voronoi_cpu(g, weights, seeds)

    # --- Analysis options
    if (
        get_nearest_poi[0] and (count_pois[0])
    ):  # Return distances, nearest poi idx and near count
        distances, nearest_poi_idx, near_count = get_distances(
            g,
            seeds,
            weights,
            voronoi_assignment,
            get_nearest_poi=True,
            count_pois=count_pois,
        )
        nearest_poi = [gdf_f.iloc[i][get_nearest_poi[1]] for i in nearest_poi_idx]
        nodes[f"dist_{poi_name}"] = distances
        nodes[f"nearest_{poi_name}"] = nearest_poi
        nodes[f"{poi_name}_{count_pois[1]}min"] = near_count

    elif get_nearest_poi[0]:  # Return distances and nearest poi idx
        distances, nearest_poi_idx = get_distances(
            g, seeds, weights, voronoi_assignment, get_nearest_poi=True
        )
        nearest_poi = [gdf_f.iloc[i][get_nearest_poi[1]] for i in nearest_poi_idx]
        nodes[f"dist_{poi_name}"] = distances
        nodes[f"nearest_{poi_name}"] = nearest_poi

    elif count_pois[0]:  # Return distances and near count
        distances, near_count = get_distances(
            g, seeds, weights, voronoi_assignment, count_pois=count_pois
        )
        nodes[f"dist_{poi_name}"] = distances
        nodes[f"{poi_name}_{count_pois[1]}min"] = near_count

    else:  # Return distances only
        distances = get_distances(g, seeds, weights, voronoi_assignment)
        nodes[f"dist_{poi_name}"] = distances

    # --- Format
    nodes.replace([np.inf, -np.inf], np.nan, inplace=True)
    idx = pd.notnull(nodes[f"dist_{poi_name}"])
    nodes = nodes[idx].copy()

    return nodes


def walk_speed(edges_elevation: GeoDataFrame) -> GeoDataFrame:
    """
    Calculate walking speeds using Tobler's Hiking Function based on slope.

    Applies Tobler's empirical formula to estimate walking speeds
    on street segments based on their grade/slope.

    Parameters
    ----------
    edges_elevation : geopandas.GeoDataFrame
        GeoDataFrame with street edges containing 'grade' column.
        Grade should be in decimal format (e.g., 0.05 for 5% slope).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of input GeoDataFrame with added 'walkspeed' column.
        Walking speed in km/h based on Tobler's function.
    """
    edges_speed = edges_elevation.copy()
    edges_speed["walkspeed"] = edges_speed.apply(
        lambda row: (4 * np.exp(-3.5 * abs((row["grade"])))), axis=1
    )
    ##To adapt to speed at 0 slope = 3.5km/hr use: (4.2*np.exp(-3.5*abs((row['grade']+0.05))))
    # Using this the max speed 4.2 at -0,05 slope
    return edges_speed


def create_network(
    nodes: GeoDataFrame, edges: GeoDataFrame, projected_crs: str = "EPSG:6372"
) -> Tuple[GeoDataFrame, GeoDataFrame]:
    """
    Create a network from nodes and edges without existing topology attributes.

    Processes raw geospatial data to create a proper network structure by
    generating unique node IDs and establishing edge connectivity.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with node point geometries in EPSG:4326.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with edge line geometries in EPSG:4326.
    projected_crs : str, default "EPSG:6372"
        Projected coordinate system for accurate distance calculations.
        Should be appropriate for the geographic area of analysis.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, geopandas.GeoDataFrame)
        - nodes: Nodes with unique 'osmid' identifiers
        - edges: Edges with 'u', 'v', 'key', and 'length' attributes
    """

    # Copy edges and nodes to avoid editing original GeoDataFrames
    nodes = nodes.copy()
    edges = edges.copy()

    # Create unique ids for nodes and edges
    ##Change coordinate system to meters for unique ids
    nodes = nodes.to_crs(projected_crs)
    edges = edges.to_crs(projected_crs)

    ##Unique id for nodes based on coordinates
    nodes["osmid"] = ((nodes.geometry.x).astype(int)).astype(str) + (
        (nodes.geometry.y).astype(int)
    ).astype(str)

    ##Set columns in edges for to [u] from[v] columns
    edges["u"] = ""
    edges["v"] = ""
    edges.u.astype(str)
    edges.v.astype(str)

    ##Extract start and end coordinates for [u,v] columns
    for index, row in edges.iterrows():
        edges.at[index, "u"] = str(int(list(row.geometry.coords)[0][0])) + str(
            int(list(row.geometry.coords)[0][1])
        )
        edges.at[index, "v"] = str(int(list(row.geometry.coords)[-1][0])) + str(
            int(list(row.geometry.coords)[-1][1])
        )

    # Add key column for compatibility with osmnx
    edges["key"] = 0

    # Change [u,v] columns to integer
    edges["u"] = edges.u.astype(int)
    edges["v"] = edges.v.astype(int)
    # Calculate edges lentgh
    edges["length"] = edges.to_crs(projected_crs).length

    # Change osmid to integer
    nodes["osmid"] = nodes.osmid.astype(int)

    # Transform coordinates
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

    return nodes, edges


def calculate_isochrone(
    G: MultiDiGraph,
    center_node: int,
    trip_length: float,
    weight_column: str,
    undirected: bool = True,
    subgraph: bool = False,
) -> Union[any, Tuple[MultiDiGraph, any]]:  # 'any' represents geometry type
    """
    Calculate an isochrone (equal-time/distance contour) from a center node.

    Creates a polygon representing all locations reachable within a specified
    travel budget from a given network node. Uses NetworkX ego_graph for
    efficient subgraph extraction and convex hull for boundary generation.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Network graph with nodes containing 'x' and 'y' coordinates.
        Must have the specified weight_column on edges.
    center_node : int
        Node ID to use as the isochrone center point.
        Must exist in the graph.
    trip_length : float
        Maximum travel cost (distance/time) for isochrone boundary.
        Units must match weight_column values.
    weight_column : str
        Edge attribute name for travel costs.
        Common values: 'length', 'time_min', 'travel_time'.
    undirected : bool, default True
        Whether to treat graph as undirected for reachability analysis.
    subgraph : bool, default False
        Whether to return the reachable subgraph along with geometry.

    Returns
    -------
    shapely.geometry.Polygon or tuple
        If subgraph=False: Polygon representing isochrone boundary.
        If subgraph=True: Tuple of (reachable_subgraph, boundary_polygon).
    """

    sub_G = nx.ego_graph(
        G,
        center_node,
        radius=trip_length,
        undirected=undirected,
        distance=weight_column,
    )
    geom = gpd.GeoSeries(
        [Point((data["x"], data["y"])) for node, data in sub_G.nodes(data=True)]
    )
    geom = gpd.GeoDataFrame(geometry=geom)
    geometry = geom.geometry.union_all().convex_hull
    log(geometry)
    if subgraph:
        return sub_G, geometry
    else:
        return geometry


def proximity_isochrone(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    point_of_interest: GeoDataFrame,
    trip_time: int,
    prox_measure: str = "length",
    projected_crs: str = "EPSG:6372",
) -> any:  # geometry type
    """
    Create an isochrone polygon around a point of interest using proximity analysis.

    Alternative isochrone calculation method that leverages the proximity analysis
    framework. Identifies all network nodes reachable within a specified threshold
    and creates a convex hull boundary around them.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Network graph with edge bearing attributes.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes in EPSG:4326.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges in EPSG:4326.
    point_of_interest : geopandas.GeoDataFrame
        Single-point GeoDataFrame representing isochrone center.
        Must contain exactly one Point geometry.
    trip_time : int
        Maximum travel time in minutes for isochrone boundary.
    prox_measure : str, default "length"
        Travel cost measure: 'length' or 'time_min'.
        'length' assumes 4 km/h walking speed.
    projected_crs : str, default "EPSG:6372"
        Projected CRS for distance calculations.

    Returns
    -------
    shapely.geometry.Polygon
        Convex hull polygon representing the isochrone boundary.
    """

    # Define projection for downloaded data
    nodes = nodes.set_crs("EPSG:4326")
    edges = edges.set_crs("EPSG:4326")
    point_of_interest = point_of_interest.set_crs("EPSG:4326")

    # Find nearest osmnx node to center node
    nearest = find_nearest_point_to_node(
        G, nodes, point_of_interest, return_distance=True
    )
    nearest = nearest.set_crs("EPSG:4326")

    # Fill NANs in length with calculated length
    no_length = len(edges.loc[edges["length"].isna()])
    edges = edges.to_crs(projected_crs)
    edges.loc[edges["length"].isna(), "length"] = edges.loc[
        edges["length"].isna()
    ].length
    edges = edges.to_crs("EPSG:4326")
    if no_length > 0:
        log(f"Calculated length for {no_length} edges that had no length data.")

    # If prox_measure = 'length', calculates time_min assuming walking speed = 4km/hr
    if prox_measure == "length":
        edges["time_min"] = (edges["length"] * 60) / 4000
    else:
        # NaNs in time_min? --> Assume walking speed = 4km/hr
        no_time = len(edges.loc[edges["time_min"].isna()])
        edges["time_min"].fillna((edges["length"] * 60) / 4000, inplace=True)
        if no_time > 0:
            log(f"Calculated time for {no_time} edges that had no time data.")

    count_pois = (True, trip_time)
    nodes_analysis = nodes.reset_index().copy()
    nodes_time = nodes.copy()

    # Calculate distances
    poi_name = "poi"  # Required by function, has no effect on output
    nodes_distance_prep = calculate_distance_nearest_poi(
        nearest,
        nodes_analysis,
        edges,
        poi_name,
        "osmid",
        weight="time_min",
        count_pois=count_pois,
    )
    # Extract from nodes_distance_prep the calculated pois count.
    nodes_time[f"{poi_name}_{count_pois[1]}min"] = nodes_distance_prep[
        f"{poi_name}_{count_pois[1]}min"
    ]

    # Organice and filter output data
    nodes_time.reset_index(inplace=True)
    nodes_time = nodes_time.set_crs("EPSG:4326")
    nodes_time = nodes_time[
        ["osmid", f"{poi_name}_{count_pois[1]}min", "x", "y", "geometry"]
    ]

    # Keep only nodes where nearest was found at an _x_ time distance
    nodes_at_15min = nodes_time.loc[nodes_time[f"{poi_name}_{count_pois[1]}min"] > 0]

    # Create isochrone using convex hull to those nodes and add osmid from which this isochrone formed
    hull_geometry = nodes_at_15min.union_all().convex_hull

    return hull_geometry


def proximity_isochrone_from_osmid(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    center_osmid: int,
    trip_time: int,
    prox_measure: str = "length",
    projected_crs: str = "EPSG:6372",
) -> any:  # geometry type
    """
    Create an isochrone from a specific network node ID.

    Variant of proximity_isochrone that takes a node ID directly instead of
    a point geometry.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Network graph with edge bearing attributes.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes in EPSG:4326.
        Must contain the specified center_osmid.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges in EPSG:4326.
    center_osmid : int
        Node ID to use as isochrone center.
        Must exist in the nodes GeoDataFrame.
    trip_time : int
        Maximum travel time in minutes for isochrone boundary.
    prox_measure : str, default "length"
        Travel cost measure: 'length' or 'time_min'.
        'length' assumes 4 km/h walking speed.
    projected_crs : str, default "EPSG:6372"
        Projected CRS for distance calculations.

    Returns
    -------
    shapely.geometry.Polygon
        Convex hull polygon representing the isochrone boundary.
    """

    point_of_interest = nodes.loc[nodes["osmid"] == center_osmid].copy()

    hull_geometry = proximity_isochrone(
        G, nodes, edges, point_of_interest, trip_time, prox_measure, projected_crs
    )

    return hull_geometry


def calculate_time_to_pois(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    pois: GeoDataFrame,
    poi_name: str,
    prox_measure: str = "length",
    walking_speed: float = 4.0,
    count_pois: Tuple[bool, int] = (False, 0),
    projected_crs: str = "EPSG:6372",
    progress_callback: Optional[Callable] = None,
) -> GeoDataFrame:
    """
    Calculate travel time from each network node to nearest points of interest.

    Comprehensive proximity analysis function that processes POIs in batches,
    calculates shortest path distances/times to nearest facilities, and
    optionally counts accessible POIs within specified thresholds.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Network graph with edge bearing attributes.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes in EPSG:4326.
        Must have 'osmid' column and coordinate information.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges in EPSG:4326.
        Should have prox_measure column.
    pois : geopandas.GeoDataFrame
        GeoDataFrame with points of interest in EPSG:4326.
    poi_name : str
        Identifier for POI type, used in output column naming.
        Example: 'pharmacy', 'school', 'hospital'.
    prox_measure : str, default 'length'
        Distance calculation method: 'length' or 'time_min'.
        'length' uses walking_speed for time conversion.
    walking_speed : float, default 4.0
        Walking speed in km/h for time calculations.
        Used when prox_measure='length' or to fill missing time_min values.
    count_pois : tuple of (bool, int), default (False, 0)
        Enable POI counting within time threshold (minutes).
        Format: (enable_counting, time_threshold_minutes).
    projected_crs : str, default "EPSG:6372"
        Projected CRS for accurate distance calculations.
        Should be appropriate for the study area.
    progress_callback : callable, optional
        Function to call for progress updates. Should accept (current, total, description).

    Returns
    -------
    geopandas.GeoDataFrame
        Nodes with additional columns:
        - f'time_{poi_name}': Travel time to nearest POI (minutes)
        - f'{poi_name}_{threshold}min': Count of POIs within threshold (if enabled)
    """
    # Validate inputs
    _validate_pois_time_inputs(
        G, nodes, edges, pois, poi_name, prox_measure, walking_speed
    )

    # Ensure proper CRS
    pois = pois.to_crs("EPSG:4326")
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

    # Handle empty POIs case
    if len(pois) == 0:
        return _handle_empty_pois(nodes, poi_name, count_pois)

    # Find or load nearest nodes
    nearest = find_nearest_point_to_node(G, nodes, pois)

    # Prepare network edges
    edges = _prepare_network_edges(edges, prox_measure, walking_speed, projected_crs)

    # Calculate distances using batch processing
    nodes_time = _calculate_poi_distances_batch(
        nearest, nodes, edges, poi_name, count_pois, progress_callback
    )

    # Format and return results
    return _format_output(nodes_time, poi_name, count_pois)


def calculate_time_to_multi_geometry_pois(
    G: MultiDiGraph,
    nodes: GeoDataFrame,
    edges: GeoDataFrame,
    pois: GeoDataFrame,
    poi_name: str,
    prox_measure: str,
    walking_speed: float,
    goi_id: str,
    count_pois: Tuple[bool, int] = (False, 0),
    projected_crs: str = "EPSG:6372",
    max_walking_distance: float = 500.0,
    progress_callback: Optional[Callable] = None,
) -> GeoDataFrame:
    """
    Calculate travel time to POIs derived from geometries of interest with unique IDs.

    Extended version of calculate_time_to_pois for cases where POIs are derived from larger
    geometric features (e.g., park vertices, bike lane segments). Groups POIs
    by their source geometry ID to avoid double-counting.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Network graph with edge bearing attributes.
    nodes : geopandas.GeoDataFrame
        GeoDataFrame with network nodes in EPSG:4326.
    edges : geopandas.GeoDataFrame
        GeoDataFrame with network edges in EPSG:4326.
    pois : geopandas.GeoDataFrame
        GeoDataFrame with POIs derived from geometries of interest.
        Must contain the goi_id column for grouping.
    poi_name : str
        Identifier for POI type, used in output column naming.
    prox_measure : str
        Distance calculation method: 'length' or 'time_min'.
    walking_speed : float
        Walking speed in km/h for time calculations.
    goi_id : str
        Column name containing unique IDs for geometries of interest.
        Used to group POIs that belong to the same source feature.
    count_pois : tuple of (bool, int), default (False, 0)
        Enable counting of distinct geometries within time threshold.
        Format: (enable_counting, time_threshold_minutes).
    projected_crs : str, default "EPSG:6372"
        Projected CRS for distance calculations.
    max_walking_distance : float, default 80.0
        Maximum distance in meters to consider POI accessible from network.
        Represents acceptable walking distance to reach geometry boundary.
    progress_callback : callable, optional
        Function to call for progress updates.

    Returns
    -------
    geopandas.GeoDataFrame
        Nodes with additional columns:
        - f'time_{poi_name}': Travel time to nearest geometry of interest
        - f'{poi_name}_{threshold}min': Count of distinct geometries within threshold
    """
    # Validate inputs
    _validate_id_pois_time_inputs(
        G, nodes, edges, pois, poi_name, prox_measure, walking_speed, goi_id
    )

    # Ensure proper CRS
    pois = pois.to_crs("EPSG:4326")
    nodes = nodes.to_crs("EPSG:4326")
    edges = edges.to_crs("EPSG:4326")

    # Handle empty POIs case
    if len(pois) == 0:
        return _handle_empty_pois(nodes, poi_name, count_pois, use_nan=True)

    # Find or load nearest nodes
    nearest = find_nearest_point_to_node(G, nodes, pois, return_distance=True)
    log(f"Calculated nearest nodes for {len(nearest)} {poi_name} POIs")
    log(nearest)

    # Group by geometry ID and filter by distance
    nearest = _process_geometry_groups(nearest, nodes, goi_id, max_walking_distance)

    # Prepare network edges
    edges = _prepare_network_edges(edges, prox_measure, walking_speed, projected_crs)

    # Calculate distances by geometry of interest
    nodes_time = _calculate_poi_distances_by_geometry(
        nearest, nodes, edges, poi_name, goi_id, count_pois, progress_callback
    )

    # Format and return results
    return _format_output(nodes_time, poi_name, count_pois)


# Helper functions for cleaner code organization


def _validate_pois_time_inputs(
    G, nodes, edges, pois, poi_name, prox_measure, walking_speed
):
    """Validate inputs for pois_time function."""
    if not isinstance(poi_name, str) or not poi_name.strip():
        raise ValueError("poi_name must be a non-empty string")

    if prox_measure not in ["length", "time_min"]:
        raise ValueError("prox_measure must be 'length' or 'time_min'")

    if walking_speed <= 0:
        raise ValueError("walking_speed must be positive")

    if "osmid" not in nodes.columns:
        nodes = nodes.reset_index()

    if "u" not in edges.columns or "v" not in edges.columns:
        edges = edges.reset_index()

    required_columns = {"nodes": ["osmid"], "edges": ["u", "v"], "pois": ["geometry"]}
    for df_name, df in [("nodes", nodes), ("edges", edges), ("pois", pois)]:
        missing_cols = [
            col for col in required_columns[df_name] if col not in df.columns
        ]
        if missing_cols:
            raise ValueError(f"{df_name} missing required columns: {missing_cols}")


def _validate_id_pois_time_inputs(
    G, nodes, edges, pois, poi_name, prox_measure, walking_speed, goi_id
):
    """Validate inputs for id_pois_time function."""
    _validate_pois_time_inputs(
        G, nodes, edges, pois, poi_name, prox_measure, walking_speed
    )

    if goi_id not in pois.columns:
        raise ValueError(f"goi_id column '{goi_id}' not found in pois GeoDataFrame")


def _handle_empty_pois(
    nodes: GeoDataFrame,
    poi_name: str,
    count_pois: Tuple[bool, int],
    use_nan: bool = False,
) -> GeoDataFrame:
    """Handle case when no POIs are found."""
    log(f"No {poi_name} found in the study area")

    nodes_time = nodes.copy().reset_index()
    nodes_time = nodes_time.to_crs("EPSG:4326")

    # Set time column
    nodes_time[f"time_{poi_name}"] = np.nan

    # Set count column
    if count_pois[0]:
        nodes_time[f"{poi_name}_{count_pois[1]}min"] = np.nan if use_nan else 0
        return nodes_time[
            [
                "osmid",
                f"time_{poi_name}",
                f"{poi_name}_{count_pois[1]}min",
                "x",
                "y",
                "geometry",
            ]
        ]
    else:
        return nodes_time[["osmid", f"time_{poi_name}", "x", "y", "geometry"]]


def _prepare_network_edges(edges, prox_measure, walking_speed, projected_crs):
    """Prepare network edges with proper length and time calculations."""
    edges = edges.copy()

    # Fill missing length values
    missing_length = edges["length"].isna().sum()
    if missing_length > 0:
        edges_projected = edges.to_crs(projected_crs)
        edges.loc[edges["length"].isna(), "length"] = edges_projected.loc[
            edges["length"].isna()
        ].length
        log(f"Calculated length for {missing_length} edges")

    # Calculate or fix time values
    if prox_measure == "length":
        edges["time_min"] = (edges["length"] * 60) / (walking_speed * 1000)
    else:
        missing_time = edges["time_min"].isna().sum()
        if missing_time > 0:
            edges.loc[edges["time_min"].isna(), "time_min"] = (
                edges.loc[edges["time_min"].isna(), "length"] * 60
            ) / (walking_speed * 1000)
            log(f"Calculated time for {missing_time} edges using walking speed")

    return edges


def _calculate_poi_distances_batch(
    nearest, nodes, edges, poi_name, count_pois, progress_callback
):
    """Calculate POI distances using batch processing."""
    nodes_analysis = nodes.reset_index().copy()
    nodes_time = nodes.reset_index().copy()

    log(f"Starting proximity analysis for {poi_name}")

    # Determine batch size
    batch_size = 250 if len(nearest) % 250 == 0 else 200
    n_batches = len(nearest) // batch_size + (1 if len(nearest) % batch_size else 0)

    time_results = []
    count_results = []

    for i in range(n_batches):
        if progress_callback:
            progress_callback(i, n_batches, f"Processing {poi_name} batch")

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(nearest))
        batch = nearest.iloc[start_idx:end_idx].copy()

        # Calculate distances for this batch
        batch_result = calculate_distance_nearest_poi(
            batch,
            nodes_analysis,
            edges,
            poi_name,
            "osmid",
            weight="time_min",
            count_pois=count_pois,
        )

        # Store results
        time_results.append(batch_result[f"dist_{poi_name}"])
        if count_pois[0]:
            count_results.append(batch_result[f"{poi_name}_{count_pois[1]}min"])

    # Combine batch results
    if time_results:
        # Find minimum time across all batches
        all_times = pd.concat(time_results, axis=1)
        nodes_time[f"time_{poi_name}"] = all_times.min(axis=1).values

        # Sum counts across batches
        if count_pois[0]:
            all_counts = pd.concat(count_results, axis=1)
            nodes_time[f"{poi_name}_{count_pois[1]}min"] = all_counts.sum(axis=1).values

    log(f"Completed proximity analysis for {poi_name}")
    return nodes_time


def _process_geometry_groups(nearest, nodes, goi_id, max_walking_distance):
    """Process POI groups by geometry of interest ID."""
    # Group by node and geometry ID, keeping minimum distance
    grouped = (
        nearest.groupby(["osmid", goi_id]).agg({"distance_node": "min"}).reset_index()
    )

    # Merge back with node geometries
    geom_gdf = nodes.reset_index()[["osmid", "geometry"]]
    nearest_processed = pd.merge(grouped, geom_gdf, on="osmid", how="left")
    nearest_processed = gpd.GeoDataFrame(nearest_processed, geometry="geometry")

    # Filter by maximum walking distance
    nearest_filtered = nearest_processed.loc[
        nearest_processed.distance_node <= max_walking_distance
    ]

    log(
        f"Filtered POIs to {len(nearest_filtered)} within {max_walking_distance}m of network"
    )
    return nearest_filtered


def _calculate_poi_distances_by_geometry(
    nearest, nodes, edges, poi_name, goi_id, count_pois, progress_callback
):
    """Calculate distances by processing each geometry of interest separately."""
    nodes_analysis = nodes.reset_index().copy()
    nodes_time = nodes.copy()

    unique_geometries = nearest[goi_id].unique()
    log(f"Processing {len(unique_geometries)} unique geometries for {poi_name}")

    for i, goi in enumerate(unique_geometries):
        if progress_callback:
            progress_callback(i, len(unique_geometries), f"Processing geometry {goi}")

        # Process this geometry's POIs
        geometry_pois = nearest.loc[nearest[goi_id] == goi]
        result = calculate_distance_nearest_poi(
            geometry_pois,
            nodes_analysis,
            edges,
            poi_name,
            "osmid",
            weight="time_min",
            count_pois=count_pois,
        )

        # Process time data
        if i == 0:
            nodes_time[f"time_{poi_name}"] = result[f"dist_{poi_name}"]
        else:
            nodes_time[f"time_{poi_name}"] = np.minimum(
                nodes_time[f"time_{poi_name}"], result[f"dist_{poi_name}"]
            )

        # Process count data (convert to binary for each geometry)
        if count_pois[0]:
            binary_count = (result[f"{poi_name}_{count_pois[1]}min"] > 0).astype(int)

            if i == 0:
                nodes_time[f"{poi_name}_{count_pois[1]}min"] = binary_count
            else:
                nodes_time[f"{poi_name}_{count_pois[1]}min"] += binary_count

    log(f"Completed geometry-based analysis for {poi_name}")
    return nodes_time


def _format_output(nodes_time, poi_name, count_pois):
    """Format the final output GeoDataFrame."""
    # Remove infinite values
    nodes_time.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter out nodes with no valid distances
    valid_mask = pd.notnull(nodes_time[f"time_{poi_name}"])
    nodes_time = nodes_time[valid_mask].copy()

    # Reset index and set CRS
    nodes_time.reset_index(drop=True, inplace=True)
    nodes_time = nodes_time.to_crs("EPSG:4326")

    # Select output columns
    output_cols = ["osmid", f"time_{poi_name}"]
    if count_pois[0]:
        output_cols.append(f"{poi_name}_{count_pois[1]}min")
    output_cols.extend(["x", "y", "geometry"])

    return nodes_time[output_cols]
