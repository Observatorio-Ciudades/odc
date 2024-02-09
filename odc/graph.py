################################################################################
# Module: Graph
# Set of graph analysis functions
# updated: 15/11/2023
################################################################################

from pyproj import CRS
from sklearn.neighbors import BallTree
from scipy.spatial import cKDTree
import numpy as np

EARTH_RADIUS_M = 6_371_009

def is_projected(crs):
    """
    Determine if a coordinate reference system is projected or not.
    This is a convenience wrapper around the pyproj.CRS.is_projected function.
    
    Arguments:
    crs (string or pyproj.CRS): the coordinate reference system

    Returns:
    projected (bool) True if crs is projected, otherwise False
    """
    return CRS.from_user_input(crs).is_projected

def nearest_nodes(G, nodes, X, Y, return_dist=False):
    """
    Find the nearest node to a point or to each of several points.
    If `X` and `Y` are single coordinate values, this will return the nearest
    node to that point. If `X` and `Y` are lists of coordinate values, this
    will return the nearest node to each point.
    If the graph is projected, this uses a k-d tree for euclidean nearest
    neighbor search, which requires that scipy is installed as an optional
    dependency. If it is unprojected, this uses a ball tree for haversine
    nearest neighbor search, which requires that scikit-learn is installed as
    an optional dependency.
    Arguments:
        G (networkx.MultiDiGraph) graph in which to find nearest nodes
        X (float or list) points' x (longitude) coordinates, in same CRS/units as graph and containing no nulls
        Y (float or list): Points' y (latitude) coordinates, in same CRS/units as graph and containing no nulls
        return_dist (bool): Optionally also return distance between points and nearest nodes
    Returns:
        nn (int/list or tuple): Nearest node IDs or optionally a tuple where `dist` contains distances 
        between the points and their nearest nodes
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
            raise ImportError("scikit-learn must be installed to search an unprojected graph")
        # haversine requires lat, lng coords in radians
        nodes_rad = np.deg2rad(nodes[["y", "x"]])
        points_rad = np.deg2rad(np.array([Y, X]).T)
        dist, pos = BallTree(nodes_rad, metric="haversine").query(points_rad, k=1)
        dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters
        nn = nodes.index[pos[:, 0]]

    # convert results to correct types for return
    nn = nn.tolist()
    dist = dist.tolist()
    if is_scalar:
        nn = nn[0]
        dist = dist[0]

    if return_dist:
        return nn, dist
    else:
        return nn