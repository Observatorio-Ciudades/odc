import igraph as ig
import numpy as np
import geopandas as gpd
import networkx as nx
import osmnx as ox
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

from .utils import *



def nearest_nodes(G: MultiDigraph,
    nodes: GeoDataFrame,
    X: float | list,
    Y: float | list,
    return_dist: bool = False) -> int | list | float:
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
        G (networkx.MultiDiGraph): graph in which to find nearest nodes
        nodes (geopandas.GeoDataFrame): geodataframe containing the nodes in case a Graph isn't available
        X (float or list): points' x (longitude) coordinates, in same CRS/units as graph and containing no nulls
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
        EARTH_RADIUS_M = 6_371_009
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


def find_nearest(G, nodes, gdf, return_distance=False): # proximity
    """
	Find the nearest graph nodes to the points in a GeoDataFrame

	Arguments:
		G (networkx.Graph): Graph created with OSMnx that contains CRS information
        nodes (geopandas.GeoDataFrame): OSMnx nodes with osmid index
		gdf (geopandas.GeoDataFrame): GeoDataFrame with the points to locate
		return_distance (bool): If True, returns distance to nearest node. Defaults to False

	Returns:
		gdf (geopandas.GeoDataFrame): original dataframe with a new column call 'nearest' with the node id closser to the point
	"""

    gdf = gdf.copy()

    osmnx_tuple = nearest_nodes(G, nodes, list(gdf.geometry.x),list(gdf.geometry.y), return_dist=return_distance)

    if return_distance:
        gdf['osmid'] = osmnx_tuple[0]
        gdf['distance_node'] = osmnx_tuple[1]
    else:
        gdf['osmid'] = osmnx_tuple
    return gdf


def to_igraph(nodes, edges, wght='lenght'):
    """
    Convert a graph from networkx to igraph

	Arguments:
		nodes (geopandas.GeoDataFrame): OSMnx nodes with osmid index
        edges (geopandas.GeoDataFrame): OSMnx edges with u and v indexes
        wght (str): weights column in edges. Defaults to length

	Returns:
		g (graph): Graph with the same number of nodes and edges as the original one
		weights (np.array) Array of weights, defined according to weight variable
		dict (dict): With the node mapping, index is the node in networkx.Graph, value is the node in igraph.Graph
    """

    nodes.reset_index(inplace=True)
    edges.reset_index(inplace=True)

    edges.set_index(['u','v'], inplace=True)
    nodes.set_index(['osmid'], inplace=True)

    node_mapping = dict(zip(nodes.index.values,range(len(nodes))))
    g = ig.Graph(len(nodes), [(node_mapping[i[0]],node_mapping[i[1]]) for i in edges.index.values])
    weights=np.array([float(e) for e in edges[wght]])

    return g, weights, node_mapping


def get_seeds(gdf, node_mapping, column_name): # proximity
	"""
	Generate the seed to be used to calculate shortest paths for the Voronoi's

	Arguments:
		gdf (geopandas.GeoDataFrame): GeoDataFrame with 'nearest' column
		node_mapping (dict): dictionary containing the node mapping from networkx.Graph to igraph.Graph
        column_name (str): column name where the nearest distance index is stored

	Returns:
		np.array: numpy.array with the set of seeds
	"""
	# Get the seed to calculate shortest paths
	return np.array(list([node_mapping[i] for i in gdf[column_name]]))


def voronoi_cpu(g, weights, seeds):
	"""
	Voronoi diagram calculator for undirected graphs
	Optimized for computational efficiency

	Arguments:
		g (igraph.Graph): graph object with Nodes and Edges
		weights (numpy.array): array of weights for all edges of length len(V)
		seeds (numpy.array): generator points as numpy array of indices from the node array

	Returns:
		seeds (numpy.array): numpy.array on len(N) where the location (index) of the node refers to the node, the value is the generator (seed) the respective nodes belongs to.
	"""
	return seeds[np.array(g.shortest_paths_dijkstra(seeds, weights=weights)).argmin(axis=0)]


def get_distances(g, seeds, weights, voronoi_assignment, get_nearest_poi=False, count_pois=(False,0)):
	"""
	Distance for the shortest path for each node to the closest seed using Dijkstra's algorithm.
	Arguments:
		g (igraph.Graph): Graph object that calculates the shortest path between two nodes
		seeds (numpy.array): Find the shortest path from each node to the closest seed
		weights (numpy.array): Specify the weights of each edge, which is used to calculate the shortest path
		voronoi_assignment (numpy.array): Assign the nodes to their respective seeds
		get_nearest_poi (bool, optional): Returns idx of nearest point of interest. Defaults to False.
		count_pois (tuple, optional): tuple containing boolean to find number of pois within given time proximity. Defaults to (False, 0)

	Returns:
		The distance for the shortest path for each node to the closest seed
	"""
	shortest_paths = np.array(g.shortest_paths_dijkstra(seeds,weights=weights))
	distances = [np.min(shortest_paths[:,i]) for i in range(len(voronoi_assignment))]
	if get_nearest_poi:
		nearest_poi_idx = [np.argmin(shortest_paths[:,i]) for i in range(len(voronoi_assignment))]
	if count_pois[0]:
		near_count = [len(np.where(shortest_paths[:,i] <= count_pois[1])[0]) for i in range(len(voronoi_assignment))]

	# Function output options
	if get_nearest_poi and count_pois[0]:
		return distances, nearest_poi_idx, near_count
	elif get_nearest_poi:
		return distances, nearest_poi_idx
	elif count_pois[0]:
		return distances, near_count
	else:
		return distances


def calculate_distance_nearest_poi(gdf_f, nodes, edges, amenity_name, column_name,
wght='length', get_nearest_poi=(False, 'poi_id_column'), count_pois=(False,0), max_distance=(0,'distance_node')):
	"""
	Calculate the distance to the shortest path to the nearest POI (in gdf_f) for all the nodes in the network G

	Arguments:
		gdf_f (geopandas.GeoDataFrame): GeoDataFrame with the Points of Interest the geometry type has to be shapely.Point
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network analysis
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network analysis
		amenity_name (str): string with the name of the amenity that is used as seed (pharmacy, hospital, shop, etc.)
		column_name (str): column name where the nearest distance index is stored
		wght (str): weights column in edges. Defaults to length
		get_nearest_poi (tuple, optional): tuple containing boolean to get the nearest POI and column name that contains that value. Defaults to (False, 'poi_id_column')
		count_pois (tuple, optional): tuple containing boolean to find number of pois within given time proximity. Defaults to (False, 0)
		max_distance (tuple): tuple containing limits for distance to node and column name that contains that value. Defaults to (0, distance_node)

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with geometry and distance to the nearest POI
	"""

	# --- Required processing
	nodes = nodes.copy()
	edges = edges.copy()
	if max_distance[0] > 0:
		gdf_f = gdf_f.loc[gdf_f[max_distance[1]]<=max_distance[0]]
	g, weights, node_mapping = to_igraph(nodes,edges,wght=wght) #convert to igraph to run the calculations
	seeds = get_seeds(gdf_f, node_mapping, column_name)
	voronoi_assignment = voronoi_cpu(g, weights, seeds)

	# --- Analysis options
	if get_nearest_poi[0] and (count_pois[0]): # Return distances, nearest poi idx and near count
		distances, nearest_poi_idx, near_count = get_distances(g,seeds,weights,voronoi_assignment,
                                                               get_nearest_poi=True,
                                                               count_pois=count_pois)
		nearest_poi = [gdf_f.iloc[i][get_nearest_poi[1]] for i in nearest_poi_idx]
		nodes[f'dist_{amenity_name}'] = distances
		nodes[f'nearest_{amenity_name}'] = nearest_poi
		nodes[f'{amenity_name}_{count_pois[1]}min'] = near_count

	elif get_nearest_poi[0]: # Return distances and nearest poi idx
		distances, nearest_poi_idx = get_distances(g,seeds,weights,voronoi_assignment,
                                                   get_nearest_poi=True)
		nearest_poi = [gdf_f.iloc[i][get_nearest_poi[1]] for i in nearest_poi_idx]
		nodes[f'dist_{amenity_name}'] = distances
		nodes[f'nearest_{amenity_name}'] = nearest_poi

	elif (count_pois[0]): # Return distances and near count
		distances, near_count = get_distances(g,seeds,weights,voronoi_assignment,
                                              count_pois=count_pois)
		nodes[f'dist_{amenity_name}'] = distances
		nodes[f'{amenity_name}_{count_pois[1]}min'] = near_count

	else: # Return distances only
		distances = get_distances(g,seeds,weights,voronoi_assignment)
		nodes[f'dist_{amenity_name}'] = distances

	# --- Format
	nodes.replace([np.inf, -np.inf], np.nan, inplace=True)
	idx = pd.notnull(nodes[f'dist_{amenity_name}'])
	nodes = nodes[idx].copy()

	return nodes


def walk_speed(edges_elevation):

	"""
	Calculates the Walking speed Using Tobler's Hiking Function and the slope in edges

	Arguments:
		edges_elevation (geopandas.GeoDataFrame): GeoDataFrame with the street edges with slope data


	Returns:
		geopandas.GeoDataFrame: edges_speed GeoDataFrame with the edges with an added column for speed
	"""
	edges_speed = edges_elevation.copy()
	edges_speed['walkspeed'] = edges_speed.apply(lambda row : (4*np.exp(-3.5*abs((row['grade'])))), axis=1)
	##To adapt to speed at 0 slope = 3.5km/hr use: (4.2*np.exp(-3.5*abs((row['grade']+0.05))))
	#Using this the max speed 4.2 at -0,05 slope
	return edges_speed


def create_network(nodes, edges, projected_crs="EPSG:6372"):

	"""
	Create a network based on nodes and edges without unique ids and to - from attributes.

	Arguments:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".

	Returns:
		geopandas.GeoDataFrame: nodes GeoDataFrame with unique ids based on coordinates named osmid in EPSG:4326
		geopandas.GeoDataFrame: edges GeoDataFrame with to - from attributes based on nodes ids named u and v respectively in EPSG:4326
	"""

	#Copy edges and nodes to avoid editing original GeoDataFrames
	nodes = nodes.copy()
	edges = edges.copy()

	#Create unique ids for nodes and edges
	##Change coordinate system to meters for unique ids
	nodes = nodes.to_crs(projected_crs)
	edges = edges.to_crs(projected_crs)

	##Unique id for nodes based on coordinates
	nodes['osmid'] = ((nodes.geometry.x).astype(int)).astype(str)+((nodes.geometry.y).astype(int)).astype(str)

	##Set columns in edges for to [u] from[v] columns
	edges['u'] = np.nan
	edges['v'] = np.nan
	edges.u.astype(str)
	edges.v.astype(str)

	##Extract start and end coordinates for [u,v] columns
	for index, row in edges.iterrows():

		edges.at[index,'u'] = str(int(list(row.geometry.coords)[0][0]))+str(int(list(row.geometry.coords)[0][1]))
		edges.at[index,'v'] = str(int(list(row.geometry.coords)[-1][0]))+str(int(list(row.geometry.coords)[-1][1]))

	#Add key column for compatibility with osmnx
	edges['key'] = 0

	#Change [u,v] columns to integer
	edges['u'] = edges.u.astype(int)
	edges['v'] = edges.v.astype(int)
	#Calculate edges lentgh
	edges['length'] = edges.to_crs(projected_crs).length

	#Change osmid to integer
	nodes['osmid'] = nodes.osmid.astype(int)

	#Transform coordinates
	nodes = nodes.to_crs("EPSG:4326")
	edges = edges.to_crs("EPSG:4326")

	return nodes, edges


def calculate_isochrone(G, center_node, trip_time, wgt_column, undirected=True, subgraph=False): # proximity
    """
		Function that that creates a isochrone from a center_node in graph G,
		 and uses parameters like distance and time to plot the result.
    Arguments:
        G (networkx.Graph): networkx Graph with travel time (time) attribute.
        center_node (int): id of the node to use
        trip_time (int): maximum travel time allowed
		wgt_column (int): column name with weight for calculation
        subgraph (bool, optional): Bool to get the resulting subgraph or only the geometry. Defaults to False.
    Returns:
        sub_G (optional): subgraph of the covered area.
        geometry (geometry): with the covered area
    """

    sub_G = nx.ego_graph(G, center_node, radius=trip_time, undirected=undirected, distance=wgt_column)
    geometry = gpd.GeoSeries([Point((data["x"], data["y"])) for node, data in sub_G.nodes(data=True)]).unary_union.convex_hull
    if subgraph:
        return sub_G, geometry
    else:
        return geometry


def pois_time(G, nodes, edges, pois, poi_name, prox_measure, walking_speed=4, count_pois=(False,0), projected_crs="EPSG:6372",
			  preprocessed_nearest=(False,'dir')): # proximity
	""" Finds time from each node to nearest poi (point of interest).
	Args:
		G (networkx.MultiDiGraph): Graph with edge bearing attributes
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes within boundaries
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges within boundaries
		pois (geopandas.GeoDataFrame): GeoDataFrame with points of interest
		poi_name (str): Text containing name of the point of interest being analysed
		prox_measure (str): Text ("length" or "time_min") used to choose a way to calculate time between nodes and points of interest.
							If "length", will use walking speed.
							If "time_min", edges with time information must be provided.
		walking_speed (float): Decimal number containing walking speed (in km/hr) to be used if prox_measure="length",
							   or if prox_measure="time_min" but needing to fill time_min NaNs. Defaults to 4 (km/hr).
		count_pois (tuple, optional): tuple containing boolean to find number of pois within given time proximity. Defaults to (False, 0)
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
		preprocessed_nearest (tuple, optional): tuple containing boolean to use a previously calculated nearest file located in a local directory.
												Used when calculating proximity in a non-continous network while keeping actual nearest nodes to each poi
												from an original continous network. Defaults to (False, 'dir).

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with nodes containing time to nearest source (s).
	"""

	# Helps by printing steps to find solutions (If using, make sure test_save_dir exists.)
	test_save = False
	test_save_dir = "../data/debugging/pois_time/"

    ##########################################################################################
    # STEP 1: NEAREST.
	# Finds and assigns nearest node OSMID to each point of interest.

    # Defines projection for downloaded data
	pois = pois.set_crs("EPSG:4326")
	nodes = nodes.set_crs("EPSG:4326")
	edges = edges.set_crs("EPSG:4326")

 	# In case there are no amenities of the type in the city, prevents it from crashing if len = 0
	if len(pois) == 0:
		nodes_time = nodes.copy()

		# Format
		nodes_time.reset_index(inplace=True)
		nodes_time = nodes_time.set_crs("EPSG:4326")

		# As no amenities were found, time columns are set to nan, while count columns are set to 0.
		nodes_time['time_'+poi_name] = np.nan # Time is set to np.nan.
		print(f"0 {poi_name} found. Time set to np.nan for all nodes.")
		if count_pois[0]:
			nodes_time[f'{poi_name}_{count_pois[1]}min'] = 0 # If requested pois_count, value is set to 0.
			print(f"0 {poi_name} found. Pois count set to 0 for all nodes.")
			nodes_time = nodes_time[['osmid','time_'+poi_name,f'{poi_name}_{count_pois[1]}min','x','y','geometry']]
			return nodes_time
		else:
			nodes_time = nodes_time[['osmid','time_'+poi_name,'x','y','geometry']]
			return nodes_time

	else:
		if preprocessed_nearest[0]:
			# Load precalculated (with entire network) nearest gdf.
			nearest = gpd.read_file(preprocessed_nearest[1]+f"nearest_{poi_name}.gpkg")
			print(f"Loaded {len(nearest)} previously calculated nearest data for {poi_name}.")
			# Filter nearest by keeping osmids located in current nodes (filtered network) gdf.
			osmid_check_list = list(nodes.reset_index().osmid.unique())
			nearest = nearest.loc[nearest.osmid.isin(osmid_check_list)]
			print(f"Filtered nearest for filtered network. Kept {len(nearest)}.")
		else:
			### Find nearest osmnx node for each DENUE point.
			nearest = find_nearest(G, nodes, pois, return_distance= True)
			nearest = nearest.set_crs("EPSG:4326")
			print(f"Found and assigned {len(nearest)} nearest node osmid to each {poi_name}.")

		##########################################################################################
		# STEP 2: DISTANCE NEAREST POI.
		# Calculates distance from each node to its nearest point of interest using previously assigned nearest node.

		# 2.1 --------------- FORMAT NETWORK DATA
		# Fill NANs in length with calculated length (prevents crash)
		no_length = len(edges.loc[edges['length'].isna()])
		edges = edges.to_crs(projected_crs)
		edges['length'].fillna(edges.length,inplace=True)
		edges = edges.to_crs("EPSG:4326")
		print(f"Calculated length for {no_length} edges that had no length data.")

		# If prox_measure = 'length', calculates time_min using walking_speed
		if prox_measure == 'length':
			edges['time_min'] = (edges['length']*60)/(walking_speed*1000)
		else:
			# NaNs in time_min? --> Use specified walking speed (defaults to 4km/hr) to calculate time_min.
			no_time = len(edges.loc[edges['time_min'].isna()])
			edges['time_min'].fillna((edges['length']*60)/(walking_speed*1000),inplace=True)
			print(f"Calculated time for {no_time} edges that had no time data.")

		# 2.2 --------------- ELEMENTS NEEDED OUTSIDE THE ANALYSIS LOOP
		# The pois are divided by batches of 200 or 250 pois and analysed using the function calculate_distance_nearest_poi.

		# -----
		if test_save:
			nodes.to_file(test_save_dir + f"01_nodes_{poi_name}.geojson", driver='GeoJSON')
			edges.to_file(test_save_dir + f"01_edges_{poi_name}.geojson", driver='GeoJSON')
		# -----

		# nodes_analysis is a nodes gdf (index reseted) used in the function aup.calculate_distance_nearest_poi.
		nodes_analysis = nodes.reset_index().copy()

		# -----
		if test_save:
			nodes_analysis.to_file(test_save_dir + f"02_empty_nodes_analysis_{poi_name}.geojson", driver='GeoJSON')
		# -----

		# nodes_time: int_gdf stores, processes time data within the loop and returns final gdf. (df_int, df_temp, df_min and nodes_distance in previous code versions)
		nodes_time = nodes.reset_index().copy()

		# -----
		if test_save:
			nodes_time.to_file(test_save_dir + f"03_empty_nodes_time_{poi_name}.gpkg", driver='GPKG')
		# -----

		# --------------- 2.3 PROCESSING DISTANCE
		print (f"Starting time analysis for {poi_name}.")

		# List of columns with output data by batch
		time_cols = []
		poiscount_cols = []

		# If possible, analyses by batches of 200 pois.
		if (len(nearest) % 250)==0:
			batch_size = len(nearest)/200
			for k in range(int(batch_size)+1):
				print(f"Starting batch200 k={k+1} of {int(batch_size)+1} for source {poi_name}.")
				# Calculate
				source_process = nearest.iloc[int(200*k):int(200*(1+k))].copy()
				nodes_distance_prep = calculate_distance_nearest_poi(source_process, nodes_analysis, edges, poi_name, 'osmid', wght='time_min',count_pois=count_pois)
				# Set osmid as column, not index
				nodes_distance_prep.drop(columns=['index'],inplace=True)
				nodes_distance_prep.reset_index(inplace=True)

				# -----
				if test_save:
					nodes_distance_prep.to_file(test_save_dir + f"04_nodes_distance_prep_{poi_name}_200batch{k}.gpkg", driver='GPKG')
				# -----

				# Extract from nodes_distance_prep to nodes_time the batch's calculated time data
				batch_time_col = 'time_'+str(k)+poi_name
				time_cols.append(batch_time_col)
				nodes_time = pd.merge(nodes_time,nodes_distance_prep[['osmid','dist_'+poi_name]],on='osmid',how='left')
				nodes_time.rename(columns={'dist_'+poi_name:batch_time_col},inplace=True)

				# Extract from nodes_distance_prep to nodes_time the batch's calculated pois count (If requested)
				if count_pois[0]:
					batch_poiscount_col = f'{poi_name}_{str(k)}_{count_pois[1]}min'
					poiscount_cols.append(batch_poiscount_col)
					nodes_time = pd.merge(nodes_time,nodes_distance_prep[['osmid',f'{poi_name}_{count_pois[1]}min']],on='osmid',how='left')
					nodes_time.rename(columns={f'{poi_name}_{count_pois[1]}min':batch_poiscount_col},inplace=True)
					# Turn count column to integer (Sometimes it got converted to float, which caused problems when uploading to database)
					nodes_time[batch_poiscount_col] = nodes_time[batch_poiscount_col].fillna(0).astype(int)

			# After batch processing is over, find final output values for all batches.
			# For time data, apply the min function to time columns.
			nodes_time['time_'+poi_name] = nodes_time[time_cols].min(axis=1)
			# Apply the sum function to pois_count columns (If requested).
			if count_pois[0]:
				# Sum pois count
				nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_time[poiscount_cols].sum(axis=1)

		# Else, analyses by batches of 250 pois.
		else:
			batch_size = len(nearest)/250
			for k in range(int(batch_size)+1):
				print(f"Starting batch250 k={k+1} of {int(batch_size)+1} for source {poi_name}.")
				# Calculate
				source_process = nearest.iloc[int(250*k):int(250*(1+k))].copy()
				nodes_distance_prep = calculate_distance_nearest_poi(source_process, nodes_analysis, edges, poi_name, 'osmid', wght='time_min',count_pois=count_pois)
				# Set osmid as column, not index
				nodes_distance_prep.drop(columns=['index'],inplace=True)
				nodes_distance_prep.reset_index(inplace=True)

				# -----
				if test_save:
					nodes_distance_prep.to_file(test_save_dir + f"04_nodes_distance_prep_{poi_name}_250batch{k}.gpkg", driver='GPKG')
				# -----

				# Extract from nodes_distance_prep to nodes_time the batch's calculated time data
				batch_time_col = 'time_'+str(k)+poi_name
				time_cols.append(batch_time_col)
				nodes_time = pd.merge(nodes_time,nodes_distance_prep[['osmid','dist_'+poi_name]],on='osmid',how='left')
				nodes_time.rename(columns={'dist_'+poi_name:batch_time_col},inplace=True)

				# Extract from nodes_distance_prep to nodes_time the batch's calculated pois count (If requested)
				if count_pois[0]:
					batch_poiscount_col = f'{poi_name}_{str(k)}_{count_pois[1]}min'
					poiscount_cols.append(batch_poiscount_col)
					nodes_time = pd.merge(nodes_time,nodes_distance_prep[['osmid',f'{poi_name}_{count_pois[1]}min']],on='osmid',how='left')
					nodes_time.rename(columns={f'{poi_name}_{count_pois[1]}min':batch_poiscount_col},inplace=True)
					# Turn count column to integer (Sometimes it got converted to float, which caused problems when uploading to database)
					nodes_time[batch_poiscount_col] = nodes_time[batch_poiscount_col].fillna(0).astype(int)

			# After batch processing is over, find final output values for all batches.
			# For time data, apply the min function to time columns.
			nodes_time['time_'+poi_name] = nodes_time[time_cols].min(axis=1)
			# Apply the sum function to pois_count columns. (If requested)
			if count_pois[0]:
				# Sum pois count
				nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_time[poiscount_cols].sum(axis=1)

		print(f"Finished time analysis for {poi_name}.")

		# -----
		if test_save:
			nodes_time.to_file(test_save_dir + f"05_nodes_time_{poi_name}.gpkg", driver='GPKG')
		# -----

		##########################################################################################
		# STEP 3: FINAL FORMAT.
  		# Organices and filters output data.
		nodes_time = nodes_time.set_crs("EPSG:4326")

		if count_pois[0]:
			nodes_time = nodes_time[['osmid','time_'+poi_name,f'{poi_name}_{count_pois[1]}min','x','y','geometry']]
			return nodes_time
		else:
			nodes_time = nodes_time[['osmid','time_'+poi_name,'x','y','geometry']]
			return nodes_time


def proximity_isochrone(G, nodes, edges, point_of_interest, trip_time, prox_measure="length", projected_crs="EPSG:6372"): # proximity

	""" This should be a TEMPORARY FUNCTION. It was developed on the idea that isochrones can be created using the code from proximity analysis,
		particularly, count_pois functionality.

		The function analyses proximity to a point of interest, filters for nodes located at a trip_time or less minutes
		from the point of interest (count_pois functionality) and returns a convex hull geometry around those nodes.

	Args:
		G (networkx.MultiDiGraph): Graph with edge bearing attributes.
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes within boundaries.
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges within boundaries.
		point_of_interest (geopandas.GeoDataFrame): GeoDataFrame with point of interest to which to find an isochrone.
		trip_time (int): maximum travel time allowed (minutes).
		prox_measure (str): Text ("length" or "time_min") used to choose a way to calculate time between nodes and points of interest.
							If "length", will assume a walking speed of 4km/hr.
							If "time_min", edges with time information must be provided.
							Defaults to "length".
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".

	Returns:
		geometry (geometry): with the covered area.
	"""

    # Define projection for downloaded data
	nodes = nodes.set_crs("EPSG:4326")
	edges = edges.set_crs("EPSG:4326")
	point_of_interest = point_of_interest.set_crs("EPSG:4326")

    # 1.0 --------------- ASSIGN CENTER NODE TO NEAREST OSMnx NODE
    # Find nearest osmnx node to center node
	nearest = find_nearest(G, nodes, point_of_interest, return_distance= True)
	nearest = nearest.set_crs("EPSG:4326")

    # 2.0 --------------- FORMAT NETWORK DATA
    # Fill NANs in length with calculated length (prevents crash)
	no_length = len(edges.loc[edges['length'].isna()])
	edges = edges.to_crs(projected_crs)
	edges['length'].fillna(edges.length,inplace=True)
	edges = edges.to_crs("EPSG:4326")
	if no_length > 0:
		print(f"Calculated length for {no_length} edges that had no length data.")

    # If prox_measure = 'length', calculates time_min assuming walking speed = 4km/hr
	if prox_measure == 'length':
		edges['time_min'] = (edges['length']*60)/4000
	else:
        # NaNs in time_min? --> Assume walking speed = 4km/hr
		no_time = len(edges.loc[edges['time_min'].isna()])
		edges['time_min'].fillna((edges['length']*60)/4000,inplace=True)
		if no_time > 0:
			print(f"Calculated time for {no_time} edges that had no time data.")

    # 3.0 --------------- PROCESS DISTANCE
	count_pois = (True,trip_time)
	nodes_analysis = nodes.reset_index().copy()
	nodes_time = nodes.copy()

    # Calculate distances
	poi_name = 'poi' #Required by function, has no effect on output
	nodes_distance_prep = calculate_distance_nearest_poi(nearest, nodes_analysis, edges, poi_name,'osmid', wght='time_min',count_pois=count_pois)
    # Extract from nodes_distance_prep the calculated pois count.
	nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_distance_prep[f'{poi_name}_{count_pois[1]}min']

    # Organice and filter output data
	nodes_time.reset_index(inplace=True)
	nodes_time = nodes_time.set_crs("EPSG:4326")
	nodes_time = nodes_time[['osmid',f'{poi_name}_{count_pois[1]}min','x','y','geometry']]

    # 4.0 --------------- GET ISOCHRONE FOR CURRENT CENTER NODE
    # Keep only nodes where nearest was found at an _x_ time distance
	nodes_at_15min = nodes_time.loc[nodes_time[f"{poi_name}_{count_pois[1]}min"]>0]

    # Create isochrone using convex hull to those nodes and add osmid from which this isochrone formed
	hull_geometry = nodes_at_15min.unary_union.convex_hull

	return hull_geometry

def proximity_isochrone_from_osmid(G, nodes, edges, center_osmid, trip_time, prox_measure="length", projected_crs="EPSG:6372"): # proximity

	""" This should be a TEMPORARY FUNCTION. It was developed on the idea that isochrones can be created using the code from proximity analysis,
		particularly, count_pois functionality.

		The function analyses proximity to a point of interest, filters for nodes located at a trip_time or less minutes
		from the point of interest (count_pois functionality) and returns a convex hull geometry around those nodes.

	Args:
		G (networkx.MultiDiGraph): Graph with edge bearing attributes.
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes within boundaries.
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges within boundaries.
		center_osmid (int): Integer with osmid of the center node from which to find an isochrone.
		trip_time (int): maximum travel time allowed (minutes).
		prox_measure (str): Text ("length" or "time_min") used to choose a way to calculate time between nodes and points of interest.
							If "length", will assume a walking speed of 4km/hr.
							If "time_min", edges with time information must be provided.
							Defaults to "length".
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".

	Returns:
		geometry (geometry): with the covered area.
	"""

    # Define projection for downloaded data
	nodes = nodes.set_crs("EPSG:4326")
	edges = edges.set_crs("EPSG:4326")
	point_of_interest = nodes.loc[nodes['osmid'] == center_osmid].copy()

    # 1.0 --------------- ASSIGN CENTER NODE TO NEAREST OSMnx NODE
    # Find nearest osmnx node to center node
	nearest = find_nearest(G, nodes, point_of_interest, return_distance= True)
	nearest = nearest.set_crs("EPSG:4326")

    # 2.0 --------------- FORMAT NETWORK DATA
    # Fill NANs in length with calculated length (prevents crash)
	no_length = len(edges.loc[edges['length'].isna()])
	edges = edges.to_crs(projected_crs)
	edges['length'].fillna(edges.length,inplace=True)
	edges = edges.to_crs("EPSG:4326")
	if no_length > 0:
		print(f"Calculated length for {no_length} edges that had no length data.")

    # If prox_measure = 'length', calculates time_min assuming walking speed = 4km/hr
	if prox_measure == 'length':
		edges['time_min'] = (edges['length']*60)/4000
	else:
        # NaNs in time_min? --> Assume walking speed = 4km/hr
		no_time = len(edges.loc[edges['time_min'].isna()])
		edges['time_min'].fillna((edges['length']*60)/4000,inplace=True)
		if no_time > 0:
			print(f"Calculated time for {no_time} edges that had no time data.")

    # 3.0 --------------- PROCESS DISTANCE
	count_pois = (True,trip_time)
	nodes_analysis = nodes.reset_index().copy()
	nodes_time = nodes.copy()

    # Calculate distances
	poi_name = 'poi' #Required by function, has no effect on output
	nodes_distance_prep = calculate_distance_nearest_poi(nearest, nodes_analysis, edges, poi_name,'osmid', wght='time_min',count_pois=count_pois)
    # Extract from nodes_distance_prep the calculated pois count.
	nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_distance_prep[f'{poi_name}_{count_pois[1]}min']

    # Organice and filter output data
	nodes_time.reset_index(inplace=True)
	nodes_time = nodes_time.set_crs("EPSG:4326")
	nodes_time = nodes_time[['osmid',f'{poi_name}_{count_pois[1]}min','x','y','geometry']]

    # 4.0 --------------- GET ISOCHRONE FOR CURRENT CENTER NODE
    # Keep only nodes where nearest was found at an _x_ time distance
	nodes_at_15min = nodes_time.loc[nodes_time[f"{poi_name}_{count_pois[1]}min"]>0]

    # Create isochrone using convex hull to those nodes and add osmid from which this isochrone formed
	hull_geometry = nodes_at_15min.unary_union.convex_hull

	return hull_geometry


def id_pois_time(G, nodes, edges, pois, poi_name, prox_measure, walking_speed, goi_id, count_pois=(False,0), projected_crs="EPSG:6372",
				 preprocessed_nearest=(False,'dir')): # proximity
	""" Finds time from each node to nearest poi (point of interest). Function to be used when pois came from another geometry.
		Function id_pois_time takes into account an ID column that contains information of origin of each poi.
        The original geometry of interest (goi)'s unique ID is called goi_id.
		[e.g. parks with a unique park ID converted to vertexes, or bike lanes with a unique bike lane ID divided in several points].
	Args:
		G (networkx.MultiDiGraph): Graph with edge bearing attributes
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes within boundaries
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges within boundaries
		pois (geopandas.GeoDataFrame): GeoDataFrame with points of interest
		poi_name (str): Text containing name of the point of interest being analysed
		prox_measure (str): Text ("length" or "time_min") used to choose a way to calculate time between nodes and points of interest.
							If "length", will use walking speed.
							If "time_min", edges with time information must be provided.
		walking_speed (float): Decimal number containing walking speed (in km/hr) to be used if prox_measure="length",
							   or if prox_measure="time_min" but needing to fill time_min NaNs.
        goi_id (str): Text containing name of column with unique ID for the geometry of interest from which pois where created.
		count_pois (tuple, optional): tuple containing boolean to find number of pois within given time proximity. Defaults to (False, 0)
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
		preprocessed_nearest (tuple, optional): tuple containing boolean to use a previously calculated nearest file located in a local directory.
										Used when calculating proximity in a non-continous network while keeping actual nearest nodes to each poi
										from an original continous network. Defaults to (False, 'dir).

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with nodes containing time to nearest source (s).
	"""
    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # DIFFERENCES WITH POIS_TIME() EXPLANATION:
    # Whenever pois are extracted from other types of geometries or interest (goi)s (e.g. lines, polygons), a unique ID could be assigned.
    # That's because when wanting to find access to one geometry of interest (goi) (e.g. one bike lane or one park), we tend to consider that geometry as one poi,
    # and do not want to measure access to ALL pois created from that one geometry of interest.
    # (e.g. all the vertexes of a park or a every point every 100 meters for a bike lane)
    # Without this ADAPTATION of function pois_time(), when using count_pois we would be counting every vertex, every subdivision.

    # Since one geometry of interest (goi) has several pois (e.g. extracted parks vertices or divided a bike lane in several points),
    # any OSMnx node would get assigned (STEP 1: NEAREST) to several pois even if they all belong to the same geometry of interest (goi).
    # The **first main ADAPTATION** works so that after nearest function, a given node gets assigned to the **closest** poi of
    # the geometry of interest (goi) only (Considers a given geometry of interest (goi) once and discard the rest).
    # The **second main ADAPTATION** (STEP 2: DISTANCE NEAREST POI) works so that proximity is measured for each geometry of interest (goi) and data is not repeated.

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # FOLLOWING CODE IS THE SAME IN FUNCTION POIS_TIME()

    ##########################################################################################
    # STEP 1: NEAREST.
    # Finds and assigns nearest node OSMID to each point of interest.

    # Defines projection for downloaded data
	pois = pois.set_crs("EPSG:4326")
	nodes = nodes.set_crs("EPSG:4326")
	edges = edges.set_crs("EPSG:4326")

    # In case there are no amenities of the type in the city, prevents it from crashing if len = 0
	if len(pois) == 0:
		nodes_time = nodes.copy()

        # Format
		nodes_time.reset_index(inplace=True)
		nodes_time = nodes_time.set_crs("EPSG:4326")

        # As no amenities were found, output columns are set to nan.
		nodes_time['time_'+poi_name] = np.nan # Time is set to np.nan.
		print(f"0 {poi_name} found. Time set to np.nan for all nodes.")
		if count_pois[0]:
			nodes_time[f'{poi_name}_{count_pois[1]}min'] = np.nan # If requested pois_count, value is set to np.nan.
			print(f"0 {poi_name} found. Pois count set to nan for all nodes.")
			nodes_time = nodes_time[['osmid','time_'+poi_name,f'{poi_name}_{count_pois[1]}min','x','y','geometry']]
			return nodes_time
		else:
			nodes_time = nodes_time[['osmid','time_'+poi_name,'x','y','geometry']]
			return nodes_time
	else:
		if preprocessed_nearest[0]:
			# Load precalculated (with entire network) nearest gdf.
			nearest = gpd.read_file(preprocessed_nearest[1]+f"nearest_{poi_name}.gpkg")
			# Filter nearest by keeping osmids located in current nodes (filtered network) gdf.
			osmid_check_list = list(nodes.reset_index().osmid.unique())
			nearest = nearest.loc[nearest.osmid.isin(osmid_check_list)]
		else:
			### Find nearest osmnx node for each DENUE point.
			nearest = find_nearest(G, nodes, pois, return_distance= True)
			nearest = nearest.set_crs("EPSG:4326")
			print(f"Found and assigned nearest node osmid to each {poi_name}.")

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ADAPTATION - FOLLOWING CODE DOES NOT EXIST IN POIS_TIME()

			# Up to this point 'nearest' has all osmnx nodes closest to a given poi (Originated from a given geometry of interest (goi)),
			# one osmnx node might be assigned to 2 or more poi of the SAME geometry of interest (goi).
			# (For example, node 54 is closest to poi 12 and poi 13, both vertexes from the same park).

			# If we leave it like that, that nearest will be repeted (e.g. the same park assigned twice to the same node) even if it is just close to 1 goi.
			# This step keeps the minimum distance (distance_node) from node osmid to each poi when originating from the same geometry of interest (goi),
			# so that if one node is close to 5 pois of the same goi, it keeps only 1 node assigned to 1 poi, not 5.

			# Group by node (osmid) and polygon (green space) considering only the closest vertex (min)
			groupby = nearest.groupby(['osmid',goi_id]).agg({'distance_node':np.min})

			# Turns back into gdf merging back with nodes geometry
			geom_gdf = nodes.reset_index()[['osmid','geometry']]
			groupby.reset_index(inplace=True)
			nearest = pd.merge(groupby,geom_gdf,on='osmid',how='left')
			nearest = gpd.GeoDataFrame(nearest, geometry="geometry")

			# Filters for pois assigned to nodes at a maximum distance of 80 meters (aprox. 1 minute)
			# That is to consider a 1 minute additional walk as acceptable (if goi is inside a park, e.g. a bike lane).
			nearest = nearest.loc[nearest.distance_node <= 80]

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # FOLLOWING CODE IS THE SAME IN FUNCTION POIS_TIME()

        ##########################################################################################
        # STEP 2: DISTANCE NEAREST POI.
        # Calculates distance from each node to its nearest point of interest using previously assigned nearest node.

        # 2.1 --------------- FORMAT NETWORK DATA
        # Fill NANs in length with calculated length (prevents crash)
		no_length = len(edges.loc[edges['length'].isna()])
		edges = edges.to_crs(projected_crs)
		edges['length'].fillna(edges.length,inplace=True)
		edges = edges.to_crs("EPSG:4326")
		print(f"Calculated length for {no_length} edges that had no length data.")

        # If prox_measure = 'length', calculates time_min using walking_speed
		if prox_measure == 'length':
			edges['time_min'] = (edges['length']*60)/(walking_speed*1000)
		else:
            # NaNs in time_min? --> Use walking_speed
			no_time = len(edges.loc[edges['time_min'].isna()])
			edges['time_min'].fillna((edges['length']*60)/(walking_speed*1000),inplace=True)
			print(f"Calculated time for {no_time} edges that had no time data.")

        # --------------- 2.2 ELEMENTS NEEDED OUTSIDE THE ANALYSIS LOOP
        # The pois are divided by batches of 200 or 250 pois and analysed using the function calculate_distance_nearest_poi.
        # nodes_analysis is a nodes gdf (index reseted) used in the function aup.calculate_distance_nearest_poi.
		nodes_analysis = nodes.reset_index().copy()
        # nodes_time: int_gdf stores, processes time data within the loop and returns final gdf. (df_int, df_temp, df_min and nodes_distance in previous code versions)
		nodes_time = nodes.copy()

        # --------------- 2.3 PROCESSING DISTANCE
		print (f"Starting time analysis for {poi_name}.")

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # ADAPTATION - FOLLOWING CODE IS DIFFERENT IN POIS_TIME()

    # At this point pois are in the network, pois are located in nearest.
    # But without ADAPTATION, for each node being analysed we might find in OSMnx Network many (e.g. 5) pois (nearest nodes) belonging to the same geometry of interest (goi).
    # For example, a node close to a park might find different 5 nodes where the park is near. But it is the same park!
    # We need to know, for each node, closest proximity to geometry of interest (goi).

    # This modified step iterates over goi_id so that when a node finds n close pois of the same geometry of interest (goi) (using pois_count),
    # it only gets assigned '1' (As we iterate over goi_id, no matter how many times we calculate proximity to nearest, it is the same geometry of interest
    # (goi), it is the same park).

		# ///
		# LOG CODE - Progress logs
		# Will create progress logs when progress reaches these percentages:
		progress_logs = [0,10,20,30,40,50,60,70,80,90,100]
		# ///

		gois_list = list(nearest[goi_id].unique())
		g = 1
		for goi in gois_list:

			# ///
			# LOG CODE - Progress logs
			# Measures current progress, prints if passed a checkpoint of progress_logs list.
			current_progress = (g / len(gois_list))*100
			for checkpoint in progress_logs:
				if current_progress >= checkpoint:
					print(f"Calculating proximity data for geometry of interest (goi) {g} of {len(gois_list)} for {poi_name}.")
					print(f"{checkpoint}% done.")
					progress_logs.remove(checkpoint)
					break
			# ///

            # Calculate
            # ADAPTATION - Dividing by batches of n pois is not necessary, since we will be examining batches of a small number of pois
            # (The pois belonging to a specific geometry of interest (goi)).
            # (e.g. all source_process will be the nodes closest to the vertexes of 1 park)
			source_process = nearest.loc[nearest[goi_id] == goi]
			nodes_distance_prep = calculate_distance_nearest_poi(source_process, nodes_analysis, edges, poi_name, 'osmid', wght='time_min',count_pois=count_pois)

            # Extract from nodes_distance_prep the calculated time data
            # ADAPTATION - Since no batches are used, we don't have 'batch_time_col', just current process.
			process_time_col = 'time_process_'+poi_name
			nodes_time[process_time_col] = nodes_distance_prep['dist_'+poi_name]

            # If requested, extract from nodes_distance_prep the calculated pois count
            # ADAPTATION - Since no batches are used, we don't have 'batch_poiscount_col', just current process.
			if count_pois[0]:
				process_poiscount_col = f'{poi_name}_process_{count_pois[1]}min'
				nodes_time[process_poiscount_col] = nodes_distance_prep[f'{poi_name}_{count_pois[1]}min']

                # ADAPTATION - Since we are only analysing one geometry of interest (goi), no node should have more than one pois count.
                # This is easly fixed if count>0, set to 1.
				tmp_poiscount_col = f'{poi_name}_tmp_{count_pois[1]}min'
				nodes_time[tmp_poiscount_col] = nodes_time[process_poiscount_col].apply(lambda x: 1 if x > 0 else 0)
				nodes_time[process_poiscount_col] = nodes_time[tmp_poiscount_col]
				nodes_time.drop(columns=[tmp_poiscount_col],inplace=True)

            # ADAPTATION - After this geometry of interest (goi)'s processing is over, find final output values for all currently examined gois.
            # ADAPTATION FOR TIME DATA:
            # If it is the first goi, assign first goi time
			if g == 1:
				print(f"First geometry of interest (goi)'s time.")
				nodes_time['time_'+poi_name] = nodes_time[process_time_col]
            # Else, apply the min function to find the minimum time so far
			else:
				time_cols = ['time_'+poi_name, process_time_col]
				nodes_time['time_'+poi_name] = nodes_time[time_cols].min(axis=1)
            # ADAPTATION FOR COUNT DATA (If requested)
            # If it is the first goi, assign first goi count
			if count_pois[0]:
				if g == 1:
					print(f"First batch count.")
					nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_time[process_poiscount_col]
				# Else, apply the sum function to find the total count so far
				else:
					count_cols = [f'{poi_name}_{count_pois[1]}min', process_poiscount_col]
					nodes_time[f'{poi_name}_{count_pois[1]}min'] = nodes_time[count_cols].sum(axis=1)

			g = g+1

		print(f"Finished time analysis for {poi_name}.")

    # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    # FOLLOWING CODE IS THE SAME IN FUNCTION POIS_TIME()

        ##########################################################################################
        # STEP 3: FINAL FORMAT.
        # Organices and filters output data.

		nodes_time.reset_index(inplace=True)
		nodes_time = nodes_time.set_crs("EPSG:4326")
		if count_pois[0]:
			nodes_time = nodes_time[['osmid','time_'+poi_name,f'{poi_name}_{count_pois[1]}min','x','y','geometry']]
			return nodes_time
		else:
			nodes_time = nodes_time[['osmid','time_'+poi_name,'x','y','geometry']]
			return nodes_time

def voronoi_points_within_aoi(area_of_interest, points, points_id_col, admissible_error=0.01, projected_crs="EPSG:6372"): # various
	""" Creates voronoi polygons within a given area of interest (aoi) from n given points.
	Args:
		area_of_interest (geopandas.GeoDataFrame): GeoDataFrame with area of interest (Determines output extents).
		points (geopandas.GeoDataFrame): GeoDataFrame with points of interest.
		points_id_col (str): Name of points ID column (Will be assigned to each resulting voronoi polygon)
		admissible_error (int, optional): Percentage of error (difference) between the input area (area_of_interest) and output area (dissolved voronoi polygons).
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with voronoi polygons (each containing the point ID it originated from) extending all up to the area of interest extent.
	"""

	# Set area of interest and points of interest for voronoi analysis to crs:6372 (Proyected)
	aoi = area_of_interest.to_crs(projected_crs)
	pois = points.to_crs(projected_crs)

    # Distance is a number used to create a buffer around the polygon and coordinates along a bounding box of that buffer.
    # Starts at 100 (works for smaller polygons) but will increase itself automatically until the diference between the area of
    # the voronoi polygons created and the area of the aoi is less than the admissible_error.
	distance = 100

    # Goal area (Area of aoi)
	# Objective is that diff between sum of all voronois polygons and goal area is within admissible error.
	goal_area_gdf = aoi.copy()
	goal_area_gdf['area'] = goal_area_gdf.geometry.area
	goal_area = goal_area_gdf['area'].sum()

	# Kick start while loop by creating area_diff
	area_diff = admissible_error + 1
	while area_diff > admissible_error:
		# Create a rectangular bound for the area of interest with a {distance} buffer.
		polygon = aoi['geometry'].unique()[0]
		bound = polygon.buffer(distance).envelope.boundary

		# Create points along the rectangular boundary every {distance} meters.
		boundarypoints = [bound.interpolate(distance=d) for d in range(0, np.ceil(bound.length).astype(int), distance)]
		boundarycoords = np.array([[p.x, p.y] for p in boundarypoints])

		# Load the points inside the polygon
		coords = np.array(pois.get_coordinates())

		# Create an array of all points on the boundary and inside the polygon
		all_coords = np.concatenate((boundarycoords, coords))

		# Calculate voronoi to all coords and create voronois gdf (No boundary)
		vor = Voronoi(points=all_coords)
		lines = [shapely.geometry.LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
		polys = shapely.ops.polygonize(lines)
		unbounded_voronois = gpd.GeoDataFrame(geometry=gpd.GeoSeries(polys), crs=projected_crs)

		# Add nodes ID data to voronoi polygons
		unbounded_voronois = gpd.sjoin(unbounded_voronois,pois[[points_id_col,'geometry']])

		# Clip voronoi with boundary
		bounded_voronois = gpd.overlay(df1=unbounded_voronois, df2=aoi, how='intersection')

		# Change back crs
		voronois_gdf = bounded_voronois.to_crs('EPSG:4326')

		# Area check for while loop
		voronois_area_gdf = voronois_gdf.to_crs(projected_crs)
		voronois_area_gdf['area'] = voronois_area_gdf.geometry.area
		voronois_area = voronois_area_gdf['area'].sum()
		area_diff = ((goal_area - voronois_area)/(goal_area))*100
		if area_diff > admissible_error:
			print(f'Error = {round(area_diff,2)}%. Repeating process.')
			distance = distance * 10
		else:
			print(f'Error = {round(area_diff,2)}%. Admissible.')

	# Out of the while loop:
	return voronois_gdf
