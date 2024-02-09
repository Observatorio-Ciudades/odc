################################################################################
# Module: analysis.py
# Set of data and spatial data analysis functions
# updated: 15/11/2023
################################################################################

import igraph as ig
import numpy as np
import networkx as nx
from .utils import *
from .data import *
import math
from scipy import optimize
from math import sqrt




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

def get_distances(g, seeds, weights, voronoi_assignment, get_nearest_poi=False):
	"""
	Distance for the shortest path for each node to the closest seed using Dijkstra's algorithm.
	Arguments:
		g (igraph.Graph): Graph object that calculates the shortest path between two nodes
		seeds (numpy.array): Find the shortest path from each node to the closest seed
		weights (numpy.array): Specify the weights of each edge, which is used to calculate the shortest path
		voronoi_assignment (numpy.array): Assign the nodes to their respective seeds
	Returns: 
		The distance for the shortest path for each node to the closest seed
	"""
	shortest_paths = np.array(g.shortest_paths_dijkstra(seeds,weights=weights))
	distances = [np.min(shortest_paths[:,i]) for i in range(len(voronoi_assignment))]
	if get_nearest_poi:
		nearest_poi_idx = [np.argmin(shortest_paths[:,i]) for i in range(len(voronoi_assignment))]
		return distances, nearest_poi_idx
	return distances

def calculate_distance_nearest_poi(gdf_f, nodes, edges, amenity_name, column_name, 
wght='length', get_nearest_poi=(False, 'poi_id_column'), max_distance=(0,'distance_node')):
	"""
	Calculate the distance to the shortest path to the nearest POI (in gdf_f) for all the nodes in the network G

	Arguments:
		gdf_f (geopandas.GeoDataFrame): GeoDataFrame with the Points of Interest the geometry type has to be shapely.Point
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network analysis
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network analysis
		amenity_name (str): string with the name of the amenity that is used as seed (pharmacy, hospital, shop, etc.) 
		column_name (str): column name where the nearest distance index is stored
		wght (str): weights column in edges. Defaults to length
		get_nearest_poi (tuple): tuple containing boolean to get the nearest POI and column name that contains that value. Defaults to (False, 'poi_id_column')
		max_distance (tuple): tuple containing limits for distance to node and column name that contains that value. Defaults to (0, distance_node)

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with geometry and distance to the nearest POI
	"""
	nodes = nodes.copy()
	edges = edges.copy()
	if max_distance[0] > 0:
		gdf_f = gdf_f.loc[gdf_f[max_distance[1]]<=max_distance[0]]
	g, weights, node_mapping = to_igraph(nodes,edges,wght=wght) #convert to igraph to run the calculations
	col_weight = f'dist_{amenity_name}'
	seeds = get_seeds(gdf_f, node_mapping, column_name)
	voronoi_assignment = voronoi_cpu(g, weights, seeds)
	if get_nearest_poi[0]:
		distances, nearest_poi_idx = get_distances(g,seeds,weights,voronoi_assignment,get_nearest_poi=True)
		nearest_poi = [gdf_f.iloc[i][get_nearest_poi[1]] for i in nearest_poi_idx]
		nodes['nearest_poi'] = nearest_poi
	else:
		distances = get_distances(g,seeds,weights,voronoi_assignment)

	nodes[col_weight] = distances

	nodes.replace([np.inf, -np.inf], np.nan, inplace=True)
	idx = pd.notnull(nodes[col_weight])
	nodes = nodes[idx].copy()

	return nodes


def group_by_hex_mean(nodes, hex_bins, resolution, col_name, osmid=True):
	"""
	Group by hexbin the nodes and calculate the mean distance from the hexbin to the closest amenity

	Arguments:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with the nodes to group
		hex_bins (geopandas.GeoDataFrame): GeoDataFrame with the hexbins
		resolution (int): resolution of the hexbins, used when doing the group by and to save the column
		amenity_name (str): string with the name of the amenity that is used as seed (pharmacy, hospital, shop, etc.)

	Returns:
		geopandas.GeoDataFrame:  GeoDataFrame with the hex_id{resolution}, geometry and average distance to amenity for each hexbin
	"""
	dist_col = col_name
	nodes = nodes.copy()
	nodes_in_hex = gpd.sjoin(nodes, hex_bins)
	nodes_hex = nodes_in_hex.groupby([f'hex_id_{resolution}']).mean()
	hex_new = pd.merge(hex_bins,nodes_hex,right_index=True,left_on=f'hex_id_{resolution}',how = 'outer')
	if osmid:
		hex_new = hex_new.drop(['index_right','osmid'],axis=1)
	else:
		hex_new = hex_new.drop(['index_right'],axis=1)
	hex_new[dist_col].apply(lambda x: x+1 if x==0 else x )
	hex_new.fillna(0, inplace=True)
	return hex_new

def socio_polygon_to_points(
	nodes,
	gdf_socio,
	column_start=0,
	column_end=-1,
	cve_column="CVEGEO",
	avg_column=None,
):
	"""
	Assign the proportion of sociodemographic data from polygons to points
	Arguments:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with the nodes to group
		gdf_socio (geopandas.GeoDataFrame): GeoDataFrame with the sociodemographic attributes of each AGEB
		column_start (int, optional): Column position were sociodemographic data starts in gdf_population. Defaults to 0.
		column_end (int, optional): Column position were sociodemographic data ends in gdf_population. Defaults to -1.
		cve_column (str, optional): Column name with unique code for identification. Defaults to "CVEGEO".
		avg_column (list, optional): Column name lists with data to average and not divide. Defaults to None.
	Returns:
		nodes (GeoDataFrame):  Shows the proportion of population by nodes in the AGEB
	"""

	if column_end == -1:
		column_end = len(list(gdf_socio.columns))

	if avg_column is None:
		avg_column = []
	totals = (
		gpd.sjoin(nodes, gdf_socio)
		.groupby(cve_column)
		.count()
		.rename(columns={"x": "nodes_in"})[["nodes_in"]]
		.reset_index()
	)  # caluculate the totals
	# get a temporal dataframe with the totals and columns
	temp = pd.merge(gdf_socio, totals, on=cve_column)
	# get the average for the values
	for col in temp.columns.tolist()[column_start:column_end]:
		if col not in avg_column:
			temp[col] = temp[col] / temp["nodes_in"]
	temp = temp.set_crs("EPSG:4326")
	nodes = gpd.sjoin(nodes, temp)
	nodes.drop(["nodes_in", "index_right"], axis=1, inplace=True)  # drop the nodes_in column
	return nodes  # spatial join the nodes with the values

def socio_points_to_polygon(
	gdf_polygon,
	gdf_socio,
	cve_column,
	string_columns,
	wgt_dict=None,
	avg_column=None):

	"""Group sociodemographic point data in polygons
    Arguments:
        gdf_polygon (geopandas.GeoDataFrame): GeoDataFrame polygon where sociodemographic data will be grouped
        gdf_socio (geopandas.GeoDataFrame): GeoDataFrame points with sociodemographic data
        cve_column (str): Column name with polygon id in gdf_polygon.
        string_columns (list): List with column names for string data in gdf_socio.
        column_start (int, optiona): Column position were sociodemographic data starts in gdf_socio. Defaults to 0.
        column_end (int, optional): Column position were sociodemographic data ends in gdf_socio. Defaults to -1.
        wgt_dict {dict, optional): Dictionary with average column names and weight column names for weighted average. Defaults to None.
        avg_column (list, optional): List with column names with average data. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame with group sociodemographic data and polygon id

	"""

	dictionary_list = []
	# Adds census data from points to polygon
	gdf_tmp = gpd.sjoin(gdf_socio, gdf_polygon)  # joins points to polygons

	# convert data types
	all_columns = list(gdf_socio.columns)
	numeric_columns = [x for x in all_columns if x not in string_columns]
	type_dict = {"string": string_columns, "float": numeric_columns}
	gdf_tmp = convert_type(gdf_tmp, type_dict)

	#group sociodemographic points to polygon
	for idx in gdf_tmp[cve_column].unique():

		socio_filter = gdf_tmp.loc[gdf_tmp[cve_column]==idx].copy()

		dict_tmp = group_sociodemographic_data(socio_filter, numeric_columns,
		avg_column=avg_column, avg_dict=wgt_dict)
		
		dict_tmp[cve_column] = idx
		
		dictionary_list.append(dict_tmp)
	
	data = pd.DataFrame.from_dict(dictionary_list)

	return data

def group_sociodemographic_data(df_socio, numeric_cols, avg_column=None, avg_dict=None):
	
	"""
    Aggregate sociodemographic variables from DataFrame.
    Arguments:
        df_socio (pd.DataFrame): DataFrame containing sociodemographic variables to be aggregated by sum or mean.
        column_start (int, optional): Column number were sociodemographic variables start at DataFrame. Defaults to 1.
        column_end (int, optional): Column number were sociodemographic variables end at DataFrame. Defaults to -1.
        avg_column (list, optional): List of column names to be averaged and not sum. Defaults to None.
        avg_dict (dict, optional): Dictionary containing column names to average and
                                            column with which a weighted average will be crated. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame with sum and mean values for sociodemographic data
    """
	# column names with sociodemographic data
	if 'geometry' in numeric_cols:
		numeric_cols.remove('geometry')
	socio_cols = numeric_cols

	# Dictionary to store aggregated variables
	group_dict = {}

	if avg_column is None:
		# creates empty lists to avoid crash for None
		avg_column = []
		avg_dict = []
	# iterate over columns: mean or sum
	for col in socio_cols:
		if col in avg_column:
			# creates weighted averages
			pop_weight = df_socio[avg_dict[col]].sum()
			if pop_weight == 0:
				group_dict[col] = 0
			else:
				tmp_df = df_socio[[avg_dict[col], col]].groupby(col).sum().reset_index()
				tmp_df["weight"] = tmp_df[col] * tmp_df[avg_dict[col]]
				tmp_df["wavg"] = tmp_df["weight"] / pop_weight
				group_dict[col] = tmp_df["wavg"].sum()
		else:
			group_dict[col] = df_socio[col].sum()

	return group_dict

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


def create_network(nodes, edges):

	"""
	Create a network based on nodes and edges without unique ids and to - from attributes.

	Arguments:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with nodes for network in EPSG:4326
		edges (geopandas.GeoDataFrame): GeoDataFrame with edges for network in EPSG:4326

	Returns:
		geopandas.GeoDataFrame: nodes GeoDataFrame with unique ids based on coordinates named osmid in EPSG:4326
		geopandas.GeoDataFrame: edges GeoDataFrame with to - from attributes based on nodes ids named u and v respectively in EPSG:4326
	"""

	#Copy edges and nodes to avoid editing original GeoDataFrames
	nodes = nodes.copy()
	edges = edges.copy()

	#Create unique ids for nodes and edges
	##Change coordinate system to meters for unique ids
	nodes = nodes.to_crs("EPSG:6372")
	edges = edges.to_crs("EPSG:6372")

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
	edges['length'] = edges.to_crs("EPSG:6372").length
	
	#Change osmid to integer
	nodes['osmid'] = nodes.osmid.astype(int)

	#Transform coordinates
	nodes = nodes.to_crs("EPSG:4326")
	edges = edges.to_crs("EPSG:4326")

	return nodes, edges

	
def gdf_in_hex(grid, gdf, resolution = 10, contain= True):

	"""
	Finds the hexagons that have or do not have a point within

	Arguments:
		grid (geopandas.GeoDataFrame): GeoDataFrame with the full H3 hex grid of the city
		resolution (int): resolution of the hexbins, used when doing the group by and to save the column
		gdf (geopandas.GeoDataFrame): GeoDataFrame of figures to be overlaid with hexes
		contain (str): True == hexes that have at least a point / False == hexes that DO NOT contain at least a point

		
	Returns:
		geopandas.GeoDataFrame: gdf_in_hex: hexes that contain or do not contain a gdf within
	"""
	#PIP (Point in Polygon). Overlays gdf with hexes to find hexes that have the gdf in them and those that do not
	pip = gpd.overlay(grid, gdf, how='intersection', keep_geom_type=False)
	pip = pip.set_index(f'hex_id_{resolution}')
	grid = grid.set_index(f'hex_id_{resolution}')
	#simplify and keep only relevant columns
	pip_idx = pip.drop(['geometry'], axis=1)
	hex_geom = grid[['geometry']]
	#Merge with indicator. Right only means that the hexagon does NOT have any node
	hex_merge = pip_idx.merge(hex_geom, left_index=True, right_index=True, how='outer', indicator=True)
	## False means it will find hexes without points
	if contain == False:
		hex_node = hex_merge[hex_merge['_merge']=='right_only']
	## False means it will find hexes with points
	if contain == True:
		hex_node = hex_merge[hex_merge['_merge']=='both']
	#(simplify)
	point_hex = gpd.GeoDataFrame(hex_node, geometry = 'geometry')
	point_hex = point_hex[['geometry']]
	point_hex.reset_index(inplace = True)
	grid.reset_index(inplace = True)
	point_hex = point_hex.drop_duplicates(subset=[f'hex_id_{resolution}'])

	return point_hex


def fill_hex(missing_hex, data_hex, resolution, data_column):

	"""
	Fills hexagons with no data with the average data of neighbors (note: hex id must be a column not index)

	Arguments:
		missing_hex (geopandas.GeoDataFrame): GeoDataFrame with the hexes that does not have the information due to lack of nodes
		data_hex (geopandas.GeoDataFrame): GeoDataFrame of hexes that do contain the information
		resolution (int): resolution of the hexbins, used when doing the group by and to save the column
		data_column (str): Name of the column with the data to be filled with (ex. distance)

		
	Returns:
		geopandas.GeoDataFrame - full_hex: hexgrid filled with relevant data
	"""
	missing_hex[[f'{data_column}']] = np.nan
	urb_hex = gpd.GeoDataFrame()
	urb_hex = data_hex.append(missing_hex)
	urb_hex = urb_hex.set_index(f'hex_id_{resolution}')
	## Start looping
	count = 0
	iter = 1
	urb_hex[f'{data_column}'+ str(count)] = urb_hex[f'{data_column}'].copy()
	while urb_hex[f'{data_column}'+str(count)].isna().sum() > 0:
		missing = urb_hex[urb_hex[f'{data_column}'+str(count)].isna()]
		urb_hex[f'{data_column}'+ str(iter)] = urb_hex[f'{data_column}'+str(count)].copy()
		for idx,row in missing.iterrows():
			###Cell 1
			near = pd.DataFrame(h3.k_ring(idx,1))
			near[f'hex_id_{resolution}'] = h3.k_ring(idx,1)
			near['a'] = np.nan
			near= near.set_index(f'hex_id_{resolution}')
			###Cell 2
			neighbors = near.merge(urb_hex, left_index=True, right_index=True, how='left')
			#Cell 3
			average = neighbors[f'{data_column}'+str(count)].mean()
			urb_hex.at[idx, f'{data_column}'+str(iter)] = average
		count = count + 1
		iter = iter + 1
	full_hex = urb_hex[['geometry']]
	full_hex[f'{data_column}'] = urb_hex[f'{data_column}'+ str(count)].copy()

	return full_hex

def calculate_isochrone(G, center_node, trip_time, dist_column, subgraph=False):
    """
		Function that that creates a isochrone from a center_node in graph G,
		 and uses parameters like distance and time to plot the result.
    Arguments:
        G (networkx.Graph): networkx Graph with travel time (time) attribute.
        center_node (int): id of the node to use
        trip_time (int): maximum travel time allowed
		dist_column (int): column name with weight for calculation
        subgraph (bool, optional): Bool to get the resulting subgraph or only the geometry. Defaults to False.
    Returns:
        sub_G (optional): subgraph of the covered area.
        geometry (geometry): with the covered area
    """

    sub_G = nx.ego_graph(G, center_node, radius=trip_time, distance=dist_column)
    geometry = gpd.GeoSeries([Point((data["x"], data["y"])) for node, data in sub_G.nodes(data=True)]).unary_union.convex_hull
    if subgraph:
        return sub_G, geometry
    else:
        return geometry


def sigmoidal_function(x, di, d0):
	"""
	The sigmoidal_function function takes in two parameters, x and di.
	It is used to calculate the equilibrium index for each node.
	Arguments:
		x (int): Calculate the sigmoidal function
		di (int): Determine the slope of the sigmoid function
		d0 (int): Set the threshold of the sigmoid function
	Returns:
		idx_eq (int): Calculations of the index
	"""
	
	idx_eq = 1 / (1 + math.exp(x * (di - d0)))
	return idx_eq


def sigmoidal_function_constant(positive_limit_value, mid_limit_value):
	"""
	The sigmoidal_function_constant function calculates the constant average decay 
	for a sigmoidal funcition with 2 quarter values at 0.25 and 0.75 of the distance 
	between positive_limit_value and mid_limit_value and an input constant at x. 
	All values collected will be stored inside an index 
	Arguments:
		positive_limit_value: (int) Define the upper limit of the sigmoidal function
		mid_limit_value: (int) Define the midpoint of the sigmoidal function
	Returns:
	constant_value_average: (int) Average constant decay value for the two quarter times with optimized values from the index

	"""

	tmp_idx = [] # list that stores constant decay values for 0.25 and 0.75

	# calculate 0.75 quarter time
	quarter_limit = mid_limit_value - ((mid_limit_value-positive_limit_value)/2)
	idx_objective = 0.75

	
	def sigmoidal_function(x, di=quarter_limit, d0=mid_limit_value):
		"""
		The sigmoidal_function function calculates the value at x,
		taking the 2 predefined values quarter_limit and mid_limit_value calculated earlier 
			Arguments:
				x (int):  Defaults to quarter_limit.
				di (int):  Defaults to quarter_limit_value.
				d0 (int):  Defaults to mid_limit_value. 
			Returns:
		idx_eq (int):  The sigmoidal function of the independent variable x.
		"""
		idx_eq = 1 / (1 + math.exp(x * (di - d0)))
		return idx_eq

	def sigmoidal_function_condition(x, di=quarter_limit, d0=mid_limit_value, idx_0=idx_objective):
		"""
		The sigmoidal_function_condition takes in the following parameters:
			x - The value of x to be evaluated.
			di - The upper limit of the sigmoidal function.
			d0 - The midpoint of the sigmoidal function.  This is also where it crosses 0 on its y-axis, and where it has an index value equal to idx_0.
			idx_0 - A float between 0 and 1 representing how much we want our index values to be at d0 (the midpoint).
		Arguments:
			x (int): Calculate the value of the sigmoidal function
			di (int): Set the value of the sigmoid function at which it reaches its maximum
			d0 (int): Define the mid-limit value of the sigmoidal function
			idx_0 (int): Set the objective value of the sigmoid function
		Returns:
		The value of the sigmoidal function at a given point x
		"""
		

		return (1 / (1 + math.exp(x * (di - d0)))) - idx_0

	# search for constant decay value in 0.75 quarter_time
	cons = {'type':'eq', 'fun': sigmoidal_function_condition}
	result = optimize.minimize(sigmoidal_function, 0.01, constraints = cons)
	tmp_idx.append(result.x[0])

	# calculate 0.25 quarter time
	quarter_limit = mid_limit_value + ((mid_limit_value-positive_limit_value)/2)
	idx_objective = 0.25

	# search for constant decay value in 0.25 quarter_time
	cons = {'type':'eq', 'fun': sigmoidal_function_condition}
	result = optimize.minimize(sigmoidal_function, 0.01, constraints = cons)
	tmp_idx.append(result.x[0])

	# calculate average constant decay for 0.25 and 0.75
	constant_value_average = sum(tmp_idx) / len(tmp_idx)

	return constant_value_average

def interpolate_to_gdf(gdf, x, y, z, power=2, search_radius=None):
	"""
	Interpolate z values at x, y coordinates for a GeoDataFrame using inverse distance weighting (IDW).

	Args:
		gdf (geopandas.GeoDataFrame): GeoDataFrame containing points to which z values will be interpolated
		x (np.array): numpy array with x coordinates of observed points
		y (np.array): numpy array with y coordinates of observed points
		z (np.array): numpy array with z values at observed points
		power (int, optional): Exponential constant for distance decay function. Defaults to 2.
		search_radius (_type_, optional): Distance limit for IDW analysis. Defaults to None.

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with interpolated values in interpolated_value column
	"""
	
	gdf_int = gdf.copy()
	xi = np.array(gdf_int.geometry.x)
	yi = np.array(gdf_int.geometry.y)
	gdf_int['interpolated_value'] = idw_at_point(x, y, z, xi, yi, power, search_radius)
	return gdf_int


def idw_at_point(x0, y0 ,z0, xi, yi, power=2, search_radius=None):
	"""Calculate inverse distance weighted (IDW) interpolation at a single point.

	Args:
		x0 (np.array): numpy array with x coordinates of observed points
		y0 (np.array): numpy array with y coordinates of observed points
		z0 (np.array): numpy array with z values at observed points
		xi (float): x coordinate for interpolation point
		yi (float): y coordinate for interpolation point
		power (int, optional): Exponential constant for distance decay function. Defaults to 2.
		search_radius (int, optional): Distance limit for IDW analysis. Defaults to None.

	Returns:
		np.array: numpy array with calculated z values at xi, yi
	"""
	# filter analysis by search radius
	if search_radius:
		id_x = (x0 <= xi + search_radius) & (x0 >= xi - search_radius)
		id_y = (y0 <= yi + search_radius) & (y0 >= yi - search_radius)
		id_xy = id_x + id_y
		z0 = z0[np.squeeze(id_xy)].copy()
		obs = np.vstack((x0[id_xy], y0[id_xy])).T
	else:
		# format observed points data
		obs = np.vstack((x0, y0)).T

	# format interpolation point data
	interp = np.vstack((xi, yi)).T

	# calculate euclidean distance in x and y between obs and interp
	d0 = np.substract.outer(obs[:,0], interp[:,0])
	d1 = np.substract.outer(obs[:,1], interp[:,1])

	# calculate hypotenuse for distances
	dist = np.hypot(d0, d1)

	# filter distances by search radius
	if search_radius:
		idx = dist <= search_radius
		dist = dist[idx]
		z0 = z0[np.squeeze(idx)]

	# calculate weights
	weights = 1.0 * (dist + 1e-12)**power
	# weights sum to 1 by row
	weights /= weights.sum(axis=0)

	# check if no observation points are within limit distance
	if weights.shape[0] == 0:
		ones = np.ones((z0.shape[1],), dtype=float)
		ones[ones == 1] = -1 # return -1 vector
		return ones
	# calculate dot product of weight matrix and z value matrix
	int_value = np.dot(weights.T, z0)
	return int_value

def interpolate_at_points(x0, y0, z0, xi, yi, power=2, search_radius=None):
	"""Interpolate z values at a set of xi, yi coordinates using inverse distance weighting (IDW).

	Args:
		x0 (np.array): numpy array with x coordinates of observed points
		y0 (np.array): numpy array with y coordinates of observed points
		z0 (np.array): numpy array with z values at observed points
		xi (np.array): numpy array with x coordinates of interpolation points
		yi (np.array): numpy array with y coordinates of interpolation points
		power (int, optional): Exponential constant for distance decay function. Defaults to 2.
		search_radius (int, optional): Distance limit for IDW analysis. Defaults to None.

	Returns:
		np.array: numpy array with calculated z values at a series of x, y
	"""
    # format observed points data
	obs = np.vstack((x0, y0)).T

	# format interpolation points data
	interp = np.vstack((xi, yi)).T

	# calculate linear distance in x and y
	d0 = np.subtract.outer(obs[:,0], interp[:,0])
	d1 = np.subtract.outer(obs[:,1], interp[:,1])
	
	# calculate linear distance from observations to interpolation points
	dist = np.hypot(d0, d1)
	
	# filter data by search radius
	if search_radius:
		idx = dist<=search_radius
		idx_num = idx * 1
		idx_num = idx_num.astype('float32')
		idx_num[idx_num == 0] = np.nan
		dist = dist*idx_num
	
	# calculate weights
	weights = 1.0/(dist+1e-12)**power
	weights /= np.nansum(weights, axis=0)

	# caculate dot product of weight matrix and z value matrix
	int_value = np.where(np.isnan(weights.T),0,weights.T).dot(np.where(np.isnan(z0),0,z0))

	return int_value