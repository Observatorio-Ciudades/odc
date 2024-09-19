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
import shapely
from scipy.spatial import Voronoi
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


def group_by_hex_mean(nodes, hex_bins, resolution, group_column_names, hex_column_id, osmid=True):
	"""
	Group by hexbin the nodes and calculate the mean distance from the hexbin to the closest amenity

	Arguments:
		nodes (geopandas.GeoDataFrame): GeoDataFrame with the nodes to group
		hex_bins (geopandas.GeoDataFrame): GeoDataFrame with the hexbins
		resolution (int): resolution of the hexbins, used when doing the group by and to save the column
		group_column_names (str,list): column name or list of column names to group with
		hex_column_id (str): column name with the hex_id

	Returns:
		geopandas.GeoDataFrame:  GeoDataFrame with the hex_id{resolution}, geometry and average distance to amenity for each hexbin
	"""
	dist_col = group_column_names
	nodes = nodes.copy()
	nodes_in_hex = gpd.sjoin(nodes, hex_bins)
	# Group data by hex_id
	nodes_in_hex = nodes_in_hex.drop(columns=['geometry']) #Added this because it tried to calculate mean of geom
	nodes_hex = nodes_in_hex.groupby([hex_column_id]).mean()
	# Merge back to geometry
	hex_new = pd.merge(hex_bins,nodes_hex,right_index=True,left_on=hex_column_id,how = 'outer')
	if osmid:
		hex_new = hex_new.drop(['index_right','osmid'],axis=1)
	else:
		hex_new = hex_new.drop(['index_right'],axis=1)
	
	# Check for NaN values
	if type(dist_col) == list:
		for dc in dist_col:
			hex_new[dc].apply(lambda x: x+1 if x==0 else x )
	else:
		hex_new[dist_col].apply(lambda x: x+1 if x==0 else x )
	# Fill NaN values
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
	avg_column=None,
	include_nearest=(False, '_'),
	projected_crs="EPSG:6372"):

	"""Group sociodemographic point data in polygons
    Arguments:
        gdf_polygon (geopandas.GeoDataFrame): GeoDataFrame polygon where sociodemographic data will be grouped
        gdf_socio (geopandas.GeoDataFrame): GeoDataFrame points with sociodemographic data
        cve_column (str): Column name with polygon id in gdf_polygon.
        string_columns (list): List with column names for string data in gdf_socio.
		count_pois (tuple): 
        wgt_dict {dict, optional): Dictionary with average column names and weight column names for weighted average. Defaults to None.
        avg_column (list, optional): List with column names with average data. Defaults to None.
		include_nearest (tuple,optional): tuple containing boolean. If False, ignores points that fall outside gdf_polygon.
																	If True, find closest poly vertex to point and assigns the point to it.
																	Defaults to (False, '_')
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".
    Returns:
        pd.DataFrame: DataFrame with group sociodemographic data and polygon id

	"""

	dictionary_list = []
	# Adds census data from points to polygon
	gdf_tmp_1 = gpd.sjoin(gdf_socio, gdf_polygon)  # joins points to polygons
	
	if include_nearest[0]:
		points_id =  include_nearest[1]
        # Find points of gdf_socio that were left outside gdf_polygon by merging gdf_socio and gdf_tmp_1
		# and (through indicator) keep only those points found in gdf_socio and not in gdf_tmp_1 ('left_only')
		gdf_socio_merge = gdf_socio.merge(gdf_tmp_1[[points_id]], on=[points_id], how='left', indicator=True)
		gdf_socio_outside = gdf_socio_merge.loc[gdf_socio_merge['_merge']=='left_only']
		gdf_socio_outside = gdf_socio_outside.drop(columns=['_merge'])
		# Extract gdf_polygon's vertices
		gdf_poly_edges = gdf_polygon.copy()
		gdf_poly_edges['geometry'] = gdf_poly_edges.geometry.boundary
        # Find nearest gdf1 to each gdf2
		gdf1 = gdf_socio_outside.to_crs(projected_crs)
		gdf2 = gdf_poly_edges.to_crs(projected_crs)
		nearest = gpd.sjoin_nearest(gdf1, gdf2,lsuffix="left", rsuffix="right")
		gdf_tmp_2 = nearest.to_crs('EPSG:4326')
		
		gdf_tmp = pd.concat([gdf_tmp_1,gdf_tmp_2])
	
	else:
		gdf_tmp = gdf_tmp_1.copy()

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

def calculate_isochrone(G, center_node, trip_time, dist_column, undirected=True, subgraph=False):
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

    sub_G = nx.ego_graph(G, center_node, radius=trip_time, undirected=undirected, distance=dist_column)
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

def weighted_average(df, weight_column, value_column):
	"""
	Weighted average function that takes a DataFrame, weight column name and value column name as inputs
	Arguments:
		df (pandas.DataFrame): DataFrame containing the data to be averaged
		weight_column (str): Column name with weight data
		value_column (str): Column name with value data
	Returns:
		weighted_average (float): Weighted average of the value column
	"""
	weighted_average = (df[weight_column] * df[value_column]).sum() / df[weight_column].sum()
	return weighted_average

def create_popdata_hexgrid(aoi, pop_dir, index_column, pop_columns, res_list, projected_crs="EPSG:6372"):
	""" Function originally designed for proximity analysis in Latinamerica. It takes an area of interest, a population directory, 
		index and pop columns and a list of desired hex res outputs and groups sociodemographic data by hex.
	Args:
		aoi (geopandas.GeoDataFrame): GeoDataFrame polygon boundary for the area of interest.
		pop_dir (str): Directory (location) of the population file.
		index_column (str): Name of the unique index column within the population file.
		pop_columns (list): List of names of columns to be added by hexagon. 
							First item of list must be name of total population column in order to calculate density.
		res_list (list): List of integers containing hex resolutions for output.
		projected_crs (str, optional): string containing projected crs to be used depending on area of interest. Defaults to "EPSG:6372".

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with grouped sociodemographic data, hex_id and its resolution.
	"""

	# Read pop GeoDataFrame
	pop_gdf = gpd.read_file(pop_dir)
	pop_gdf = pop_gdf.to_crs("EPSG:4326")
	print(f"Loaded pop_gdf.")

	# Make sure all data is in .lower()
	pop_gdf.columns = pop_gdf.columns.str.lower()
	index_column = index_column.lower()
	columns_ofinterest = []
	for col in pop_columns:
		columns_ofinterest.append(col.lower())

	# Format gdf with all columns of interest
	columns_ofinterest.append(index_column)
	columns_ofinterest.append('geometry')
	block_pop = pop_gdf[columns_ofinterest]
	print(f"Filtered pop_gdf.")

	# Extract point from polygon
	block_pop = block_pop.to_crs(projected_crs)
	block_pop = block_pop.set_index(index_column)
	point_within_polygon = gpd.GeoDataFrame(geometry=block_pop.representative_point())
	print(f"Extracted pop_gdf centroids.")

	# Format centroids with pop data
	# Add census data to points
	centroid_block_pop = point_within_polygon.merge(block_pop, right_index=True, left_index=True) 
	# Format geometry column
	centroid_block_pop.drop(columns=['geometry_y'], inplace=True)
	centroid_block_pop.rename(columns={'geometry_x':'geometry'}, inplace=True)
	# Create GeoDataFrame with that geometry column
	centroid_block_pop = gpd.GeoDataFrame(centroid_block_pop, geometry='geometry')
	# Format population column
	centroid_block_pop.rename(columns={pop_columns[0]:'pobtot'},inplace=True)
	# General final formatting
	centroid_block_pop = centroid_block_pop.to_crs("EPSG:4326")
	centroid_block_pop = centroid_block_pop.reset_index()
	print(f"Converted to centroids with {centroid_block_pop.pobtot.sum()} " + f"pop vs {block_pop[pop_columns[0]].sum()} pop in original gdf.")

	# Create buffer for aoi to include outer blocks when creating hexgrid
	aoi_buffer = aoi.copy()
	aoi_buffer = aoi_buffer.dissolve()
	aoi_buffer = aoi_buffer.to_crs(projected_crs).buffer(2500)
	aoi_buffer = gpd.GeoDataFrame(geometry=aoi_buffer)
	aoi_buffer = aoi_buffer.to_crs("EPSG:4326")

	hex_socio_gdf = gpd.GeoDataFrame()

	for res in res_list:
		# Generate hexagon gdf
		hex_gdf = create_hexgrid(aoi_buffer, res)
		hex_gdf = hex_gdf.set_crs("EPSG:4326")

		# Format - Remove res from index name and add column with res
		hex_gdf.rename(columns={f'hex_id_{res}':'hex_id'},inplace=True)
		hex_gdf['res'] = res
		print(f"Created hex_grid with {res} resolution")

		# Group pop data
		string_columns = [index_column]
		hex_socio_df = socio_points_to_polygon(hex_gdf, centroid_block_pop,'hex_id', string_columns, projected_crs=projected_crs) 
		print(f"Agregated socio data to hex with a total of {hex_socio_df.pobtot.sum()} population for resolution {res}.")

		# Hexagons data to hex_gdf GeoDataFrame
		hex_socio_gdf_tmp = hex_gdf.merge(hex_socio_df, on='hex_id')
		
		# Calculate population density
		hectares = hex_socio_gdf_tmp.to_crs(projected_crs).area / 10000
		hex_socio_gdf_tmp['dens_pob_ha'] = hex_socio_gdf_tmp['pobtot'] / hectares 
		print(f"Calculated an average density of {hex_socio_gdf_tmp.dens_pob_ha.mean()}")
		
		# Concatenate in hex_socio_gdf, where (if more resolutions) next resolution will also be stored.
		hex_socio_gdf = pd.concat([hex_socio_gdf,hex_socio_gdf_tmp])    

	print(f"Finished calculating population by hexgrid for res {res_list}.")

	return hex_socio_gdf

def pois_time(G, nodes, edges, pois, poi_name, prox_measure, walking_speed, count_pois=(False,0), projected_crs="EPSG:6372",
			  preprocessed_nearest=(False,'dir')):
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
							   or if prox_measure="time_min" but needing to fill time_min NaNs.
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
	test_save_dir = "../data/external/debugging/pois_time/"

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
			# NaNs in time_min? --> Use walking speed
			no_time = len(edges.loc[edges['time_min'].isna()])
			edges['time_min'].fillna((edges['length']*60)/(walking_speed*1000),inplace=True)
			print(f"Calculated time for {no_time} edges that had no time data.")

		# 2.2 --------------- ELEMENTS NEEDED OUTSIDE THE ANALYSIS LOOP
		# The pois are divided by batches of 200 or 250 pois and analysed using the function calculate_distance_nearest_poi.
		
		# -----
		if test_save:
			nodes.to_file(test_save_dir + f"nodes_{poi_name}.geojson", driver='GeoJSON')
			edges.to_file(test_save_dir + f"edges_{poi_name}.geojson", driver='GeoJSON')
		# -----

		# nodes_analysis is a nodes gdf (index reseted) used in the function aup.calculate_distance_nearest_poi.
		nodes_analysis = nodes.reset_index().copy()

		# -----
		if test_save:
			nodes_analysis.to_file(test_save_dir + f"empty_nodes_analysis_{poi_name}.geojson", driver='GeoJSON')
		# -----

		# nodes_time: int_gdf stores, processes time data within the loop and returns final gdf. (df_int, df_temp, df_min and nodes_distance in previous code versions)
		nodes_time = nodes.reset_index().copy()

		# -----
		if test_save:
			nodes_time.to_file(test_save_dir + f"empty_nodes_time_{poi_name}.gpkg", driver='GPKG')
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
				print(f"Starting range k = {k+1} of {int(batch_size)+1} for {poi_name}.")
				# Calculate
				source_process = nearest.iloc[int(200*k):int(200*(1+k))].copy()
				nodes_distance_prep = calculate_distance_nearest_poi(source_process, nodes_analysis, edges, poi_name, 'osmid', wght='time_min',count_pois=count_pois)
				# Set osmid as column, not index
				nodes_distance_prep.drop(columns=['index'],inplace=True)
				nodes_distance_prep.reset_index(inplace=True)
				
				# -----
				if test_save:
					nodes_distance_prep.to_file(test_save_dir + f"nodes_distance_prep_{poi_name}_200batch{k}.gpkg", driver='GPKG')
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
				print(f"Starting range k = {k+1} of {int(batch_size)+1} for source {poi_name}.")
				# Calculate
				source_process = nearest.iloc[int(250*k):int(250*(1+k))].copy()
				nodes_distance_prep = calculate_distance_nearest_poi(source_process, nodes_analysis, edges, poi_name, 'osmid', wght='time_min',count_pois=count_pois)
				# Set osmid as column, not index
				nodes_distance_prep.drop(columns=['index'],inplace=True)
				nodes_distance_prep.reset_index(inplace=True)

				# -----
				if test_save:
					nodes_distance_prep.to_file(test_save_dir + f"nodes_distance_prep_{poi_name}_250batch{k}.gpkg", driver='GPKG')
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
			nodes_time.to_file(test_save_dir + f"nodes_time_{poi_name}.gpkg", driver='GPKG')
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


def weighted_average(df, weight_column, value_column):
	"""
	Weighted average function that takes a DataFrame, weight column name and value column name as inputs
	Arguments:
		df (pandas.DataFrame): DataFrame containing the data to be averaged
		weight_column (str): Column name with weight data
		value_column (str): Column name with value data
	Returns:
		weighted_average (float): Weighted average of the value column
	"""
	weighted_average = (df[weight_column] * df[value_column]).sum() / df[weight_column].sum()
	return weighted_average


def calculate_censo_nan_values_v1(pop_ageb_gdf,pop_mza_gdf,extended_logs=False):
	""" Calculates (and/or distributes, work in progress) values to NaN cells in population columns of INEGI's censo blocks gdf.
		As of this version, applies only to columns located in columns_of_interest list.
		As of this version, if couldn't find all values, distributes data of AGEB to blocks taking POBTOT in those blocks as distributing method.
	Args:
		pop_ageb_gdf (geopandas.GeoDataFrame): GeoDataFrame with AGEB polygons containing pop data.
		pop_mza_gdf (geopandas.GeoDataFrame): GeoDataFrame with block polygons containing pop data.
		extended_logs (bool, optional): Boolean - if true prints statistical logs while processing for each AGEB.

	Returns:
		geopandas.GeoDataFrame: GeoDataFrame with blocks containing pop data with no NaNs.
								(All population columns except for: P_5YMAS, P_5YMAS_F, P_5YMAS_M,
								P_8A14, P_8A14_F, P_8A14_M) Added PCON_DISC.
	"""
	
	##########################################################################################
	# STEP 1: CHECK FOR DIFFERENCES IN AVAILABLE AGEBs (PREVENTS CRASH)
	print("INSPECTING AGEBs.")

	# --------------- 1.1 SET COLUMNS TO .UPPER() EXCEPT FOR GEOMETRY
	# (When the equations were written, we used UPPER names, easier to read and also
	# easier to change it this way and then return output with .lower columns)
	pop_ageb_gdf = pop_ageb_gdf.copy()
	pop_ageb_gdf.columns = pop_ageb_gdf.columns.str.upper()
	pop_ageb_gdf.rename(columns={'GEOMETRY':'geometry'},inplace=True)

	pop_mza_gdf = pop_mza_gdf.copy()
	pop_mza_gdf.columns = pop_mza_gdf.columns.str.upper()
	pop_mza_gdf.rename(columns={'GEOMETRY':'geometry'},inplace=True)

	# --------------- 1.2 CHECK FOR DIFFERENCES IN AGEBs
	# Look for AGEBs in both gdfs
	agebs_in_ageb_gdf = list(pop_ageb_gdf['CVE_AGEB'].unique())
	agebs_in_mza_gdf = list(pop_mza_gdf['CVE_AGEB'].unique())

	if (len(agebs_in_ageb_gdf) == 0) and (len(agebs_in_mza_gdf) == 0):
		print("Error: Area of interest has no pop data.")
		intended_crash

	# Test for AGEBs present in mza_gdf but not in AGEB_gdf (could crash if unchecked)
	missing_agebs = list(set(agebs_in_mza_gdf) - set(agebs_in_ageb_gdf))
	if len(missing_agebs) > 0:
		print(f'WARNING: AGEBs {missing_agebs} present in mza_gdf but missing from ageb_gdf.')
		print(f'WARNING: Removing AGEBs {missing_agebs} from AGEB analysis.')

	##########################################################################################
	# STEP 2: CALCULATE NAN VALUES
	print("STARTING NANs calculation.")

	# LOG CODE - Progress logs
	# Will create progress logs when progress reaches these percentages:
	progress_logs = [10,20,30,40,50,60,70,80,90,100] # for log statistics
	# This df stores accumulative (All AGEBs) statistics for logs.
	acc_statistics = pd.DataFrame() # for log statistics

	# --------------- 2.0 NaNs CALCULATION Start
	i = 1
	for ageb in agebs_in_mza_gdf: # Most of the code of this function iterates over each AGEB

		if extended_logs:
			print('--'*20)
			print(f'Calculating NaNs for AGEB {ageb} ({i}/{len(agebs_in_mza_gdf)}.)')
		
		# LOG CODE - Progress logs
		# Measures current progress, prints if passed a checkpoint of progress_logs list.
		current_progress = (i / len(agebs_in_mza_gdf))*100
		for checkpoint in progress_logs:
			if current_progress >= checkpoint:
				print(f'Calculating NaNs. {checkpoint}% done.')
				progress_logs.remove(checkpoint)
				break

		# --------------- 2.1 FIND CURRENT AGEB BLOCK DATA
		mza_ageb_gdf = pop_mza_gdf.loc[pop_mza_gdf['CVE_AGEB'] == ageb].copy()

		# --------------- 2.2 KEEP OUT OF THE PROCESS ROWS WHICH HAVE 0 VALUES (Blocks where ALL values are NaNs)
		# 2.2a) Set columns to be analysed
		columns_of_interest = ['POBFEM','POBMAS',
							'P_0A2','P_0A2_F','P_0A2_M',
							'P_3A5','P_3A5_F','P_3A5_M',
							'P_6A11','P_6A11_F','P_6A11_M',
							'P_12A14','P_12A14_F','P_12A14_M',
							'P_15A17','P_15A17_F','P_15A17_M',
							'P_18A24','P_18A24_F','P_18A24_M',
							'P_60YMAS','P_60YMAS_F','P_60YMAS_M',
							'P_3YMAS','P_3YMAS_F','P_3YMAS_M',
							'P_12YMAS','P_12YMAS_F','P_12YMAS_M',
							'P_15YMAS','P_15YMAS_F','P_15YMAS_M',
							'P_18YMAS','P_18YMAS_F','P_18YMAS_M',
							'REL_H_M','POB0_14','POB15_64','POB65_MAS',
							'PCON_DISC'] #Added later
		blocks = mza_ageb_gdf[['CVEGEO','POBTOT'] + columns_of_interest].copy()
		
		# 2.2b) Find rows with nan values and sum of nan values
		blocks['found_values'] = 0
		checker_cols = []

		for col in columns_of_interest:
			# Define checker column
			checker_col = f'check_{col}'
			# Turn column to numeric
			blocks[col] = pd.to_numeric(blocks[col])
			# Set checker column to 'exist' (1)
			blocks[checker_col] = 1
			# If it doesn't exist, set that row's check to (0)
			idx = blocks[col].isna()
			blocks.loc[idx, checker_col] = 0
			# Add checker column to checker_cols list
			checker_cols.append(checker_col)
		
		# Sum total number of identified nan values by row
		blocks['found_values'] = blocks[checker_cols].sum(axis=1)
		# Drop checker columns
		blocks.drop(columns=checker_cols, inplace=True)
		
		# 2.2c) Loc rows with values in columns_of_interest (In these rows, NaNs calculation is possible)
		blocks_values = blocks.loc[blocks['found_values'] > 0].copy()
		blocks_values.drop(columns=['found_values'],inplace=True)
		
		# 2.2d) Save rows with 0 values for later. (In these rows, can't calculate NaNs, must distribute values)
		blocks_nans = blocks.loc[blocks['found_values'] == 0].copy()
		blocks_nans.drop(columns=['found_values'],inplace=True)

		del blocks

		# --------------- 2.3 CALCULATE NaN values in blocks
		"""
		This is the first important sub-step towards filling missing values. It works by filling NaNs using known relations.
		e.g. It is known that within a block the Total population (POBTOT) = total fem. population (POBFEM) + total masc. population (POBMAS). 
		Therefore it is possible to stablish the following relations:

			POBTOT = POBFEM + POBMAS
			POBFEM = POBTOT - POBMAS
			POBMAS = POBTOT - POBFEM

		Using this approach and a collection of known relations, NaNs that can be filled are filled.

		This process is repeated (while loop) while the number of NaN values in the dataset is less than the previous number of NaN values.
		(While the loop is still filling NaNs using the known relations)
		"""

		if extended_logs:
			print(f'Calculating NaNs using block data for AGEB {ageb}.')

		# 2.3a) Count current (original) nan values
		original_nan_values = int(blocks_values.isna().sum().sum())
		
		# 2.3b) Set a start and finish nan value for while loop and run
		start_nan_values = original_nan_values
		finish_nan_values = start_nan_values - 1 #To kick start while loop, actual value will be calculated within loop
		loop_count = 1

		while start_nan_values > finish_nan_values:
			# Amount of nans starting while loop round
			start_nan_values = blocks_values.isna().sum().sum()

			# 2.3c) Set of equation with structure [TOTAL] = [FEM.] + [MASC.]
			# POBTOT = POBFEM + POBMAS
			blocks_values.POBTOT.fillna(blocks_values.POBFEM + blocks_values.POBMAS, inplace=True)
			blocks_values.POBFEM.fillna(blocks_values.POBTOT - blocks_values.POBMAS, inplace=True)
			blocks_values.POBMAS.fillna(blocks_values.POBTOT - blocks_values.POBFEM, inplace=True)
			# <><><><><><><><><><>
			# P_0A2 = P_0A2_F + P_0A2_M
			blocks_values.P_0A2.fillna(blocks_values.P_0A2_F + blocks_values.P_0A2_M, inplace=True)
			blocks_values.P_0A2_F.fillna(blocks_values.P_0A2 - blocks_values.P_0A2_M, inplace=True)
			blocks_values.P_0A2_M.fillna(blocks_values.P_0A2 - blocks_values.P_0A2_F, inplace=True)
			# <><><><><><><><><><>
			# P_3A5 = P_3A5_F + P_3A5_M
			blocks_values.P_3A5.fillna(blocks_values.P_3A5_F + blocks_values.P_3A5_M, inplace=True)
			blocks_values.P_3A5_F.fillna(blocks_values.P_3A5 - blocks_values.P_3A5_M, inplace=True)
			blocks_values.P_3A5_M.fillna(blocks_values.P_3A5 - blocks_values.P_3A5_F, inplace=True)
			# <><><><><><><><><><>
			# P_6A11 = P_6A11_F + P_6A11_M
			blocks_values.P_6A11.fillna(blocks_values.P_6A11_F + blocks_values.P_6A11_M, inplace=True)
			blocks_values.P_6A11_F.fillna(blocks_values.P_6A11 - blocks_values.P_6A11_M, inplace=True)
			blocks_values.P_6A11_M.fillna(blocks_values.P_6A11 - blocks_values.P_6A11_F, inplace=True)
			# <><><><><><><><><><>
			# P_12A14 = P_12A14_F + P_12A14_M
			blocks_values.P_12A14.fillna(blocks_values.P_12A14_F + blocks_values.P_12A14_M, inplace=True)
			blocks_values.P_12A14_F.fillna(blocks_values.P_12A14 - blocks_values.P_12A14_M, inplace=True)
			blocks_values.P_12A14_M.fillna(blocks_values.P_12A14 - blocks_values.P_12A14_F, inplace=True)
			# <><><><><><><><><><>
			# P_15A17 = P_15A17_F + P_15A17_M
			blocks_values.P_15A17.fillna(blocks_values.P_15A17_F + blocks_values.P_15A17_M, inplace=True)
			blocks_values.P_15A17_F.fillna(blocks_values.P_15A17 - blocks_values.P_15A17_M, inplace=True)
			blocks_values.P_15A17_M.fillna(blocks_values.P_15A17 - blocks_values.P_15A17_F, inplace=True)
			# <><><><><><><><><><>
			# P_18A24 = P_18A24_F + P_18A24_M
			blocks_values.P_18A24.fillna(blocks_values.P_18A24_F + blocks_values.P_18A24_M, inplace=True)
			blocks_values.P_18A24_F.fillna(blocks_values.P_18A24 - blocks_values.P_18A24_M, inplace=True)
			blocks_values.P_18A24_M.fillna(blocks_values.P_18A24 - blocks_values.P_18A24_F, inplace=True)
			# <><><><><><><><><><>
			# P_60YMAS = P_60YMAS_F + P_60YMAS_M
			blocks_values.P_60YMAS.fillna(blocks_values.P_60YMAS_F + blocks_values.P_60YMAS_M, inplace=True)
			blocks_values.P_60YMAS_F.fillna(blocks_values.P_60YMAS - blocks_values.P_60YMAS_M, inplace=True)
			blocks_values.P_60YMAS_M.fillna(blocks_values.P_60YMAS - blocks_values.P_60YMAS_F, inplace=True)
			# <><><><><><><><><><>
			
			# 2.3d) Set of equation with structure [TOTAL] = (Age_group + Age_group + ... + Age_group) + [Specific age and beyond]
			# POBTOT = (P_0A2) + P_3YMAS
			# --> P_0A2 = POBTOT - P_3YMAS
			blocks_values.P_0A2.fillna(blocks_values.POBTOT - blocks_values.P_3YMAS, inplace=True)
			blocks_values.P_0A2_F.fillna(blocks_values.POBFEM - blocks_values.P_3YMAS_F, inplace=True)
			blocks_values.P_0A2_M.fillna(blocks_values.POBMAS - blocks_values.P_3YMAS_M, inplace=True)
			# --> P_3YMAS = POBTOT - P_0A2
			blocks_values.P_3YMAS.fillna(blocks_values.POBTOT - blocks_values.P_0A2, inplace=True)
			blocks_values.P_3YMAS_F.fillna(blocks_values.POBFEM - blocks_values.P_0A2_F, inplace=True)
			blocks_values.P_3YMAS_M.fillna(blocks_values.POBMAS - blocks_values.P_0A2_M, inplace=True)
			# <><><><><><><><><><>
			# POBTOT = (P_0A2 + P_3A5 + P_6A11) + P_12YMAS
			# --> P_0A2 = POBTOT - P_12YMAS - P_3A5 - P_6A11
			blocks_values.P_0A2.fillna(blocks_values.POBTOT - blocks_values.P_12YMAS - blocks_values.P_3A5 - blocks_values.P_6A11, inplace=True)
			blocks_values.P_0A2_F.fillna(blocks_values.POBFEM - blocks_values.P_12YMAS_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F, inplace=True)
			blocks_values.P_0A2_M.fillna(blocks_values.POBMAS - blocks_values.P_12YMAS_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M, inplace=True)
			# --> P_3A5 = POBTOT - P_12YMAS - P_0A2 - P_6A11
			blocks_values.P_3A5.fillna(blocks_values.POBTOT - blocks_values.P_12YMAS - blocks_values.P_0A2 - blocks_values.P_6A11, inplace=True)
			blocks_values.P_3A5_F.fillna(blocks_values.POBFEM - blocks_values.P_12YMAS_F - blocks_values.P_0A2_F - blocks_values.P_6A11_F, inplace=True)
			blocks_values.P_3A5_M.fillna(blocks_values.POBMAS - blocks_values.P_12YMAS_M - blocks_values.P_0A2_M - blocks_values.P_6A11_M, inplace=True)
			# --> P_6A11 = POBTOT - P_12YMAS - P_0A2 - P_3A5
			blocks_values.P_6A11.fillna(blocks_values.POBTOT - blocks_values.P_12YMAS - blocks_values.P_0A2 - blocks_values.P_3A5, inplace=True)
			blocks_values.P_6A11_F.fillna(blocks_values.POBFEM - blocks_values.P_12YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F, inplace=True)
			blocks_values.P_6A11_M.fillna(blocks_values.POBMAS - blocks_values.P_12YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M, inplace=True)
			# --> P_12YMAS = POBTOT - P_0A2 - P_3A5 -P_6A11
			blocks_values.P_12YMAS.fillna(blocks_values.POBTOT - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11, inplace=True)
			blocks_values.P_12YMAS_F.fillna(blocks_values.POBFEM - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F, inplace=True)
			blocks_values.P_12YMAS_M.fillna(blocks_values.POBMAS - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M, inplace=True)
			# <><><><><><><><><><>
			# POBTOT = (P_0A2 + P_3A5 + P_6A11 + P_12A14) + P_15YMAS
			# --> P_0A2 = POBTOT - P_15YMAS - P_3A5 - P_6A11 - P_12A14
			blocks_values.P_0A2.fillna(blocks_values.POBTOT - blocks_values.P_15YMAS - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			blocks_values.P_0A2_F.fillna(blocks_values.POBFEM - blocks_values.P_15YMAS_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F, inplace=True)
			blocks_values.P_0A2_M.fillna(blocks_values.POBMAS - blocks_values.P_15YMAS_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M, inplace=True)
			# --> P_3A5 = POBTOT - P_15YMAS - P_0A2 - P_6A11 - P_12A14
			blocks_values.P_3A5.fillna(blocks_values.POBTOT - blocks_values.P_15YMAS - blocks_values.P_0A2 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			blocks_values.P_3A5_F.fillna(blocks_values.POBFEM - blocks_values.P_15YMAS_F - blocks_values.P_0A2_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F, inplace=True)
			blocks_values.P_3A5_M.fillna(blocks_values.POBMAS - blocks_values.P_15YMAS_M - blocks_values.P_0A2_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M, inplace=True)
			# --> P_6A11 = POBTOT - P_15YMAS - P_0A2 - P_3A5 - P_12A14
			blocks_values.P_6A11.fillna(blocks_values.POBTOT - blocks_values.P_15YMAS - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_12A14, inplace=True)
			blocks_values.P_6A11_F.fillna(blocks_values.POBFEM - blocks_values.P_15YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_12A14_F, inplace=True)
			blocks_values.P_6A11_M.fillna(blocks_values.POBMAS - blocks_values.P_15YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_12A14_M, inplace=True)
			# --> P_12A14 = POBTOT - P_15YMAS - P_0A2 - P_3A5 - P_6A11
			blocks_values.P_12A14.fillna(blocks_values.POBTOT - blocks_values.P_15YMAS - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11, inplace=True)
			blocks_values.P_12A14_F.fillna(blocks_values.POBFEM - blocks_values.P_15YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F, inplace=True)
			blocks_values.P_12A14_M.fillna(blocks_values.POBMAS - blocks_values.P_15YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M, inplace=True)
			# --> P_15YMAS = POBTOT - P_0A2 - P_3A5 - P_6A11 - P_12A14
			blocks_values.P_15YMAS.fillna(blocks_values.POBTOT - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			blocks_values.P_15YMAS_F.fillna(blocks_values.POBFEM - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F, inplace=True)
			blocks_values.P_15YMAS_M.fillna(blocks_values.POBMAS - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M, inplace=True)
			# <><><><><><><><><><>
			# POBTOT = (P_0A2 + P_3A5 + P_6A11 + P_12A14 + P_15A17) + P_18YMAS
			# --> P_0A2 = POBTOT - P_18YMAS - P_3A5 - P_6A11 - P_12A14 - P_15A17
			blocks_values.P_0A2.fillna(blocks_values.POBTOT - blocks_values.P_18YMAS - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14 - blocks_values.P_15A17, inplace=True)
			blocks_values.P_0A2_F.fillna(blocks_values.POBFEM - blocks_values.P_18YMAS_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F - blocks_values.P_15A17_F, inplace=True)
			blocks_values.P_0A2_M.fillna(blocks_values.POBMAS - blocks_values.P_18YMAS_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M - blocks_values.P_15A17_M, inplace=True)
			# --> P_3A5 = POBTOT - P_18YMAS - P_0A2 - P_6A11 - P_12A14 - P_15A17
			blocks_values.P_3A5.fillna(blocks_values.POBTOT - blocks_values.P_18YMAS - blocks_values.P_0A2 - blocks_values.P_6A11 - blocks_values.P_12A14 - blocks_values.P_15A17, inplace=True)
			blocks_values.P_3A5_F.fillna(blocks_values.POBFEM - blocks_values.P_18YMAS_F - blocks_values.P_0A2_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F - blocks_values.P_15A17_F, inplace=True)
			blocks_values.P_3A5_M.fillna(blocks_values.POBMAS - blocks_values.P_18YMAS_M - blocks_values.P_0A2_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M - blocks_values.P_15A17_M, inplace=True)
			# --> P_6A11 = POBTOT - P_18YMAS - P_0A2 - P_3A5 - P_12A14 - P_15A17
			blocks_values.P_6A11.fillna(blocks_values.POBTOT - blocks_values.P_18YMAS - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_12A14 - blocks_values.P_15A17, inplace=True)
			blocks_values.P_6A11_F.fillna(blocks_values.POBFEM - blocks_values.P_18YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_12A14_F - blocks_values.P_15A17_F, inplace=True)
			blocks_values.P_6A11_M.fillna(blocks_values.POBMAS - blocks_values.P_18YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_12A14_M - blocks_values.P_15A17_M, inplace=True)
			# --> P_12A14 = POBTOT - P_18YMAS - P_0A2 - P_3A5 - P_6A11 - P_15A17
			blocks_values.P_12A14.fillna(blocks_values.POBTOT - blocks_values.P_18YMAS - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_15A17, inplace=True)
			blocks_values.P_12A14_F.fillna(blocks_values.POBFEM - blocks_values.P_18YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_15A17_F, inplace=True)
			blocks_values.P_12A14_M.fillna(blocks_values.POBMAS - blocks_values.P_18YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_15A17_M, inplace=True)
			# --> P_15A17 = POBTOT - P_18YMAS - P_0A2 - P_3A5 - P_6A11 - P_12A14
			blocks_values.P_15A17.fillna(blocks_values.POBTOT - blocks_values.P_18YMAS - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			blocks_values.P_15A17_F.fillna(blocks_values.POBFEM - blocks_values.P_18YMAS_F - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F, inplace=True)
			blocks_values.P_15A17_M.fillna(blocks_values.POBMAS - blocks_values.P_18YMAS_M - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M, inplace=True)
			# --> P_18YMAS = POBTOT - P_0A2 - P_3A5 - P_6A11 - P_12A14 - P_15A17
			blocks_values.P_18YMAS.fillna(blocks_values.POBTOT - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14 - blocks_values.P_15A17, inplace=True)
			blocks_values.P_18YMAS_F.fillna(blocks_values.POBFEM - blocks_values.P_0A2_F - blocks_values.P_3A5_F - blocks_values.P_6A11_F - blocks_values.P_12A14_F - blocks_values.P_15A17_F, inplace=True)
			blocks_values.P_18YMAS_M.fillna(blocks_values.POBMAS - blocks_values.P_0A2_M - blocks_values.P_3A5_M - blocks_values.P_6A11_M - blocks_values.P_12A14_M - blocks_values.P_15A17_M, inplace=True)
			# <><><><><><><><><><>

			# 2.3e) Set of complementary equations
			# REL_H_M = (POBMAS/POBFEM)*100
			# --> POBMAS = (REL_H_M/100) * POBFEM
			blocks_values.POBMAS.fillna(round((blocks_values.REL_H_M / 100) * blocks_values.POBFEM,0), inplace=True)
			# --> POBFEM = (POBMAS * 100) / REL_H_M
			blocks_values.POBFEM.fillna(round((blocks_values.POBMAS * 100) / blocks_values.REL_H_M,0), inplace=True)
			# <><><><><><><><><><>
			# POBTOT = POB0_14 + POB15_64 + POB65_MAS
			# --> POB0_14 = POBTOT - POB15_64 - POB65_MAS
			blocks_values.POB0_14.fillna(blocks_values.POBTOT - blocks_values.POB15_64 - blocks_values.POB65_MAS, inplace=True)
			# --> POB15_64 = POBTOT - POB0_14 - POB65_MAS
			blocks_values.POB15_64.fillna(blocks_values.POBTOT - blocks_values.POB0_14 - blocks_values.POB65_MAS, inplace=True)
			# --> POB65_MAS = POBTOT - POB0_14 - POB15_64
			blocks_values.POB65_MAS.fillna(blocks_values.POBTOT - blocks_values.POB0_14 - blocks_values.POB15_64, inplace=True)
			# <><><><><><><><><><>
			# POB0_14 = P_0A2 + P_3A5 + P_6A11 + P_12A14
			# --> POB0_14 = P_0A2 + P_3A5 + P_6A11 + P_12A14
			blocks_values.POB0_14.fillna(blocks_values.P_0A2 + blocks_values.P_3A5 + blocks_values.P_6A11 + blocks_values.P_12A14, inplace=True)
			# --> P_0A2 = POB0_14 - P_3A5 - P_6A11 - P_12A14
			blocks_values.P_0A2.fillna(blocks_values.POB0_14 - blocks_values.P_3A5 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			# --> P_3A5 = POB0_14 - P_0A2 - P_6A11 - P_12A14
			blocks_values.P_3A5.fillna(blocks_values.POB0_14 - blocks_values.P_0A2 - blocks_values.P_6A11 - blocks_values.P_12A14, inplace=True)
			# --> P_6A11 = POB0_14 - P_0A2 - P_3A5 - P_12A14
			blocks_values.P_6A11.fillna(blocks_values.POB0_14 - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_12A14, inplace=True)				
			# --> P_12A14 = POB0_14 - P_0A2 - P_3A5 - P_6A11
			blocks_values.P_12A14.fillna(blocks_values.POB0_14 - blocks_values.P_0A2 - blocks_values.P_3A5 - blocks_values.P_6A11, inplace=True)
			# <><><><><><><><><><>
			# POBTOT = POB0_14 + P_15YMAS
			# --> P_15YMAS = POBTOT - POB0_14
			blocks_values.P_15YMAS.fillna(blocks_values.POBTOT - blocks_values.POB0_14, inplace=True)
			# --> POB0_14 = POBTOT - P_15YMAS
			blocks_values.POB0_14.fillna(blocks_values.POBTOT - blocks_values.P_15YMAS, inplace=True)
			# <><><><><><><><><><>
			
			# Amount of nans finishing while loop round
			finish_nan_values = blocks_values.isna().sum().sum()

			if extended_logs:
				print(f'Round {loop_count} Starting with {start_nan_values} nan values. Finishing with {finish_nan_values} nan values.')
			
			loop_count += 1
		
		# LOG CODE - Statistics
		nan_reduction = round(((1-(finish_nan_values/original_nan_values))*100),2) # for log statistics
		if extended_logs:
			print(f'Originally had {original_nan_values} nan values, now there are {finish_nan_values}. A {nan_reduction}% reduction.')
		
		# 2.3f) Join back blocks with values with blocks without values
		blocks_calc = pd.concat([blocks_values,blocks_nans])
		
		# --------------- 2.4 CALCULATE NaN values using AGEBs.
  		# For the nan values that couldn't be solved, distributes AGEB data.
		"""
		This is the second important sub-step towards filling missing values. It's meant to be a temporary solution while a more accurate solution is built.
		
		It fills the remaining NaNs by distributing the known AGEB data into each of the blocks inside each AGEB, using POBTOT as an indicator 
		to the amount of each age group to be distributed. Blocks which originaly had no data (NaN values only), remain empty.
		"""
		
		# 2.4a) Prepare for second loop
		# Remove masc/fem relation from analysis as it complicates this and further processes
    	# If and when needed, calculate using (REL_H_M = (POBMAS/POBFEM)*100)
		ageb_filling_cols = columns_of_interest.copy()
		ageb_filling_cols.remove('REL_H_M')
		blocks_calc.drop(columns=['REL_H_M'],inplace=True)

		# If not in crash check from STEP 1:
		if ageb not in missing_agebs:

			if extended_logs:
				print(f'Calculating NaNs using AGEB data for AGEB {ageb}.')

			# Locate AGEB data in pop_ageb_gdf (Total known data that should be found if adding all blocks data)
			ageb_gdf = pop_ageb_gdf.loc[pop_ageb_gdf['CVE_AGEB'] == ageb]

			# LOG CODE - Statistics
			# This log registers the solving method that was used to solve columns
			solved_using_blocks = 0 # for log statistics
			solved_using_ageb = 0 # for log statistics
			
			# 2.4b) Fill with AGEB values.
			for col in ageb_filling_cols:
				# Find number of nan values in current col
				col_nan_values = blocks_calc.isna().sum()[col]

				# If there are no nan values left in col, does nothing.
				if col_nan_values == 0:
					solved_using_blocks += 1 # for log statistics
				
				# Elif there is only one value left, assign missing value directly to cell.
				elif col_nan_values == 1: 
					# Calculate missing value
					ageb_col_value = ageb_gdf[col].unique()[0]
					current_block_sum = blocks_calc[col].sum()
					missing_value = ageb_col_value - current_block_sum
					# Add missing value to na cell in column
					blocks_calc[col].fillna(missing_value,inplace=True)
					solved_using_ageb += 1 # for log statistics
				
				# Elif there are more than one nan in col, distribute using POBTOT of those blocks as distributing indicator.
				elif col_nan_values > 1:        
					# Locate rows with NaNs in current col 
					# This ensures POBTOT as a distributing indicator works for the missing data rows only
					idx = blocks_calc[col].isna()
					# Set distributing factor to 0
					blocks_calc['dist_factor'] = 0
					# Assign to those rows a distributing factor ==> (POBTOT of each row / sum of POBTOT of those rows)
					blocks_calc.loc[idx,'dist_factor'] = (blocks_calc['POBTOT']) / blocks_calc.loc[idx]['POBTOT'].sum()
					# Calculate missing value
					ageb_col_value = ageb_gdf[col].unique()[0]
					current_block_sum = blocks_calc[col].sum()
					missing_value = ageb_col_value - current_block_sum
					# Distribute missing value in those rows using POBTOT factor
					blocks_calc[col].fillna(missing_value * blocks_calc['dist_factor'], inplace=True)
					blocks_calc.drop(columns=['dist_factor'],inplace=True)
					solved_using_ageb += 1 # for log statistics

			# LOG CODE - Statistics - How was this AGEB solved?
			if extended_logs:
				pct_col_byblocks = (solved_using_blocks / len(ageb_filling_cols))*100 # for log statistics
				pct_col_byagebs = (solved_using_ageb / len(ageb_filling_cols))*100 # for log statistics
				print(f'{pct_col_byblocks}% of columns solved using block data only.')
				print(f'{pct_col_byagebs}% of columns required AGEB filling.')
		
			# Logs Statistics - Add currently examined AGEB statistics to log df
			acc_statistics.loc[i,'ageb'] = ageb # for log statistics
			# Percentage of NaNs found using blocks gdf
			acc_statistics.loc[i,'nans_calculated'] = nan_reduction # for log statistics
			# Columns which could be solved entirely using equations in block_gdf
			acc_statistics.loc[i,'block_calculated'] = solved_using_blocks # for log statistics
			# Columns which required AGEB filling
			acc_statistics.loc[i,'ageb_filling'] = solved_using_ageb # for log statistics
			# All could be solved, so
			acc_statistics.loc[i,'unable_to_solve'] = 0 # for log statistics

		else: #current AGEB is in missing_agebs list (Present in mza_gdf, but not in ageb_gdf) 
			# EVERYTHING here is for log statistics
			if extended_logs:
				print(f"NANs on AGEB {ageb} cannot be calculated using AGEB data because it doesn't exist.")

			# Solving method used to solve column
			solved_using_blocks = 0
			unable_tosolve = 0
			
			# LOG CODE - Statistics - Register how columns where solved.
			for col in ageb_filling_cols:
				# Find number of nan values in current col
				col_nan_values = blocks_calc.isna().sum()[col]
				# If there are no nan values left in col, pass.
				if col_nan_values == 0:
					solved_using_blocks += 1
				else:
					unable_tosolve += 1

			# Logs Statistics - How was this AGEB solved?
			if extended_logs:
				pct_col_byblocks = (solved_using_blocks / len(ageb_filling_cols))*100
				pct_col_notsolved = (unable_tosolve / len(ageb_filling_cols))*100
				print(f"{pct_col_byblocks}% of columns solved using block data only.")
				print(f"{pct_col_notsolved}% of columns couldn't be solved.")

			# Logs Statistics - Add currently examined AGEB statistics to log df
			acc_statistics.loc[i,'ageb'] = ageb
			# Percentage of NaNs found using blocks gdf
			acc_statistics.loc[i,'nans_calculated'] = nan_reduction
			# Columns which could be solved entirely using equations in block_gdf
			acc_statistics.loc[i,'block_calculated'] = solved_using_blocks
			# There wasn't AGEB filling, therefore:
			acc_statistics.loc[i,'ageb_filling'] = 0
			# Columns which couldn't be solved because there was no AGEB filling
			acc_statistics.loc[i,'unable_to_solve'] = unable_tosolve

		# --------------- 2.5 Return calculated data to original block gdf for the current AGEB (mza_ageb_gdf)
		# 2.5a) Change original cols for calculated cols
		calculated_cols = ['POBTOT'] + ageb_filling_cols
		mza_ageb_gdf = mza_ageb_gdf.drop(columns=calculated_cols) #Drops current block pop cols
		mza_ageb_gdf = pd.merge(mza_ageb_gdf, blocks_calc, on='CVEGEO') #Replaces with calculated (blocks_calc) cols

		# 2.5b) Restore column order from original (input) block gdf
		column_order = list(pop_mza_gdf.columns.values)
		mza_ageb_gdf = mza_ageb_gdf[column_order]

		# 2.5c) Save to mza_calc gdf (Function's output)
		if i == 1:
			mza_calc = mza_ageb_gdf.copy()
		else:
			mza_calc = pd.concat([mza_calc,mza_ageb_gdf])

		i += 1

	# --------------- 2.6 Final format and release of log statistics

	# Format final output
	mza_calc.reset_index(inplace=True)
	mza_calc.drop(columns=['index'],inplace=True)
	# Deliver output cols as .lower()
	mza_calc.columns = mza_calc.columns.str.lower()

	# Release final log statistics.
	nans_calculated = round(acc_statistics['nans_calculated'].mean(),2) # for log statistics
	# Columns solving method
	block_calculated = acc_statistics['block_calculated'].sum() # for log statistics
	ageb_filling = acc_statistics['ageb_filling'].sum() # for log statistics
	unable_to_solve = acc_statistics['unable_to_solve'].sum() # for log statistics
	# % columns solving method
	total_cols = block_calculated + ageb_filling + unable_to_solve # for log statistics
	pct_block_calculated = round((block_calculated/total_cols)*100,2) # for log statistics
	pct_ageb_filling = round((ageb_filling/total_cols)*100,2) # for log statistics
	pct_unable_to_solve = round((unable_to_solve/total_cols)*100,2) # for log statistics

	print("Finished calculating NaNs.")
	print(f"Percentage of NaNs found using blocks gdf: {nans_calculated}%.")
	print(f"Columns which could be solved entirely using relations between age groups in block_gdf: {block_calculated} ({pct_block_calculated}%).")
	print(f"Columns which required AGEB filling: {ageb_filling} ({pct_ageb_filling}%).")
	print(f"Columns which couldn't be solved: {unable_to_solve} ({pct_unable_to_solve}%).")
	
	return mza_calc

def voronoi_points_within_aoi(area_of_interest, points, points_id_col, admissible_error=0.01, projected_crs="EPSG:6372"):
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


def proximity_isochrone(G, nodes, edges, point_of_interest, trip_time, prox_measure="length", projected_crs="EPSG:6372"):
	
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

def proximity_isochrone_from_osmid(G, nodes, edges, center_osmid, trip_time, prox_measure="length", projected_crs="EPSG:6372"):
	
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
				 preprocessed_nearest=(False,'dir')):
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