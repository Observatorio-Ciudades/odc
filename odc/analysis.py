import numpy as np
import geopandas as gpd
import pandas as pd
import h3
import shapely
from scipy.spatial import Voronoi

import math
from scipy import optimize
from math import sqrt

from .utils import *
from .data import *


def group_points_by_bins(
    points: gpd.GeoDataFrame,
    bins: gpd.GeoDataFrame,
    bin_id_column: str,
    aggregate_columns: Union[str, List[str]],
    aggregation_func: Union[str, Dict[str, Union[str, Callable]]] = 'mean',
    fill_missing: bool = True,
    fill_value: Union[float, Dict[str, float]] = 0.0,
    zero_replacement: Optional[Union[float, Dict[str, float]]] = None,
    drop_columns: Optional[List[str]] = None,
    spatial_predicate: str = 'within'
) -> gpd.GeoDataFrame:
    """
    Aggregate point data within spatial bins using specified aggregation functions.

    Performs spatial join between points and bins, then aggregates point attributes
    using configurable aggregation functions. Supports multiple aggregation methods,
    custom fill values, and flexible handling of missing data and edge cases.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries and attributes to aggregate.
        Must contain valid geometry column with point features.
    bins : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries for spatial binning.
        Must contain valid geometry column with polygon features.
    bin_id_column : str
        Column name in bins DataFrame containing unique bin identifiers.
        Must be present in bins DataFrame and contain unique values.
    aggregate_columns : str or list of str
        Column name(s) in points DataFrame to aggregate within each bin.
        All specified columns must exist in the points DataFrame.
    aggregation_func : str or dict, default 'mean'
        Aggregation function(s) to apply. Can be:
        - String: single function for all columns ('mean', 'sum', 'count', etc.)
        - Dict: mapping column names to specific functions
        Common functions: 'mean', 'sum', 'count', 'min', 'max', 'std', 'median'
    fill_missing : bool, default True
        Whether to fill bins with no intersecting points using fill_value.
        If False, bins without points will have NaN values.
    fill_value : float or dict, default 0.0
        Value(s) to use for bins with no intersecting points when fill_missing=True.
        Can be single value or dict mapping column names to specific fill values.
    zero_replacement : float or dict, optional
        Value(s) to replace exact zeros in aggregated results.
        Useful for distance calculations where zero may be problematic.
        If None, no zero replacement is performed.
    drop_columns : list of str, optional
        Additional column names to drop from final results.
        Remove unwanted columns from spatial join.
    spatial_predicate : str, default 'within'
        Spatial relationship predicate for point-bin intersection.
        Options: 'within', 'intersects', 'contains', 'crosses', 'touches'

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with bin geometries and aggregated values for specified columns.
        Contains bin_id_column, geometry, and aggregated columns from aggregate_columns.
        Bins without intersecting points handled according to fill_missing parameter.
    """

    # Input validation
    if not isinstance(points, gpd.GeoDataFrame):
        raise TypeError("points must be a GeoDataFrame")

    if not isinstance(bins, gpd.GeoDataFrame):
        raise TypeError("bins must be a GeoDataFrame")

    if bin_id_column not in bins.columns:
        raise ValueError(f"Column '{bin_id_column}' not found in bins DataFrame")

    # Ensure aggregate_columns is a list
    if isinstance(aggregate_columns, str):
        aggregate_columns = [aggregate_columns]

    # Check that all aggregate columns exist in points
    missing_cols = [col for col in aggregate_columns if col not in points.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in points DataFrame")

    # Validate spatial predicate
    valid_predicates = ['within', 'intersects', 'contains', 'crosses', 'touches']
    if spatial_predicate not in valid_predicates:
        raise ValueError(f"spatial_predicate must be one of {valid_predicates}")

    # Create working copy
    points_copy = points.copy()

    # Perform spatial join
    try:
        points_in_bins = gpd.sjoin(
            points_copy,
            bins,
            how='inner',
            predicate=spatial_predicate
        )
    except Exception as e:
        raise RuntimeError(f"Spatial join failed: {str(e)}")

    # Drop geometry column to avoid aggregation issues
    points_in_bins = points_in_bins.drop(columns=['geometry'])

    # Prepare aggregation functions
    if isinstance(aggregation_func, str):
        agg_dict = {col: aggregation_func for col in aggregate_columns}
    elif isinstance(aggregation_func, dict):
        # Validate that all aggregate columns have specified functions
        missing_agg = [col for col in aggregate_columns if col not in aggregation_func]
        if missing_agg:
            raise ValueError(f"Aggregation functions not specified for columns: {missing_agg}")
        agg_dict = {col: aggregation_func[col] for col in aggregate_columns}
    else:
        raise TypeError("aggregation_func must be string or dictionary")

    # Perform aggregation
    try:
        aggregated = points_in_bins.groupby(bin_id_column)[aggregate_columns].agg(agg_dict)
        aggregated = aggregated.reset_index()

    except Exception as e:
        raise RuntimeError(f"Aggregation failed: {str(e)}")

    # Merge back with bin geometries
    result = pd.merge(
        bins,
        aggregated,
        left_on=bin_id_column,
        right_on=bin_id_column,
        how='left'
    )

    # Drop unwanted columns
    columns_to_drop = ['index_right']  # Standard spatial join artifact
    if drop_columns:
        columns_to_drop.extend(drop_columns)

    # Only drop columns that actually exist
    columns_to_drop = [col for col in columns_to_drop if col in result.columns]
    if columns_to_drop:
        result = result.drop(columns=columns_to_drop)

    # Handle zero replacement
    if zero_replacement is not None:
        if isinstance(zero_replacement, (int, float)):
            # Apply same replacement to all aggregate columns
            for col in aggregate_columns:
                if col in result.columns:
                    result[col] = result[col].replace(0, zero_replacement)
        elif isinstance(zero_replacement, dict):
            # Apply column-specific replacements
            for col, replacement in zero_replacement.items():
                if col in result.columns:
                    result[col] = result[col].replace(0, replacement)
        else:
            raise TypeError("zero_replacement must be numeric or dictionary")

    # Handle missing values
    if fill_missing:
        if isinstance(fill_value, (int, float)):
            # Apply same fill value to all aggregate columns
            fill_dict = {col: fill_value for col in aggregate_columns if col in result.columns}
        elif isinstance(fill_value, dict):
            # Apply column-specific fill values
            fill_dict = {col: val for col, val in fill_value.items() if col in result.columns}
        else:
            raise TypeError("fill_value must be numeric or dictionary")

        result = result.fillna(fill_dict)

    # Ensure result is a GeoDataFrame
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry=result['geometry'])

    return result


def fill_hex(
    missing_hex: gpd.GeoDataFrame,
    data_hex: gpd.GeoDataFrame,
    resolution: int,
    data_column: str
) -> gpd.GeoDataFrame:
    """
    Fill missing hexagon data using iterative neighborhood averaging.

    Fills hexagons lacking data by calculating weighted averages from neighboring
    hexagons using H3 k-ring operations. Iteratively processes hexagons until
    all missing values are filled, expanding search radius as needed.

    Parameters
    ----------
    missing_hex : geopandas.GeoDataFrame
        GeoDataFrame containing hexagons without data values.
        Must contain hex_id column and geometry column.
    data_hex : geopandas.GeoDataFrame
        GeoDataFrame containing hexagons with existing data values.
        Must contain hex_id column, geometry column, and data_column.
    resolution : int
        H3 hexagon resolution level for k-ring neighbor calculations.
        Used to construct hex_id column name as f'hex_id_{resolution}'.
    data_column : str
        Name of column containing data values to interpolate.
        Must be present in data_hex DataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with complete hexagon coverage and filled data values.
        Contains hex_id column, geometry column, and interpolated data_column.
        All hexagons will have non-null values in the data column.
    """

    # create a complete GeoDataFrame with missing values
    missing_hex[[f'{data_column}']] = np.nan
    all_hexagons = data_hex.append(missing_hex)
    all_hexagons = all_hexagons.set_index(f'hex_id_{resolution}')
    ## Start looping to fill missing values
    count = 0
    iter = 1
    # create column that counts the number of processes passed
    all_hexagons[f'{data_column}'+ str(count)] = all_hexagons[f'{data_column}'].copy()
    # iterate while there is missing data
    while all_hexagons[f'{data_column}'+str(count)].isna().sum() > 0:
        # filter hexagons with missing data
        missing = all_hexagons[all_hexagons[f'{data_column}'+str(count)].isna()]

        all_hexagons[f'{data_column}'+ str(iter)] = all_hexagons[f'{data_column}'+str(count)].copy()
        for idx,row in missing.iterrows():
    		# gather surrounding hexagons ids
            near = pd.DataFrame(h3.k_ring(idx,1))
            near[f'hex_id_{resolution}'] = h3.k_ring(idx,1)
            near['a'] = np.nan
            near= near.set_index(f'hex_id_{resolution}')
            # filter hexagons geometries for neighbors according to ids
            neighbors = near.merge(all_hexagons, left_index=True, right_index=True, how='left')
            # calculate average for neighbors
            average = neighbors[f'{data_column}'+str(count)].mean()
            all_hexagons.at[idx, f'{data_column}'+str(iter)] = average
        count = count + 1
        iter = iter + 1
    # return filled hexagons with hex_id_{resolution} as index and data_column and geometry columns
    full_hex = all_hexagons[['geometry']]
    full_hex[f'{data_column}'] = all_hexagons[f'{data_column}'+ str(count)].copy()

    return full_hex


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
	d0 = np.subtract.outer(obs[:,0], interp[:,0])
	d1 = np.subtract.outer(obs[:,1], interp[:,1])

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
			log(f'Error = {round(area_diff,2)}%. Repeating process.')
			distance = distance * 10
		else:
			log(f'Error = {round(area_diff,2)}%. Admissible.')

	# Out of the while loop:
	return voronois_gdf
