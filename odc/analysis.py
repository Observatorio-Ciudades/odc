################################################################################
# Module: Analysis
# Set of spatial data treatment and analysis functions
# updated: 21/10/2025
################################################################################


import numpy as np
import geopandas as gpd
import pandas as pd
import h3
import shapely
from scipy.spatial import Voronoi

import math
from scipy import optimize, special

from typing import Union, List, Dict, Optional, Callable, Tuple

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

def fill_missing_h3_data(
    missing_data: gpd.GeoDataFrame,
    data_with_values: gpd.GeoDataFrame,
    id_column: str,
    value_columns: Union[str, list],
    max_iterations: int = 100,
    aggregation_method: str = 'mean',
    fill_isolated: bool = True,
    isolated_fill_value: Optional[Union[float, dict]] = None
) -> gpd.GeoDataFrame:
    """
    Fill missing spatial h3 data using iterative neighborhood processing.

    Iteratively fills missing values by calculating aggregated values from spatial
    neighbors until convergence or maximum iterations reached.

    Parameters
    ----------
    missing_data : geopandas.GeoDataFrame
        GeoDataFrame containing spatial units without data values.
        Must contain id_column and geometry.
    data_with_values : geopandas.GeoDataFrame
        GeoDataFrame containing spatial units with existing data values.
        Must contain id_column, geometry, and value_columns.
    id_column : str
        Column name containing unique spatial unit identifiers.
        Must be present in both input DataFrames.
    value_columns : str or list of str
        Column name(s) containing data values to interpolate.
        Must be present in data_with_values DataFrame.
    max_iterations : int, default 100
        Maximum number of iterations before stopping interpolation.
    aggregation_method : str, default 'mean'
        Method for aggregating neighbor values ('mean', 'median', 'min', 'max').
    fill_isolated : bool, default True
        Whether to fill isolated spatial units with no data neighbors.
        If False, isolated units remain as NaN.
    isolated_fill_value : float or dict, optional
        Value(s) to use for isolated spatial units when fill_isolated=True.
        If None, uses global mean of available data.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with complete spatial coverage and interpolated values.
        Contains id_column, geometry, and interpolated value_columns.

    Raises
    ------
    ValueError
        If required columns are missing or if inputs are invalid.
    RuntimeError
        If interpolation fails to converge within max_iterations.
    """

    # Input validation
    if not isinstance(missing_data, gpd.GeoDataFrame):
        raise TypeError("missing_data must be a GeoDataFrame")

    if not isinstance(data_with_values, gpd.GeoDataFrame):
        raise TypeError("data_with_values must be a GeoDataFrame")

    if id_column not in missing_data.columns:
        raise ValueError(f"Column '{id_column}' not found in missing_data")

    if id_column not in data_with_values.columns:
        raise ValueError(f"Column '{id_column}' not found in data_with_values")

    # Ensure value_columns is a list
    if isinstance(value_columns, str):
        value_columns = [value_columns]

    # Check value columns exist
    missing_cols = [col for col in value_columns if col not in data_with_values.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in data_with_values")

    # Prepare combined dataset
    missing_copy = missing_data.copy()
    for col in value_columns:
        missing_copy[col] = np.nan

    # Combine datasets
    all_spatial_units = pd.concat([data_with_values, missing_copy], ignore_index=True)
    all_spatial_units = all_spatial_units.set_index(id_column)

    # Initialize tracking variables
    iteration = 0
    previous_missing_count = np.inf

    # Calculate global means for isolated units if needed
    if fill_isolated and isolated_fill_value is None:
        global_means = {col: data_with_values[col].mean() for col in value_columns}
        isolated_fill_value = global_means

    # Prepare isolated fill values
    if isolated_fill_value is not None:
        if isinstance(isolated_fill_value, (int, float)):
            isolated_values = {col: isolated_fill_value for col in value_columns}
        elif isinstance(isolated_fill_value, dict):
            isolated_values = isolated_fill_value
        else:
            raise TypeError("isolated_fill_value must be numeric or dictionary")

    # Main interpolation loop
    while iteration < max_iterations:
        # Count current missing values
        current_missing = sum(all_spatial_units[col].isna().sum() for col in value_columns)

        if current_missing == 0:
            break

        # Track units that get filled this iteration
        units_filled_this_iteration = set()

        # Process each spatial unit with missing data
        for spatial_id in all_spatial_units.index:
            unit_data = all_spatial_units.loc[spatial_id]

            # Check if any value columns are missing for this unit
            if any(pd.isna(unit_data[col]) for col in value_columns):

                try:
                    # Get neighbors
                    neighbor_ids = h3.grid_ring(spatial_id, 1)

                    # Filter to existing neighbors
                    existing_neighbors = [nid for nid in neighbor_ids
                                        if nid in all_spatial_units.index]

                    if existing_neighbors:
                        neighbor_data = all_spatial_units.loc[existing_neighbors]

                        # Fill missing columns that have neighbor data
                        for col in value_columns:
                            if pd.isna(unit_data[col]):
                                neighbor_values = neighbor_data[col].dropna()

                                if len(neighbor_values) > 0:
                                    # Calculate aggregated value
                                    if aggregation_method == 'mean':
                                        new_value = neighbor_values.mean()
                                    elif aggregation_method == 'median':
                                        new_value = neighbor_values.median()
                                    elif aggregation_method == 'min':
                                        new_value = neighbor_values.min()
                                    elif aggregation_method == 'max':
                                        new_value = neighbor_values.max()
                                    else:
                                        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

                                    all_spatial_units.at[spatial_id, col] = new_value
                                    units_filled_this_iteration.add(spatial_id)

                    else:
                        # Handle isolated units
                        if fill_isolated and isolated_fill_value is not None:
                            for col in value_columns:
                                if pd.isna(unit_data[col]):
                                    all_spatial_units.at[spatial_id, col] = isolated_values[col]
                                    units_filled_this_iteration.add(spatial_id)

                except Exception as e:
                    log(f"Error processing spatial unit {spatial_id}: {str(e)}")
                    continue

        # Update tracking variables
        previous_missing_count = current_missing
        iteration += 1

        # Check if no progress was made
        if len(units_filled_this_iteration) == 0:
            remaining_missing = sum(all_spatial_units[col].isna().sum() for col in value_columns)
            if remaining_missing > 0:
                log(f"No progress made in iteration {iteration}. "
                            f"{remaining_missing} values remain unfilled.")
                break

    # Prepare output
    result = all_spatial_units[['geometry'] + value_columns].copy()
    result = result.reset_index()

    # Ensure it's a GeoDataFrame
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry='geometry')

    return result


def sigmoidal_function(
    x: float,
    k: float,
    x0: float,
    invert = True,
) -> float:
    """
    Calculate sigmoidal transformation value for equilibrium index.

    Applies sigmoid function to transform input value using specified
    slope and threshold parameters. Commonly used for distance decay
    modeling and equilibrium index calculations.

    Parameters
    ----------
    x : float
        Input value to transform using sigmoid function.
        Can be any real number.
    k : float
        Slope parameter controlling steepness of sigmoid curve.
        Higher values create steeper transitions.
    x0 : float
        Threshold parameter defining sigmoid inflection point.
        Value where sigmoid function equals 0.5.
    invert : bool, default True
        If True, returns 1- sigmoidal_function, creating a decreasing
        curve

    Returns
    -------
    float
        Sigmoid transformation result bounded between 0 and 1.
        Returns values approaching 0 for large positive inputs
        and approaching 1 for large negative inputs.
    """

    idx_eq = 1 / (1 + math.exp(k * (x - x0)))
    idx_eq = -k * (x - x0)
    res = special.expit(idx_eq)
    res = 1-0 - res if invert else res
    return res


def _sigmoid_objective_function(
    x: float,
    k: float,
    x0: float,
    target_value: float
) -> float:
    """
    Helper function for sigmoid parameter optimization.

    Calculates the difference between sigmoid function output and target value.
    Used as objective function for numerical optimization.

    Parameters
    ----------
    x : float
        Input parameter to optimize (decay constant).
    k : float
        Slope parameter for sigmoid function.
    x0 : float
        Threshold parameter for sigmoid function.
    target_value : float
        Target value that sigmoid function should achieve.

    Returns
    -------
    float
        Absolute difference between sigmoid output and target value.
    """
    sigmoid_value = sigmoidal_function(x, k, x0)
    return abs(sigmoid_value - target_value) ** 2


def _find_decay_constant(
    k: float,
    x0: float,
    target_value: float,
    initial_guess: float = 0.01,
    bounds : Tuple[float, float] = (-20.0, 20.0)
) -> float:
    """
    Find optimal decay constant for sigmoid function to achieve target value.

    Uses numerical optimization to find x value that makes sigmoid function
    equal to the specified target value at given di and d0 parameters.

    Parameters
    ----------
    k : float
        Slope parameter for sigmoid function.
    x0 : float
        Threshold parameter for sigmoid function.
    target_value : float
        Desired output value from sigmoid function (between 0 and 1).
    initial_guess : float, default 0.01
        Starting point for optimization algorithm.
    bounds : tuple(float, float), default (-20.0, 20.0)
        Search interval for the solution

    Returns
    -------
    float
        Optimized decay constant that achieves target sigmoid value.

    Raises
    ------
    RuntimeError
        If optimization fails to converge to a solution.
    """
    try:
        result = optimize.minimize_scalar(
            fun=lambda x: _sigmoid_objective_function(x, k, x0, target_value),
            bounds=bounds,
            method='bounded'
        )

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x

    except Exception as e:
        raise RuntimeError(f"Failed to find decay constant: {str(e)}")


def sigmoidal_function_constant(
    positive_limit_value: float,
    mid_limit_value: float,
    target_low: float = 0.25,
    target_high: float = 0.75,
    bounds: Tuple[float,float] = (-20.0, 20.0)
) -> float:
    """
    Calculate optimal sigmoid decay constant for quarter-point constraints.

    Determines decay constant for sigmoid function that satisfies specific
    constraints at 0.25 and 0.75 quantile positions. Uses numerical
    optimization to find constant producing desired sigmoid behavior
    between specified limit values.

    Parameters
    ----------
    positive_limit_value : float
        Upper boundary value for sigmoid function domain.
        Must be greater than mid_limit_value.
    mid_limit_value : float
        Midpoint value for sigmoid function, typically the inflection point.
        Must be less than positive_limit_value.
    target_low : float, default 0.target_25
        Desired sigmoid output at the lower quarter
    target_high : float, default 0.75
        Desired sigmoid output for upper quarter
    bounds : tuple(float,float), default (-20.0,20.0)
        Bounds passed for internal optimizer value search

    Returns
    -------
    float
        Optimized decay constant for sigmoid function.
        Averaged result from quarter-point optimization constraints
        ensuring desired sigmoid behavior between limit values.

    Raises
    ------
    ValueError
        If positive_limit_value <= mid_limit_value or if inputs are invalid.
    RuntimeError
        If optimization fails to find valid decay constants.
    """
    # Input validation
    if positive_limit_value <= mid_limit_value:
        raise ValueError("positive_limit_value must be greater than mid_limit_value")

    if not all(isinstance(val, (int, float)) for val in [positive_limit_value, mid_limit_value]):
        raise ValueError("Input values must be numeric")

    # Calculate quarter-point positions
    range_diff = mid_limit_value - positive_limit_value
    quarter_75_limit = mid_limit_value - (range_diff / 2)
    quarter_25_limit = mid_limit_value + (range_diff / 2)

    # Target values for quarter points
    target_75 = 0.75
    target_25 = 0.25

    try:
        # Find decay constant for 0.75 quarter point
        decay_75 = _find_decay_constant(
            k=quarter_75_limit,
            x0=mid_limit_value,
            target_value=target_75,
            bounds=bounds
        )

        # Find decay constant for 0.25 quarter point
        decay_25 = _find_decay_constant(
            k=quarter_25_limit,
            x0=mid_limit_value,
            target_value=target_25,
            bounds=bounds
        )

        # Return average of both decay constants
        constant_value_average = (decay_75 + decay_25) / 2.0

        return constant_value_average

    except Exception as e:
        raise RuntimeError(f"Failed to calculate sigmoid constant: {str(e)}")


def interpolate_to_gdf(
    gdf: gpd.GeoDataFrame,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    power: int = 2,
    search_radius: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Interpolate values to GeoDataFrame points using inverse distance weighting.

    Applies inverse distance weighting (IDW) interpolation to estimate values
    at GeoDataFrame point locations based on known observation points.
    Supports configurable power parameter and optional search radius limiting.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries for interpolation targets.
        Must contain valid geometry column with point features.
    x : numpy.ndarray
        Array of x coordinates for known observation points.
        Must have same length as y and z arrays.
    y : numpy.ndarray
        Array of y coordinates for known observation points.
        Must have same length as x and z arrays.
    z : numpy.ndarray
        Array of values at known observation points for interpolation.
        Must have same length as x and y arrays.
    power : int, default 2
        Exponential power parameter for distance decay weighting.
        Higher values increase influence of nearby points.
    search_radius : float, optional
        Maximum distance for including observation points in interpolation.
        If None, all observation points are considered regardless of distance.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of input GeoDataFrame with added 'interpolated_value' column.
        Contains IDW interpolated values for all point locations.
    """

    gdf_int = gdf.copy()
    xi = np.array(gdf_int.geometry.x)
    yi = np.array(gdf_int.geometry.y)
    gdf_int['interpolated_value'] = interpolate_at_points(
        x, y, z, xi, yi, power, search_radius)
    return gdf_int


def idw_at_point(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    xi: float,
    yi: float,
    power: int = 2,
    search_radius: Optional[float] = None
) -> float:
    """
    Calculate inverse distance weighted interpolation at single point.

    Computes IDW interpolation value at specified coordinates using
    known observation points. Applies distance-based weighting with
    configurable power parameter and optional search radius constraint.

    Parameters
    ----------
    x0 : numpy.ndarray
        Array of x coordinates for known observation points.
        Must have same length as y0 and z0 arrays.
    y0 : numpy.ndarray
        Array of y coordinates for known observation points.
        Must have same length as x0 and z0 arrays.
    z0 : numpy.ndarray
        Array of values at known observation points.
        Must have same length as x0 and y0 arrays.
    xi : float
        X coordinate of interpolation target point.
    yi : float
        Y coordinate of interpolation target point.
    power : int, default 2
        Exponential power parameter for distance decay weighting.
        Higher values increase influence of nearby points.
    search_radius : float, optional
        Maximum distance for including observation points in calculation.
        If None, all observation points are considered.

    Returns
    -------
    float
        IDW interpolated value at target point coordinates.
        Returns -1 if no observation points within search radius.
    """
    # stack observation coordinates
    obs = np.column_stack((x0,y0))
    
    # calculate distances to target point
    dist = np.hypot(obs[:,0] - xi, obs[:,1] - yi)

    # filter distances by search radius
    if search_radius:
        idx = dist <= search_radius
        if not np.any(idx): # if there aren't any values within search radius
            return -1.0

        dist = dist[idx]
        z0 = z0[np.squeeze(idx)]

    # calculate weights
    weights = 1.0 / (dist + 1e-12)**power
    # weights sum to 1 by row
    weights /= weights.sum(axis=0)

    # check if no observation points are within limit distance
    if weights.shape[0] == 0:
        if z0.ndim == 1:
            return np.full(1, -1.0)
        else:
            return np.full(z0.shape[1], -1.0)
    # calculate dot product of weight matrix and z value matrix
    int_value = np.dot(weights.T, z0)
    return float(int_value)


def interpolate_at_points(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    power: int = 2,
    search_radius: Optional[float] = None
) -> np.ndarray:
    """
    Interpolate values at multiple points using inverse distance weighting.

    Applies IDW interpolation to estimate values at array of target coordinates
    based on known observation points. Supports vectorized computation for
    efficient processing of multiple interpolation points simultaneously.

    Parameters
    ----------
    x0 : numpy.ndarray
        Array of x coordinates for known observation points.
        Must have same length as y0 and z0 arrays.
    y0 : numpy.ndarray
        Array of y coordinates for known observation points.
        Must have same length as x0 and z0 arrays.
    z0 : numpy.ndarray
        Array of values at known observation points.
        Must have same length as x0 and y0 arrays.
    xi : numpy.ndarray
        Array of x coordinates for interpolation target points.
        Must have same length as yi array.
    yi : numpy.ndarray
        Array of y coordinates for interpolation target points.
        Must have same length as xi array.
    power : int, default 2
        Exponential power parameter for distance decay weighting.
        Higher values increase influence of nearby points.
    search_radius : float, optional
        Maximum distance for including observation points in calculations.
        If None, all observation points are considered for each target.

    Returns
    -------
    numpy.ndarray
        Array of IDW interpolated values at target point coordinates.
        Has same length as xi and yi input arrays.
    """
    x0 = np.asarray(x0).ravel()
    y0 = np.asarray(y0).ravel()
    z0 = np.asarray(z0).ravel()
    xi = np.asarray(xi).ravel()
    yi = np.asarray(yi).ravel()

    # format observed points data
    obs = np.vstack((x0, y0)).T

    # format interpolation points data
    interp = np.vstack((xi, yi)).T

    # calculate linear distance in x and y
    diff = obs[:,np.newaxis,:] - interp[np.newaxis,:,:]
    dist = np.hypot(diff[...,0], diff[...,1])


    # filter data by search radius:where
    if search_radius:
        idx = dist<=search_radius
        dist = np.where(idx, dist, np.nan)

    # calculate weights
    weights = 1.0/(dist+1e-12)**power
    weights /= np.nansum(weights, axis=0)

    # caculate dot product of weight matrix and z value matrix
    int_value = np.where(np.isnan(weights.T),0,weights.T).dot(np.where(np.isnan(z0),0,z0))

    return int_value


def weighted_average(
    df: pd.DataFrame,
    weight_column: str,
    value_column: str
) -> float:
    """
    Calculate weighted average of DataFrame column values.

    Computes weighted average by multiplying values by their corresponding
    weights and dividing by sum of weights. Handles standard weighted
    mean calculation for pandas DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing weight and value columns for calculation.
        Must contain both weight_column and value_column.
    weight_column : str
        Name of column containing weight values for averaging.
        Must exist in df and contain numeric values.
    value_column : str
        Name of column containing values to be averaged.
        Must exist in df and contain numeric values.

    Returns
    -------
    float
        Weighted average of value_column using weight_column weights.
        Result of sum(weights * values) / sum(weights).
    """
    weighted_average = (df[weight_column] * df[value_column]).sum() / df[weight_column].sum()
    return weighted_average


def voronoi_points_within_aoi(
    area_of_interest: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    points_id_col: str,
    admissible_error: float = 0.01,
    projected_crs: str = "EPSG:6372"
) -> gpd.GeoDataFrame:
    """
    Create Voronoi polygons for points constrained within area of interest.

    Generates Voronoi tessellation for input points clipped to specified
    area boundary. Uses iterative buffer expansion to ensure complete
    coverage while maintaining specified area accuracy tolerance.

    Parameters
    ----------
    area_of_interest : geopandas.GeoDataFrame
        GeoDataFrame defining spatial extent for Voronoi polygon clipping.
    points : geopandas.GeoDataFrame
        GeoDataFrame containing point locations for Voronoi generation.
    points_id_col : str
        Column name in points DataFrame containing unique point identifiers.
        Used to assign identifiers to resulting Voronoi polygons.
    admissible_error : float, default 0.01
        Maximum acceptable percentage difference between area_of_interest
        and total area of generated Voronoi polygons (as decimal).
    projected_crs : str, default "EPSG:6372"
        Coordinate reference system for area calculations and processing.
        Should be appropriate projected CRS for the area of interest.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing Voronoi polygons clipped to area of interest.
        Each polygon contains point identifier from points_id_col and
        geometry extending to area boundary with specified accuracy.
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
