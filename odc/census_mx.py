################################################################################
# Module: Census_MX
# Set of Census data treatment and analysis function for Mexico
# updated: 07/10/2025
################################################################################

import geopandas as gpd
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from .data import *
from .utils import *


@dataclass
class CensusColumns:
    """Configuration class for Mexican census column definitions (INEGI format)."""

    # Core demographic columns that follow INEGI census structure
    DEMOGRAPHIC_COLUMNS = [
        'POBFEM', 'POBMAS',
        'P_0A2', 'P_0A2_F', 'P_0A2_M',
        'P_3A5', 'P_3A5_F', 'P_3A5_M',
        'P_6A11', 'P_6A11_F', 'P_6A11_M',
        'P_12A14', 'P_12A14_F', 'P_12A14_M',
        'P_15A17', 'P_15A17_F', 'P_15A17_M',
        'P_18A24', 'P_18A24_F', 'P_18A24_M',
        'P_60YMAS', 'P_60YMAS_F', 'P_60YMAS_M',
        'P_3YMAS', 'P_3YMAS_F', 'P_3YMAS_M',
        'P_12YMAS', 'P_12YMAS_F', 'P_12YMAS_M',
        'P_15YMAS', 'P_15YMAS_F', 'P_15YMAS_M',
        'P_18YMAS', 'P_18YMAS_F', 'P_18YMAS_M',
        'REL_H_M', 'POB0_14', 'POB15_64', 'POB65_MAS',
        'PCON_DISC'
    ]

    # Columns excluded from AGEB distribution (complex calculations)
    AGEB_EXCLUDED_COLUMNS = ['REL_H_M']

    # Key identifier columns
    AGEB_ID_COLUMN = 'CVE_AGEB'
    BLOCK_ID_COLUMN = 'CVEGEO'
    TOTAL_POPULATION_COLUMN = 'POBTOT'


@dataclass
class NanCalculationStats:
    """Statistics tracking for NaN calculation process."""

    ageb_id: str
    original_nans: int
    final_nans: int
    nan_reduction_pct: float
    solved_by_equations: int
    solved_by_ageb_distribution: int
    unable_to_solve: int
    total_columns: int


def socio_polygon_to_points(
    points: gpd.GeoDataFrame,
    gdf_socio: gpd.GeoDataFrame,
    column_start: int = 0,
    column_end: int = -1,
    cve_column: str = "CVEGEO",
    avg_columns: Optional[List[str]] = None,
    target_crs: str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Assign proportional sociodemographic data from polygons to points within each polygon.

    This function distributes sociodemographic attributes from polygon areas (AGEBs)
    to point locations within those polygons. For most attributes, the values are
    divided proportionally based on the number of points in each polygon. Attributes
    specified in avg_columns are averaged instead of divided.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries to receive sociodemographic data
    gdf_socio : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries with sociodemographic attributes
    column_start : int, default 0
        Starting column index for sociodemographic data in gdf_socio
    column_end : int, default -1
        Ending column index for sociodemographic data in gdf_socio.
        If -1, uses all columns from column_start to end
    cve_column : str, default "CVEGEO"
        Column name containing unique identifiers for polygons
    avg_columns : list of str, optional
        Column names that should be averaged rather than proportionally divided.
        These typically include rates, percentages, or intensive variables
    target_crs : str, default "EPSG:4326"
        Target coordinate reference system for the output

    Returns
    -------
    geopandas.GeoDataFrame
        Input points GeoDataFrame with added sociodemographic attributes

    Raises
    ------
    KeyError
        If specified columns don't exist in the data

    """


    if cve_column not in gdf_socio.columns:
        raise KeyError(f"Column '{cve_column}' not found in gdf_socio")

    # Handle column indices
    if column_end == -1:
        column_end = len(gdf_socio.columns)

    if column_start < 0 or column_end > len(gdf_socio.columns) or column_start >= column_end:
        raise ValueError(f"Invalid column range: start={column_start}, end={column_end}, "
                        f"total_columns={len(gdf_socio.columns)}")

    # Handle avg_columns parameter
    if avg_columns is None:
        avg_columns = []

    # Validate avg_columns exist in the data
    socio_columns = gdf_socio.columns.tolist()[column_start:column_end]
    invalid_avg_columns = [col for col in avg_columns if col not in socio_columns]
    if invalid_avg_columns:
        log(f"avg_columns not found in data: {invalid_avg_columns}")
        avg_columns = [col for col in avg_columns if col in socio_columns]

    try:
        # Step 1: Calculate number of nodes within each polygon
        log(f"Performing spatial join between {len(points)} nodes and {len(gdf_socio)} polygons")

        # Ensure both GeoDataFrames have the same CRS for spatial join
        if points.crs != gdf_socio.crs:
            log(f"CRS mismatch: nodes ({points.crs}) vs gdf_socio ({gdf_socio.crs})")
            points_aligned = points.to_crs(gdf_socio.crs)
        else:
            points_aligned = points

        # Spatial join to find which nodes fall within which polygons
        joined = gpd.sjoin(points_aligned, gdf_socio[[cve_column, 'geometry']], how='left')

        # Count nodes per polygon
        point_count = (
            joined.groupby(cve_column, dropna=False)
            .size()
            .reset_index(name='points_count')
        )

        # Check for nodes that don't fall within any polygon
        points_without_polygon = joined[cve_column].isna().sum()
        if points_without_polygon > 0:
            log(f"{points_without_polygon} points don't fall within any polygon")

        log(f"Found points in {len(point_count)} polygons")

        # Step 2: Merge sociodemographic data with point counts
        socio_with_counts = pd.merge(
            gdf_socio,
            point_count,
            on=cve_column,
            how='left'
        ).copy()

        # Handle polygons with no points
        socio_with_counts['points_count'] = socio_with_counts['points_count'].fillna(0)

        # Step 3: Calculate proportional values for each column
        log(f"Processing {len(socio_columns)} sociodemographic columns")

        for col in socio_columns:
            if col in ['geometry', cve_column]:  # Skip non-data columns
                continue

            # Convert to numeric, handling any non-numeric values
            socio_with_counts[col] = pd.to_numeric(socio_with_counts[col], errors='coerce')
            socio_with_counts[col] = socio_with_counts[col].astype(float)

            # Skip division for columns with zero points (would result in inf/NaN)
            mask_has_points = socio_with_counts['points_count'] > 0

            if col in avg_columns:
                # For average columns, keep the original value (don't divide)
                log(f"Column '{col}' will be averaged (not divided)")
            else:
                # For other columns, divide by number of nodes
                socio_with_counts.loc[mask_has_points, col] = (
                    socio_with_counts.loc[mask_has_points, col] /
                    socio_with_counts.loc[mask_has_points, 'points_count']
                )

        # Step 4: Ensure target CRS
        if socio_with_counts.crs != target_crs:
            socio_with_counts = socio_with_counts.to_crs(target_crs)

        # Step 5: Spatial join nodes with processed sociodemographic data
        if points_aligned.crs != target_crs:
            points_aligned = points_aligned.to_crs(target_crs)

        result = gpd.sjoin(points_aligned, socio_with_counts, how='left')

        # Step 6: Clean up result
        # Remove helper columns and spatial join artifacts
        columns_to_drop = ['points_count', 'index_right']
        columns_to_drop = [col for col in columns_to_drop if col in result.columns]

        if columns_to_drop:
            result = result.drop(columns=columns_to_drop)

        log(f"Successfully assigned sociodemographic data to {len(result)} nodes")

        return result

    except Exception as e:
        log(f"Error in socio_polygon_to_points: {str(e)}")
        raise


def socio_points_to_polygon(
    gdf_polygon: gpd.GeoDataFrame,
    gdf_socio: gpd.GeoDataFrame,
    cve_column: str,
    string_columns: List[str],
    wgt_dict: Optional[Dict[str, str]] = None,
    avg_columns: Optional[List[str]] = None,  # Keep original name for compatibility
    include_nearest: bool = False,
    points_id_column: Optional[str] = None,
    projected_crs: str = "EPSG:6372",
    target_crs: str = "EPSG:4326"
) -> pd.DataFrame:
    """
    Aggregate sociodemographic point data within polygon boundaries.

    This function groups point-based sociodemographic data into polygon areas,
    calculating sums for extensive variables and averages for intensive variables.
    Optionally assigns points outside polygons to the nearest polygon boundary.

    Parameters
    ----------
    gdf_polygon : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries where data will be aggregated
    gdf_socio : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries with sociodemographic attributes
    cve_column : str
        Column name containing unique polygon identifiers
    string_columns : list of str
        Column names containing string/categorical data (will be excluded from aggregation)
    wgt_dict : dict, optional
        Dictionary mapping column names to weight column names for weighted averages.
        Format: {'column_to_average': 'weight_column'}
    avg_columns : list of str, optional
        Column names that should be averaged rather than summed.
        These typically include rates, percentages, or intensive variables
    include_nearest : bool, default False
        If True, assigns points outside polygons to nearest polygon boundary.
        If False, excludes points that don't fall within any polygon
    points_id_column : str, optional
        Column name containing unique point identifiers. Required if include_nearest=True
    projected_crs : str, default "EPSG:6372"
        Projected coordinate system for distance calculations when include_nearest=True
    target_crs : str, default "EPSG:4326"
        Target coordinate reference system for output

    Returns
    -------
    pandas.DataFrame
        DataFrame with aggregated sociodemographic data and polygon identifiers

    Raises
    ------
    ValueError
        If input parameters are invalid or required columns are missing
    KeyError
        If specified columns don't exist in the data

    """

    # Input validation
    if gdf_polygon.empty:
        raise ValueError("Input 'gdf_polygon' GeoDataFrame cannot be empty")
    if gdf_socio.empty:
        raise ValueError("Input 'gdf_socio' GeoDataFrame cannot be empty")

    if cve_column not in gdf_polygon.columns:
        raise KeyError(f"Column '{cve_column}' not found in gdf_polygon")

    if include_nearest and points_id_column is None:
        raise ValueError("points_id_column is required when include_nearest=True")

    if include_nearest and points_id_column not in gdf_socio.columns:
        raise KeyError(f"points_id_column '{points_id_column}' not found in gdf_socio")

    # Validate string_columns exist
    missing_string_cols = [col for col in string_columns if col not in gdf_socio.columns]
    if missing_string_cols:
        raise KeyError(f"String columns not found in gdf_socio: {missing_string_cols}")

    # Initialize optional parameters
    if wgt_dict is None:
        wgt_dict = {}
    if avg_columns is None:
        avg_columns = []

    # Validate weighted average columns
    for col, weight_col in wgt_dict.items():
        if col not in gdf_socio.columns:
            raise KeyError(f"Column for weighted average '{col}' not found in gdf_socio")
        if weight_col not in gdf_socio.columns:
            raise KeyError(f"Weight column '{weight_col}' not found in gdf_socio")

    # Validate avg_columns exist
    missing_avg_cols = [col for col in avg_columns if col not in gdf_socio.columns]
    if missing_avg_cols:
        log(f"avg_columns not found in data: {missing_avg_cols}")
        avg_columns = [col for col in avg_columns if col in gdf_socio.columns]

    try:
        log(f"Aggregating {len(gdf_socio)} points into {len(gdf_polygon)} polygons")

        # Ensure compatible CRS for spatial operations
        if gdf_polygon.crs != gdf_socio.crs:
            log(f"CRS mismatch: polygon ({gdf_polygon.crs}) vs points ({gdf_socio.crs})")
            gdf_socio_aligned = gdf_socio.to_crs(gdf_polygon.crs)
        else:
            gdf_socio_aligned = gdf_socio.copy()

        # Spatial join points to polygons
        points_in_polygons = gpd.sjoin(gdf_socio_aligned, gdf_polygon, how='left')

        # Count points that fall within polygons vs outside
        points_with_polygon = points_in_polygons[cve_column].notna().sum()
        points_outside = points_in_polygons[cve_column].isna().sum()

        log(f"Points within polygons: {points_with_polygon}, "
                   f"Points outside: {points_outside}")

        # Step 3: Handle points outside polygons if requested
        if include_nearest and points_outside > 0:
            log("Assigning points outside polygons to nearest polygon boundaries")

            # Get points that fell outside polygons
            outside_mask = points_in_polygons[cve_column].isna()
            points_outside_polys = gdf_socio_aligned[outside_mask].copy()

            if not points_outside_polys.empty:
                # Create polygon boundaries for nearest neighbor search
                polygon_boundaries = gdf_polygon.copy()
                polygon_boundaries['geometry'] = polygon_boundaries.geometry.boundary

                # Convert to projected CRS for accurate distance calculations
                points_projected = points_outside_polys.to_crs(projected_crs)
                boundaries_projected = polygon_boundaries.to_crs(projected_crs)

                # Find nearest polygon boundary for each outside point
                nearest_assignments = gpd.sjoin_nearest(
                    points_projected,
                    boundaries_projected,
                    how='left'
                )

                # Convert back to original CRS
                nearest_assignments = nearest_assignments.to_crs(gdf_polygon.crs)

                # Update the main dataset with nearest assignments
                points_in_polygons.loc[outside_mask, cve_column] = (
                    nearest_assignments[cve_column].values
                )

                # Clean up duplicate geometry columns if they exist
                geom_cols = [col for col in nearest_assignments.columns if 'geometry' in col.lower()]
                if len(geom_cols) > 1:
                    cols_to_drop = [col for col in geom_cols if col != 'geometry']
                    points_in_polygons = points_in_polygons.drop(columns=cols_to_drop, errors='ignore')

                log(f"Assigned {len(points_outside_polys)} outside points to nearest polygons")

        # Step 4: Clean up spatial join artifacts
        points_in_polygons = points_in_polygons.drop(columns=['index_right'], errors='ignore')

        # Remove points that still don't have polygon assignments
        final_points = points_in_polygons.dropna(subset=[cve_column]).copy()

        if len(final_points) < len(points_in_polygons):
            excluded_points = len(points_in_polygons) - len(final_points)
            log(f"Excluded {excluded_points} points without polygon assignments")

        # Prepare columns for aggregation
        # Identify numeric columns (excluding geometry and string columns)
        excluded_cols = string_columns + ['geometry', 'index_right', 'index_left']
        numeric_columns = [
            col for col in final_points.columns
            if col not in excluded_cols and
            pd.api.types.is_numeric_dtype(final_points[col])
        ]

        log(f"Processing {len(numeric_columns)} numeric columns for aggregation")

        # Convert numeric columns to proper types
        for col in numeric_columns:
            final_points[col] = pd.to_numeric(final_points[col], errors='coerce')

        # Aggregate data by polygon using the helper function
        aggregated_data = []

        for polygon_id in final_points[cve_column].unique():
            if pd.isna(polygon_id):
                continue

            # Filter points for current polygon
            polygon_points = final_points[final_points[cve_column] == polygon_id].copy()

            # Use the helper function for aggregation
            agg_dict = group_sociodemographic_data(
                df_socio=polygon_points,
                numeric_cols=numeric_columns,
                avg_column=avg_columns,
                avg_dict=wgt_dict
            )

            # Add polygon identifier
            agg_dict[cve_column] = polygon_id

            aggregated_data.append(agg_dict)

        # Step 8: Create final DataFrame
        result_df = pd.DataFrame(aggregated_data)

        log(f"Successfully aggregated data for {len(result_df)} polygons")

        return result_df

    except Exception as e:
        log(f"Error in socio_points_to_polygon: {str(e)}")
        raise


def group_sociodemographic_data(
    df_socio: pd.DataFrame,
    numeric_cols: List[str],
    avg_column: Optional[List[str]] = None,
    avg_dict: Optional[Dict[str, str]] = None
) -> Dict[str, Union[float, int]]:
    """
    Aggregate sociodemographic variables from DataFrame.

    This function performs aggregation of sociodemographic data, with support for
    simple sums, simple averages, and weighted averages.

    Parameters
    ----------
    df_socio : pandas.DataFrame
        DataFrame containing sociodemographic variables to aggregate
    numeric_cols : list of str
        List of numeric column names to process
    avg_column : list of str, optional
        Column names to average instead of sum
    avg_dict : dict, optional
        Dictionary mapping column names to weight column names for weighted averages.
        Format: {'column_to_average': 'weight_column'}

    Returns
    -------
    dict
        Dictionary with aggregated values for each column

    Raises
    ------
    ValueError
        If required columns for weighted averages are missing
    TypeError
        If avg_dict is not properly formatted
    """

    # Input validation
    if df_socio.empty:
        log("Empty DataFrame provided to group_sociodemographic_data")
        return {}

    # Initialize parameters
    if avg_column is None:
        avg_column = []
    if avg_dict is None:
        avg_dict = {}

    # Validate avg_dict format
    if not isinstance(avg_dict, dict):
        raise TypeError(f"avg_dict must be a dictionary, got {type(avg_dict)}")

    # Remove geometry column if present
    numeric_cols_clean = [col for col in numeric_cols if col != 'geometry']

    # Validate that weighted average columns and their weights exist
    for col, weight_col in avg_dict.items():
        if col not in df_socio.columns:
            raise ValueError(f"Column '{col}' for weighted average not found in DataFrame")
        if weight_col not in df_socio.columns:
            raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    result_dict = {}

    for col in numeric_cols_clean:
        if col not in df_socio.columns:
            log(f"Column '{col}' not found in DataFrame, skipping")
            continue

        # Convert to numeric and handle NaN values
        try:
            series = pd.to_numeric(df_socio[col], errors='coerce')
        except Exception as e:
            log(f"Could not convert column '{col}' to numeric: {e}")
            result_dict[col] = 0
            continue

        if series.isna().all():
            result_dict[col] = 0
            continue

        if col in avg_dict:
            weight_col = avg_dict[col]

            try:
                weights = pd.to_numeric(df_socio[weight_col], errors='coerce')

                # Remove rows where either value or weight is NaN
                valid_mask = series.notna() & weights.notna()
                valid_values = series[valid_mask]
                valid_weights = weights[valid_mask]

                total_weight = valid_weights.sum()

                if total_weight == 0 or len(valid_values) == 0:
                    result_dict[col] = 0
                else:
                    # Simple weighted average formula: sum(value * weight) / sum(weight)
                    weighted_sum = (valid_values * valid_weights).sum()
                    result_dict[col] = weighted_sum / total_weight

            except Exception as e:
                log(f"Error calculating weighted average for '{col}': {e}")
                result_dict[col] = 0

        elif col in avg_column:
            # Simple average
            result_dict[col] = series.mean()
        else:
            # Sum (default for extensive variables)
            result_dict[col] = series.sum()

    return result_dict



def calculate_censo_nan_values(
    pop_ageb_gdf: gpd.GeoDataFrame,
    pop_mza_gdf: gpd.GeoDataFrame,
    extended_logs: bool = False,
    columns_config: Optional[CensusColumns] = None
) -> gpd.GeoDataFrame:
    """
    Calculate and impute NaN values in Mexican census block data based on AGEB-level data.

    This function fills missing demographic data in census blocks using two approaches:
    1. Mathematical relationships between demographic variables (e.g., POBTOT = POBFEM + POBMAS)
    2. Proportional distribution of AGEB-level data to blocks based on total population

    The function is specifically designed for INEGI census data structure and maintains
    compatibility with Mexican census column naming conventions.

    Parameters
    ----------
    pop_ageb_gdf : geopandas.GeoDataFrame
        GeoDataFrame with AGEB (Area Geoestadística Básica) polygons containing population data
    pop_mza_gdf : geopandas.GeoDataFrame
        GeoDataFrame with block (manzana) polygons containing population data with NaN values
    extended_logs : bool, default False
        If True, prints detailed statistics for each AGEB during processing
    columns_config : CensusColumns, optional
        Configuration object with column definitions. Uses default INEGI structure if None

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with blocks containing population data with imputed NaN values

    Raises
    ------
    ValueError
        If input data is empty or required columns are missing
    KeyError
        If required INEGI columns are not found in the data
    """

    # Initialize configuration
    if columns_config is None:
        columns_config = CensusColumns()

    # Input validation
    if pop_ageb_gdf.empty:
        raise ValueError("AGEB GeoDataFrame cannot be empty")
    if pop_mza_gdf.empty:
        raise ValueError("Block GeoDataFrame cannot be empty")

    log("Starting NaN value calculation for census data")

    # Step 1: Validate and prepare data
    ageb_data, block_data = _prepare_census_data(pop_ageb_gdf, pop_mza_gdf, columns_config)

    # Step 2: Validate AGEB consistency
    ageb_stats = _validate_ageb_consistency(ageb_data, block_data, columns_config)

    # Step 3: Process each AGEB
    processed_blocks = []
    all_stats = []

    total_agebs = len(ageb_stats['blocks_agebs'])
    log(f"Processing {total_agebs} AGEBs for NaN calculation")

    for i, ageb_id in enumerate(ageb_stats['blocks_agebs'], 1):
        if extended_logs:
            log(f"Processing AGEB {ageb_id} ({i}/{total_agebs})")

        # Progress reporting
        _report_progress(i, total_agebs)

        # Process single AGEB
        try:
            processed_ageb, stats = _process_single_ageb(
                ageb_id=ageb_id,
                ageb_data=ageb_data,
                block_data=block_data,
                columns_config=columns_config,
                missing_agebs=ageb_stats['missing_agebs'],
                extended_logs=extended_logs
            )

            processed_blocks.append(processed_ageb)
            all_stats.append(stats)

        except Exception as e:
            log(f"Error processing AGEB {ageb_id}: {str(e)}")
            continue

    # Step 4: Combine results and generate final statistics
    if not processed_blocks:
        raise ValueError("No AGEBs were successfully processed")

    final_result = pd.concat(processed_blocks, ignore_index=True)

    # Convert back to lowercase column names (INEGI compatibility)
    final_result.columns = final_result.columns.str.lower()

    # Generate summary statistics
    _generate_summary_statistics(all_stats)

    log("Finished NaN calculation process")

    return final_result


def _prepare_census_data(
    ageb_gdf: gpd.GeoDataFrame,
    block_gdf: gpd.GeoDataFrame,
    config: CensusColumns
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Prepare and validate census data format."""

    # Create copies and standardize column names to uppercase (INEGI standard)
    ageb_data = ageb_gdf.copy()
    block_data = block_gdf.copy()

    # Convert to uppercase except geometry
    ageb_data.columns = ageb_data.columns.str.upper()
    block_data.columns = block_data.columns.str.upper()

    # Preserve geometry column name
    if 'GEOMETRY' in ageb_data.columns:
        ageb_data = ageb_data.rename(columns={'GEOMETRY': 'geometry'})
    if 'GEOMETRY' in block_data.columns:
        block_data = block_data.rename(columns={'GEOMETRY': 'geometry'})

    # Validate required columns exist
    required_ageb_cols = [config.AGEB_ID_COLUMN, config.TOTAL_POPULATION_COLUMN]
    required_block_cols = [config.AGEB_ID_COLUMN, config.BLOCK_ID_COLUMN, config.TOTAL_POPULATION_COLUMN]

    missing_ageb_cols = [col for col in required_ageb_cols if col not in ageb_data.columns]
    missing_block_cols = [col for col in required_block_cols if col not in block_data.columns]

    if missing_ageb_cols:
        raise KeyError(f"Required AGEB columns missing: {missing_ageb_cols}")
    if missing_block_cols:
        raise KeyError(f"Required block columns missing: {missing_block_cols}")

    log("Census data prepared and validated")

    return ageb_data, block_data


def _validate_ageb_consistency(
    ageb_data: gpd.GeoDataFrame,
    block_data: gpd.GeoDataFrame,
    config: CensusColumns
) -> Dict:
    """Validate consistency between AGEB and block data."""

    log("Validating AGEB consistency")

    agebs_in_ageb_data = set(ageb_data[config.AGEB_ID_COLUMN].unique())
    agebs_in_block_data = set(block_data[config.AGEB_ID_COLUMN].unique())

    # Check for empty data
    if len(agebs_in_ageb_data) == 0 and len(agebs_in_block_data) == 0:
        raise ValueError("No population data found in area of interest")

    # Find AGEBs present in blocks but missing from AGEB data
    missing_agebs = agebs_in_block_data - agebs_in_ageb_data

    if missing_agebs:
        log(f"WARNING: {len(missing_agebs)} AGEBs present in blocks but missing from AGEB data")
        log(f"Missing AGEBs will have limited NaN filling capabilities: {sorted(missing_agebs)}")

    return {
        'ageb_agebs': sorted(agebs_in_ageb_data),
        'blocks_agebs': sorted(agebs_in_block_data),
        'missing_agebs': sorted(missing_agebs)
    }


def _process_single_ageb(
    ageb_id: str,
    ageb_data: gpd.GeoDataFrame,
    block_data: gpd.GeoDataFrame,
    columns_config: CensusColumns,
    missing_agebs: List[str],
    extended_logs: bool
) -> Tuple[pd.DataFrame, NanCalculationStats]:
    """Process NaN calculation for a single AGEB."""

    # Get blocks for this AGEB
    ageb_blocks = block_data[block_data[columns_config.AGEB_ID_COLUMN] == ageb_id].copy()

    # Separate demographic columns for processing
    demographic_cols = [col for col in columns_config.DEMOGRAPHIC_COLUMNS
                       if col in ageb_blocks.columns]

    required_cols = [columns_config.BLOCK_ID_COLUMN, columns_config.TOTAL_POPULATION_COLUMN] + demographic_cols
    blocks_subset = ageb_blocks[required_cols].copy()

    # Identify blocks with some vs no demographic data
    blocks_with_data, blocks_without_data = _separate_blocks_by_data_availability(
        blocks_subset, demographic_cols, extended_logs
    )

    original_nan_count = int(blocks_with_data.isna().sum().sum()) if not blocks_with_data.empty else 0

    # Apply mathematical relationships to fill NaNs
    if not blocks_with_data.empty:
        blocks_with_data = _apply_demographic_equations(blocks_with_data, extended_logs)

    final_nan_count = int(blocks_with_data.isna().sum().sum()) if not blocks_with_data.empty else 0

    # Combine blocks back together
    processed_blocks = pd.concat([blocks_with_data, blocks_without_data], ignore_index=True) if not blocks_without_data.empty else blocks_with_data

    # Apply AGEB-level distribution for remaining NaNs
    if ageb_id not in missing_agebs:
        processed_blocks, solved_by_equations, solved_by_ageb = _apply_ageb_distribution(
            processed_blocks, ageb_data, ageb_id, columns_config, demographic_cols
        )
    else:
        solved_by_equations, solved_by_ageb, unsolved = _count_solution_methods(
            processed_blocks, demographic_cols
        )

    # Calculate statistics
    nan_reduction = ((original_nan_count - final_nan_count) / max(original_nan_count, 1)) * 100

    stats = NanCalculationStats(
        ageb_id=ageb_id,
        original_nans=original_nan_count,
        final_nans=final_nan_count,
        nan_reduction_pct=round(nan_reduction, 2),
        solved_by_equations=solved_by_equations,
        solved_by_ageb_distribution=solved_by_ageb,
        unable_to_solve=0,  # Will be calculated in AGEB distribution
        total_columns=len(demographic_cols)
    )

    # Merge back with original block data structure
    result = _merge_processed_data(ageb_blocks, processed_blocks, columns_config)

    return result, stats


def _separate_blocks_by_data_availability(
    blocks: pd.DataFrame,
    demographic_cols: List[str],
    extended_logs: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separate blocks into those with some data vs those with no demographic data."""

    # Count non-null values per row for demographic columns
    blocks['data_availability'] = blocks[demographic_cols].notna().sum(axis=1)

    blocks_with_data = blocks[blocks['data_availability'] > 0].copy()
    blocks_without_data = blocks[blocks['data_availability'] == 0].copy()

    # Clean up temporary column
    blocks_with_data = blocks_with_data.drop(columns=['data_availability'])
    blocks_without_data = blocks_without_data.drop(columns=['data_availability'])

    if extended_logs:
        log(f"Blocks with some data: {len(blocks_with_data)}, Blocks with no data: {len(blocks_without_data)}")

    return blocks_with_data, blocks_without_data


def _apply_demographic_equations(blocks: pd.DataFrame, extended_logs: bool) -> pd.DataFrame:
    """Apply demographic relationship equations to fill NaN values."""

    if extended_logs:
        log("Applying demographic equations to fill NaN values")

    original_nans = blocks.isna().sum().sum()
    iteration = 0
    max_iterations = 50  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        start_nans = blocks.isna().sum().sum()

        # Apply all demographic relationships
        blocks = _apply_gender_relationships(blocks)
        blocks = _apply_age_group_relationships(blocks)
        blocks = _apply_age_cohort_relationships(blocks)
        blocks = _apply_supplementary_relationships(blocks)

        end_nans = blocks.isna().sum().sum()

        if extended_logs:
            log(f"Iteration {iteration}: {start_nans} → {end_nans} NaN values")

        # Stop if no improvement
        if end_nans >= start_nans:
            break

    final_reduction = ((original_nans - end_nans) / max(original_nans, 1)) * 100

    if extended_logs:
        log(f"Equation solving completed: {final_reduction:.1f}% NaN reduction in {iteration} iterations")

    return blocks


def _apply_gender_relationships(blocks: pd.DataFrame) -> pd.DataFrame:
    """Apply gender-based demographic relationships."""
    
    # Total population = Female + Male population
    blocks.fillna({'POBTOT': blocks['POBFEM'] + blocks['POBMAS']}, inplace=True)
    blocks.fillna({'POBFEM': blocks['POBTOT'] - blocks['POBMAS']}, inplace=True)
    blocks.fillna({'POBMAS': blocks['POBTOT'] - blocks['POBFEM']}, inplace=True)
    
    # Age groups by gender
    age_groups = ['P_0A2', 'P_3A5', 'P_6A11', 'P_12A14', 'P_15A17', 'P_18A24', 'P_60YMAS']
    
    for age_group in age_groups:
        total_col = age_group
        female_col = f"{age_group}_F"
        male_col = f"{age_group}_M"
        
        if all(col in blocks.columns for col in [total_col, female_col, male_col]):
            blocks.fillna({total_col: blocks[female_col] + blocks[male_col]}, inplace=True)
            blocks.fillna({female_col: blocks[total_col] - blocks[male_col]}, inplace=True)
            blocks.fillna({male_col: blocks[total_col] - blocks[female_col]}, inplace=True)
    
    return blocks


def _apply_age_group_relationships(blocks: pd.DataFrame) -> pd.DataFrame:
    """Apply age group sum relationships."""
    
    # POBTOT = P_0A2 + P_3YMAS
    blocks.fillna({'P_0A2': blocks['POBTOT'] - blocks['P_3YMAS']}, inplace=True)
    blocks.fillna({'P_3YMAS': blocks['POBTOT'] - blocks['P_0A2']}, inplace=True)
    
    # Apply same logic to gender-specific columns
    blocks.fillna({'P_0A2_F': blocks['POBFEM'] - blocks['P_3YMAS_F']}, inplace=True)
    blocks.fillna({'P_3YMAS_F': blocks['POBFEM'] - blocks['P_0A2_F']}, inplace=True)
    blocks.fillna({'P_0A2_M': blocks['POBMAS'] - blocks['P_3YMAS_M']}, inplace=True)
    blocks.fillna({'P_3YMAS_M': blocks['POBMAS'] - blocks['P_0A2_M']}, inplace=True)
    
    return blocks


def _apply_age_cohort_relationships(blocks: pd.DataFrame) -> pd.DataFrame:
    """Apply relationships between age cohorts and cumulative age groups."""

    # Define age progression relationships
    age_progressions = [
        (['P_0A2', 'P_3A5', 'P_6A11'], 'P_12YMAS'),
        (['P_0A2', 'P_3A5', 'P_6A11', 'P_12A14'], 'P_15YMAS'),
        (['P_0A2', 'P_3A5', 'P_6A11', 'P_12A14', 'P_15A17'], 'P_18YMAS')
    ]

    for younger_groups, older_group in age_progressions:
        # For total population
        _apply_cohort_equations(blocks, younger_groups, older_group, 'POBTOT')
        # For female population
        younger_f = [f"{col}_F" for col in younger_groups]
        _apply_cohort_equations(blocks, younger_f, f"{older_group}_F", 'POBFEM')
        # For male population
        younger_m = [f"{col}_M" for col in younger_groups]
        _apply_cohort_equations(blocks, younger_m, f"{older_group}_M", 'POBMAS')

    return blocks


def _apply_cohort_equations(blocks: pd.DataFrame, younger_cols: List[str], older_col: str, total_col: str):
    """Apply cohort-based equations for age groups."""

    # Check if all columns exist
    all_cols = younger_cols + [older_col, total_col]
    if not all(col in blocks.columns for col in all_cols):
        return

    # Total = sum of younger groups + older group
    younger_sum = blocks[younger_cols].sum(axis=1, min_count=1)
    blocks.fillna({older_col : blocks[total_col] - younger_sum}, inplace=True)

    # Each younger group = total - other groups
    for i, target_col in enumerate(younger_cols):
        other_cols = [col for j, col in enumerate(younger_cols) if j != i] + [older_col]
        other_sum = blocks[other_cols].sum(axis=1, min_count=1)
        blocks.fillna({target_col : blocks[total_col] - other_sum}, inplace=True)


def _apply_supplementary_relationships(blocks: pd.DataFrame) -> pd.DataFrame:
    """Apply supplementary demographic relationships."""

    # Gender ratio relationship: REL_H_M = (POBMAS/POBFEM)*100
    # Only apply if we won't create division by zero
    mask_nonzero_fem = (blocks['POBFEM'] > 0) & blocks['POBMAS'].notna()
    blocks.loc[mask_nonzero_fem, 'POBMAS'].fillna(
        (blocks.loc[mask_nonzero_fem, 'REL_H_M'] / 100) * blocks.loc[mask_nonzero_fem, 'POBFEM']
    )

    mask_nonzero_rel = (blocks['REL_H_M'] > 0) & blocks['POBFEM'].notna()
    blocks.loc[mask_nonzero_rel, 'POBFEM'].fillna(
        (blocks.loc[mask_nonzero_rel, 'POBMAS'] * 100) / blocks.loc[mask_nonzero_rel, 'REL_H_M']
    )

    # Broad age group relationships: POBTOT = POB0_14 + POB15_64 + POB65_MAS
    age_groups_broad = ['POB0_14', 'POB15_64', 'POB65_MAS']
    if all(col in blocks.columns for col in age_groups_broad):
        _apply_cohort_equations(blocks, age_groups_broad[:-1], age_groups_broad[-1], 'POBTOT')

    # POB0_14 detailed breakdown
    detailed_0_14 = ['P_0A2', 'P_3A5', 'P_6A11', 'P_12A14']
    if all(col in blocks.columns for col in detailed_0_14 + ['POB0_14']):
        detailed_sum = blocks[detailed_0_14].sum(axis=1, min_count=1)
        # blocks['POB0_14'].fillna(detailed_sum, inplace=True)
        blocks.fillna({'POB0_14' : detailed_sum}, inplace=True)

        # Reverse calculation for individual age groups
        for target_col in detailed_0_14:
            other_cols = [col for col in detailed_0_14 if col != target_col]
            other_sum = blocks[other_cols].sum(axis=1, min_count=1)
            blocks.fillna({target_col : blocks['POB0_14'] - other_sum}, inplace=True)

    return blocks


def _apply_ageb_distribution(
    blocks: pd.DataFrame,
    ageb_data: gpd.GeoDataFrame,
    ageb_id: str,
    config: CensusColumns,
    demographic_cols: List[str]
) -> Tuple[pd.DataFrame, int, int]:
    """Apply AGEB-level data distribution to fill remaining NaNs."""

    # Get AGEB reference data
    ageb_row = ageb_data[ageb_data[config.AGEB_ID_COLUMN] == ageb_id]
    if ageb_row.empty:
        return blocks, 0, 0

    # Exclude complex calculation columns
    distribution_cols = [col for col in demographic_cols
                        if col not in config.AGEB_EXCLUDED_COLUMNS]

    solved_by_equations = 0
    solved_by_ageb = 0

    for col in distribution_cols:
        if col not in blocks.columns or col not in ageb_row.columns:
            continue

        nan_count = blocks[col].isna().sum()

        if nan_count == 0:
            solved_by_equations += 1
        elif nan_count == 1:
            # Single missing value - direct calculation
            ageb_total = ageb_row[col].iloc[0]
            current_sum = blocks[col].sum()
            missing_value = ageb_total - current_sum
            blocks.fillna({col : missing_value}, inplace=True)
            solved_by_ageb += 1
        elif nan_count > 1:
            # Multiple missing values - proportional distribution
            missing_mask = blocks[col].isna()
            total_pop_missing = blocks.loc[missing_mask, config.TOTAL_POPULATION_COLUMN].sum()

            if total_pop_missing > 0:
                ageb_total = ageb_row[col].iloc[0]
                current_sum = blocks[col].sum()
                missing_total = ageb_total - current_sum

                # Distribute proportionally based on total population
                blocks.loc[missing_mask, col] = (
                    missing_total *
                    blocks.loc[missing_mask, config.TOTAL_POPULATION_COLUMN] /
                    total_pop_missing
                )

            solved_by_ageb += 1

    return blocks, solved_by_equations, solved_by_ageb


def _count_solution_methods(blocks: pd.DataFrame, demographic_cols: List[str]) -> Tuple[int, int, int]:
    """Count how columns were solved when AGEB data is not available."""

    solved_by_equations = 0
    unsolved = 0

    for col in demographic_cols:
        if col not in blocks.columns:
            continue

        if blocks[col].isna().sum() == 0:
            solved_by_equations += 1
        else:
            unsolved += 1

    return solved_by_equations, 0, unsolved


def _merge_processed_data(
    original_blocks: pd.DataFrame,
    processed_blocks: pd.DataFrame,
    config: CensusColumns
) -> pd.DataFrame:
    """Merge processed demographic data back with original block structure."""

    # Get columns that were processed
    processed_cols = [config.TOTAL_POPULATION_COLUMN] + [
        col for col in config.DEMOGRAPHIC_COLUMNS
        if col in processed_blocks.columns and col not in config.AGEB_EXCLUDED_COLUMNS
    ]

    # Remove processed columns from original data
    original_reduced = original_blocks.drop(columns=processed_cols, errors='ignore')

    # Merge with processed data
    result = original_reduced.merge(
        processed_blocks[[config.BLOCK_ID_COLUMN] + processed_cols],
        on=config.BLOCK_ID_COLUMN,
        how='left'
    )

    # Restore original column order
    original_order = original_blocks.columns.tolist()
    result = result[original_order]

    return result


def _report_progress(current: int, total: int):
    """Report processing progress at key milestones."""

    milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    progress_pct = (current / total) * 100

    for milestone in milestones:
        if progress_pct >= milestone:
            log(f"NaN calculation progress: {milestone}% complete ({current}/{total} AGEBs)")
            milestones.remove(milestone)
            break


def _generate_summary_statistics(stats_list: List[NanCalculationStats]):
    """Generate and log summary statistics for the entire process."""

    if not stats_list:
        log("No statistics available - no AGEBs were processed")
        return

    # Calculate aggregate statistics
    total_solved_equations = sum(s.solved_by_equations for s in stats_list)
    total_solved_ageb = sum(s.solved_by_ageb_distribution for s in stats_list)
    total_unsolved = sum(s.unable_to_solve for s in stats_list)
    total_columns = total_solved_equations + total_solved_ageb + total_unsolved

    avg_nan_reduction = sum(s.nan_reduction_pct for s in stats_list) / len(stats_list)

    # Calculate percentages
    pct_equations = (total_solved_equations / max(total_columns, 1)) * 100
    pct_ageb = (total_solved_ageb / max(total_columns, 1)) * 100
    pct_unsolved = (total_unsolved / max(total_columns, 1)) * 100

    # Log final summary
    log("="*60)
    log("CENSUS NaN CALCULATION SUMMARY")
    log("="*60)
    log(f"Average NaN reduction using block equations: {avg_nan_reduction:.2f}%")
    log(f"Columns solved by demographic equations: {total_solved_equations} ({pct_equations:.2f}%)")
    log(f"Columns requiring AGEB data distribution: {total_solved_ageb} ({pct_ageb:.2f}%)")
    log(f"Columns unable to solve: {total_unsolved} ({pct_unsolved:.2f}%)")
    log(f"Total AGEBs processed: {len(stats_list)}")
    log("="*60)
