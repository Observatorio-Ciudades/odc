################################################################################
# Module: Raster
# Set of raster data treatment and analysis functions, mainly using Planetary Computer
# updated: 09/09/2025
################################################################################

import multiprocessing as mp
import os
import tempfile
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pymannkendall as mk
import rasterio
import rasterio.mask
import scipy.ndimage as ndimage
from dateutil.relativedelta import relativedelta
from func_timeout import func_timeout
from pystac.extensions.eo import EOExtension as eo
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from tqdm import tqdm

from .data import *
from .utils import *

# Flags to ignore division by zero and invalid floating point operations
np.seterr(divide="ignore", invalid="ignore")


# A class is created it receives a message and has a function that returns it as a string
class AvailableData(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class NanValues(Exception):
    def __init__(self, message):
        self.message = message


class PCRasterData:
    """
    A class for downloading and processing raster data from Planetary Computer.

    Required Parameters:
        gdf: GeoDataFrame with area of interest
        index_analysis: Name of the index to calculate (e.g., "ndvi", "evi")
        area_of_analysis_name: Name for the analysis area
        start_date: Start date for analysis (YYYY-MM-DD)
        end_date: End date for analysis (YYYY-MM-DD)
        tmp_dir: Directory for temporary files
        band_name_dict: Dictionary mapping band names to processing parameters in format: {'common_band_name':[bool identifier for 1/2 upscale],
        'common_band_name':[bool identifier for 1/2 upscale]}
        index_equation: List with equation for index calculation using common band names: ['common_band_name1/common_band_name2']

    Optional Parameters (with defaults):
        freq: Frequency for date range ("MS" for month start)
        sat_query: Additional query parameters for satellite data
        satellite: Satellite to use ("sentinel-2-l2a" or "landsat-c2-l2")
        projection_crs: CRS for projections ("EPSG:6372")
        catalog: STAC catalog URL
        compute_unavailable_dates: Whether to handle missing dates (True)
        missing_months_pct_limit: Max percentage of missing months allowed (50), used if compute_unavailable_dates=True
        continuous_missing_months_limit: Max consecutive missing months (6), used if compute_unavailable_dates=True
        buffer_radius: Buffer around geometries in meters (500)
        download_time_limit: Timeout for downloads in seconds (1500)
    """

    # class constants
    DEFAULT_MAX_SEARCH_DISTANCE = 50
    DEFAULT_SMOOTHING_ITERATIONS = 0
    BAND_NODATA_VALUE = 0
    TEMPERATURE_NODATA_VALUE = -124.25
    MAX_RETRY_ATTEMPTS = 5

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        index_analysis: str,
        area_of_analysis_name: str,
        start_date: str,
        end_date: str,
        tmp_dir: Union[str, Path],
        band_name_dict: Dict[str, List],
        index_equation: List[str],
        **kwargs,
    ) -> None:
        """
        Initialize PCRasterData for satellite raster processing.

        Args:
            gdf: GeoDataFrame containing the area of interest geometries
            index_analysis: Name of the vegetation/spectral index to calculate (e.g., "NDVI", "EVI")
            area_of_analysis_name: Identifier name for the analysis area (used in file naming)
            start_date: Start date for analysis in YYYY-MM-DD format
            end_date: End date for analysis in YYYY-MM-DD format
            tmp_dir: Directory path for storing temporary and output files
            band_name_dict: Mapping of band names to processing parameters
                            Format: {'band_name': [upscale_factor, band_identifier]}
            index_equation: Mathematical equation for index calculation using band names
                            Format: ['(nir-red)/(nir+red)'] for NDVI
            **kwargs: Optional parameters (freq, sat_query, satellite, etc.)

        Raises:
            TypeError: If gdf is not a GeoDataFrame
            ValueError: If required parameters are invalid or missing
        """

        # === REQUIRED PARAMETERS ===
        self.gdf: gpd.GeoDataFrame = gdf
        self.index_analysis: str = index_analysis
        self.area_of_analysis_name: str = area_of_analysis_name
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.tmp_dir = Path(tmp_dir)
        self.band_name_dict: Dict[str, List] = band_name_dict
        self.index_equation: List[str] = index_equation

        # === DEFAULT VALUES FOR OPTIONAL PARAMETERS ===
        defaults = {
            "freq": "MS",
            "sat_query": {},
            "satellite": "sentinel-2-l2a",
            "projection_crs": "EPSG:6372",
            "catalog": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "compute_unavailable_dates": True,
            "compute_month_fallback": True,
            "missing_months_pct_limit": 50,
            "continuous_missing_months_limit": 6,
            "buffer_radius": 500,
            "download_time_limit": 1500,
        }

        # === SET OPTIONAL PARAMETERS ===
        self.freq: str = kwargs.get("freq", defaults["freq"])
        self.sat_query: Dict = kwargs.get("sat_query", defaults["sat_query"].copy())
        self.satellite: str = kwargs.get("satellite", defaults["satellite"])
        self.projection_crs: str = kwargs.get(
            "projection_crs", defaults["projection_crs"]
        )
        self.catalog: str = kwargs.get("catalog", defaults["catalog"])
        self.compute_unavailable_dates: bool = kwargs.get(
            "compute_unavailable_dates", defaults["compute_unavailable_dates"]
        )
        self.compute_month_fallback: bool = kwargs.get(
            "compute_month_fallback", defaults["compute_month_fallback"]
        )
        self.missing_months_pct_limit: int = kwargs.get(
            "missing_months_pct_limit", defaults["missing_months_pct_limit"]
        )
        self.continuous_missing_months_limit: int = kwargs.get(
            "continuous_missing_months_limit",
            defaults["continuous_missing_months_limit"],
        )
        self.buffer_radius: int = kwargs.get("buffer_radius", defaults["buffer_radius"])
        self.download_time_limit: int = kwargs.get(
            "download_time_limit", defaults["download_time_limit"]
        )

        # === COMPUTED ATTRIBUTES (set during processing) ===
        self.area_of_interest: Optional[Dict] = None
        self.buffer_polygon: Optional[gpd.GeoSeries] = None
        self.time_of_interest: Optional[List[str]] = None
        self.items: Optional[List] = None
        self.date_list: Optional[List] = None
        self.complete_date_list: Optional[List] = None
        self.assets_hrefs: Optional[Dict] = None
        self.missing_months: Optional[int] = None
        self.band_name_list: Optional[List[str]] = None
        self.min_cloud_value: Optional[float] = None
        self.aoi_tiles: Optional[List] = None
        self.gdf_bb: Optional[gpd.GeoDataFrame] = None
        self.gdf_raster_test: Optional[gpd.GeoDataFrame] = None
        self.skip_date_list: Optional[List] = None
        self.iter_count: Optional[int] = None

        # === TEMPORARY PROCESSING ATTRIBUTES ===
        self.tmp_dir_name: Optional[Path] = None
        self.processing_raster_dir: Optional[Path] = None
        self.month_: Optional[int] = None
        self.year_: Optional[int] = None
        self.dates_ordered: Optional[List] = None

        # === PARAMETER VALIDATION ===
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate constructor parameters and raise helpful errors.

        Raises:
            TypeError: If parameter types are incorrect
            ValueError: If parameter values are invalid or out of range
        """

        # Validate required parameters
        if not isinstance(self.gdf, gpd.GeoDataFrame):
            raise TypeError("gdf must be a GeoDataFrame")

        if not self.gdf.crs:
            raise ValueError("gdf must have a coordinate reference system (CRS)")

        if self.gdf.empty:
            raise ValueError("gdf cannot be empty")

        # Validate string parameters
        if not isinstance(self.index_analysis, str) or not self.index_analysis.strip():
            raise ValueError("index_analysis must be a non-empty string")

        if (
            not isinstance(self.area_of_analysis_name, str)
            or not self.area_of_analysis_name.strip()
        ):
            raise ValueError("area_of_analysis_name must be a non-empty string")

        # Validate dates
        try:
            pd.to_datetime(self.start_date)
            pd.to_datetime(self.end_date)
        except:
            raise ValueError(
                "start_date and end_date must be valid date strings (YYYY-MM-DD)"
            )

        if pd.to_datetime(self.start_date) >= pd.to_datetime(self.end_date):
            raise ValueError("start_date must be before end_date")

        # Validate directories
        try:
            self.tmp_dir.mkdir(exist_ok=True)
        except:
            raise ValueError(f"tmp_dir parent directory does not exist: {self.tmp_dir}")

        # Validate numeric parameters
        if self.missing_months_pct_limit < 0 or self.missing_months_pct_limit > 100:
            raise ValueError("missing_months_pct_limit must be between 0 and 100")

        if self.continuous_missing_months_limit < 1:
            raise ValueError("continuous_missing_months_limit must be >= 1")

        if self.buffer_radius <= 0:
            raise ValueError("buffer_radius must be positive")

        if self.download_time_limit <= 0:
            raise ValueError("download_time_limit must be positive")

        # Validate satellite options
        valid_satellites = ["sentinel-2-l2a", "landsat-c2-l2"]
        if self.satellite not in valid_satellites:
            raise ValueError(
                f"currently supported satellite options are: {valid_satellites}"
            )

        # Validate band_name_dict
        if not isinstance(self.band_name_dict, dict) or not self.band_name_dict:
            raise ValueError("band_name_dict must be a non-empty dictionary")

        # Validate index_equation
        if not isinstance(self.index_equation, list):
            raise ValueError("index_equation must be a list")

    def download_raster_from_pc(self) -> pd.DataFrame:
        """
        Main workflow method to download and process satellite raster data.

        This method orchestrates the complete raster processing pipeline:
        1. Creates spatial and temporal areas of interest
        2. Queries Planetary Computer for available satellite imagery
        3. Downloads and mosaics raster data by month
        4. Calculates indices and performs quality control
        5. Interpolates missing data if requested

        Returns:
            DataFrame containing processing status for each time period with columns:
            - year: Year of the time period
            - month: Month of the time period
            - data_id: Status code (0=missing, 1=available, 11=pre-existing)
            - able_to_download: Whether data was successfully processed
            - interpolate: Whether data was interpolated (if applicable)

        Raises:
            AvailableData: If too many months are missing or consecutive gaps exceed limits
        """

        log(
            f"\n download_raster_from_pc() - {self.area_of_analysis_name} - Starting raster analysis"
        )

        # Create area of interest coordinates from hexagons to download raster data
        log("Extracting bounding coordinates from hexagons")
        self.create_area_of_interest()

        # Create time of interest (Creates a list for all to-be-analysed-months with structure [start_day/end_day,(...)])
        log("Defining time of interest")
        self.create_time_of_interest()

        # Gather items for time and area of interest (Creates of list of available image items)
        log("Gathering items for time and area of interest")
        self.gather_items()
        log(f"Fetched {len(self.items)} items")

        # Check available dates and tiles for area of interest
        log("Checking available tiles for area of interest")
        self.available_datasets()

        # Store complete amount of tiles and complete date list
        aoi_tiles = self.df_tile.columns.to_list()[
            :-1
        ]  # all df_tile columns except 'avg_cloud'
        aoi_tiles = [
            tile.replace("_cloud", "") for tile in aoi_tiles
        ]  # remove suffix '_cloud' from tile names
        self.aoi_tiles = aoi_tiles  # set aoi_tiles attribute
        log(f"Raster tiles per date: {len(self.aoi_tiles)}.")
        log(f"Raster tiles: {self.aoi_tiles}.")
        # Save complete date_list (self.date_list gets overwritten later if compute_month_fallback is used)
        if self.compute_month_fallback:
            self.complete_date_list = self.date_list

        # Create dictionary from links (assets_hrefs is a dict. of dates and links with structure {available_date:{band_n:[link]}})
        self.band_name_list = list(self.band_name_dict.keys())
        self.link_dict()
        log("Created dictionary from items")

        # Analyze available data according to raster properties (Creates df_raster_inventory for the first time)
        df_raster_inventory = self.df_date_links()

        # Test for missing months, raises errors
        if self.compute_unavailable_dates:
            self._check_available_data(df_raster_inventory)

        # Raster cropping with bounding box from earlier
        bounding_box = gpd.GeoDataFrame(geometry=self.buffer_polygon).envelope
        gdf_bb = gpd.GeoDataFrame(gpd.GeoSeries(bounding_box), columns=["geometry"])
        self.gdf_bb = gdf_bb
        log("Created bounding box for raster cropping")

        # Create GeoDataFrame to test nan values in raster
        gdf_raster_test = self.gdf.to_crs(self.projection_crs).buffer(1)
        gdf_raster_test = gdf_raster_test.to_crs("EPSG:4326")
        gdf_raster_test = gpd.GeoDataFrame(geometry=gdf_raster_test)
        self.gdf_raster_test = gdf_raster_test

        # Raster creation - Download raster data by month
        log("Starting raster creation for specified time")
        df_raster_inventory = self.create_raster_by_month(df_raster_inventory)
        log("Finished raster creation")

        # Calculate percentage of missing months
        missing_months = len(df_raster_inventory.loc[df_raster_inventory.data_id == 0])
        log(
            f"Updated missing months to {missing_months} ({round(missing_months / len(df_raster_inventory), 2) * 100}%)"
        )

        if self.compute_unavailable_dates:
            # Starts raster interpolation by predicting points from existing values and updates missing months percentage
            log("Starting raster interpolation")
            df_raster_inventory = self.raster_interpolation(df_raster_inventory)
            log("Finished raster interpolation")
            missing_months = len(
                df_raster_inventory.loc[df_raster_inventory.data_id == 0]
            )
            log(
                f"Updated missing months to {missing_months} ({round(missing_months / len(df_raster_inventory), 2) * 100}%)"
            )

        # returns final raster
        return df_raster_inventory

    def _check_available_data(self, df_raster_inventory):
        pct_missing = round(self.missing_months / len(df_raster_inventory), 2) * 100
        log(
            f"Created DataFrame with {self.missing_months} ({pct_missing}%) missing months"
        )
        if pct_missing >= self.missing_months_pct_limit:
            log(
                "_check_available_data() - Missing more than 50 percent of data points."
            )
            raise AvailableData("Missing more than 50 percent of data points")
        df_rol = (
            df_raster_inventory["data_id"]
            .rolling(self.continuous_missing_months_limit)
            .sum()
        )
        if (df_rol == 0).any():
            log("_check_available_data() - Multiple missing months together.")
            raise AvailableData("Multiple missing months together")
        del df_rol

    def create_area_of_interest(self) -> Dict:
        """
        Create spatial area of interest from input geometries.

        Transforms the input GeoDataFrame to create a buffered polygon
        and converts it to a format suitable for Planetary Computer queries.

        Returns:
            Dictionary containing GeoJSON-style polygon coordinates for the area of interest

        Sets:
            Sets self.area_of_interest and self.buffer_polygon attributes
        """

        # Create buffer around hexagons
        poly = self.gdf.to_crs(self.projection_crs).buffer(self.buffer_radius)
        poly = poly.to_crs("EPSG:4326")
        poly = gpd.GeoDataFrame(geometry=poly).dissolve().geometry
        self.buffer_polygon = poly
        # Extract coordinates from polygon as DataFrame
        coord_val = poly.bounds
        # Get coordinates for bounding box
        n = coord_val.maxy.max()
        s = coord_val.miny.min()
        e = coord_val.maxx.max()
        w = coord_val.minx.min()

        # Set the coordinates for the area of interest
        area_of_interest = {
            "type": "Polygon",
            "coordinates": [
                [
                    [e, s],
                    [w, s],
                    [w, n],
                    [e, n],
                    [e, s],
                ]
            ],
        }

        self.area_of_interest = area_of_interest

    def create_time_of_interest(self) -> List[str]:
        """
        Generate time ranges for satellite data queries.

        Creates monthly time intervals between start_date and end_date
        formatted for Planetary Computer API queries.

        Returns:
            List of date range strings in format ["YYYY-MM-DD/YYYY-MM-DD", ...]

        Sets:
            Sets self.time_of_interest attribute
        """
        df_tmp_dates = pd.DataFrame()  # temporary date dataframe
        df_tmp_dates["date"] = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.freq
        )
        # extract month and year from date
        df_tmp_dates["month"] = df_tmp_dates.apply(
            lambda row: row["date"].month, axis=1
        )
        df_tmp_dates["year"] = df_tmp_dates.apply(lambda row: row["date"].year, axis=1)

        time_of_interest = []

        # create a time range by month
        for d in range(len(df_tmp_dates)):
            month = df_tmp_dates.loc[df_tmp_dates.index == d].month.values[0]
            year = df_tmp_dates.loc[df_tmp_dates.index == d].year.values[0]

            sample_date = datetime(year, month, 1)
            first_day = sample_date + relativedelta(day=1)  # first day of the month
            last_day = sample_date + relativedelta(day=31)  # last day of the month

            # append time range to time of interest list with planetary computer format
            time_of_interest.append(
                f"{year}-{month:02d}-{first_day.day:02d}/{year}"
                + f"-{month:02d}-{last_day.day:02d}"
            )

        # Returns array with time of interest
        self.time_of_interest = time_of_interest

    def gather_items(self) -> List:
        """
        Query Planetary Computer for available satellite imagery.

        Searches the STAC catalog for satellite images that intersect
        the spatial and temporal areas of interest.

        Returns:
            List of STAC items matching the search criteria

        Sets:
            Sets self.items attribute
        """
        # gather items from planetary computer by date and area of interest
        catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        items = []

        log(
            f"Gathering items for date range {self.time_of_interest} on satellite {self.satellite} with query: {self.sat_query}"
        )

        for t in self.time_of_interest:
            try:
                search = catalog.search(
                    collections=[self.satellite],
                    intersects=self.area_of_interest,
                    datetime=t,
                    query=self.sat_query,
                )

                # Check how many items were returned
                items.extend(list(search.items()))
            except:
                log(f"No items found on datetime {t}.")
                continue

        self.items = items

    def available_datasets(self) -> List:
        """
        Filter satellite imagery by cloud coverage and quality metrics.

        Analyzes cloud coverage statistics across tiles and dates,
        filtering out images with excessive cloud coverage.

        Returns:
            List of datetime objects representing dates with acceptable data quality

        Sets:
            Sets self.date_list and self.min_cloud_value attributes
        """
        if self.sat_query:
            if "eo:cloud_cover" in list(self.sat_query.keys()):
                self.min_cloud_value = self.sat_query["eo:cloud_cover"]["lt"]

        # test raster outliers by date
        date_dict = {}

        # iterate over raster tiles by date
        for i in self.items:
            if self.satellite == "sentinel-2-l2a":
                # check and add raster properties to dictionary by tile and date
                # if date is within dictionary append properties from item to list
                if i.datetime.date() in list(date_dict.keys()):
                    # gather cloud percentage, high_proba_clouds_percentage, no_data values and nodata_pixel_percentage
                    # check if properties are within dictionary date keys
                    if i.properties["s2:mgrs_tile"] + "_cloud" in list(
                        date_dict[i.datetime.date()].keys()
                    ):
                        date_dict[i.datetime.date()].update(
                            {
                                i.properties["s2:mgrs_tile"] + "_cloud": i.properties[
                                    "s2:high_proba_clouds_percentage"
                                ]
                            }
                        )
                    else:
                        date_dict[i.datetime.date()].update(
                            {
                                i.properties["s2:mgrs_tile"] + "_cloud": i.properties[
                                    "s2:high_proba_clouds_percentage"
                                ]
                            }
                        )
                # create new date key and add properties to it
                else:
                    date_dict[i.datetime.date()] = {}
                    date_dict[i.datetime.date()].update(
                        {
                            i.properties["s2:mgrs_tile"] + "_cloud": i.properties[
                                "s2:high_proba_clouds_percentage"
                            ]
                        }
                    )

            elif self.satellite == "landsat-c2-l2":
                # check and add raster properties to dictionary by tile and date
                # if date is within dictionary append properties from item to list
                if i.datetime.date() in list(date_dict.keys()):
                    # gather cloud percentage, high_proba_clouds_percentage, no_data values and nodata_pixel_percentage
                    # check if properties are within dictionary date keys
                    if (
                        f"{int(i.properties['landsat:wrs_path']):03d}{int(i.properties['landsat:wrs_row']):03d}_cloud"
                        in list(date_dict[i.datetime.date()].keys())
                    ):
                        date_dict[i.datetime.date()].update(
                            {
                                f"{int(i.properties['landsat:wrs_path']):03d}{int(i.properties['landsat:wrs_row']):03d}_cloud": i.properties[
                                    "landsat:cloud_cover_land"
                                ]
                            }
                        )
                    else:
                        date_dict[i.datetime.date()].update(
                            {
                                f"{int(i.properties['landsat:wrs_path']):03d}{int(i.properties['landsat:wrs_row']):03d}_cloud": i.properties[
                                    "landsat:cloud_cover_land"
                                ]
                            }
                        )
                # create new date key and add properties to it
                else:
                    date_dict[i.datetime.date()] = {}
                    date_dict[i.datetime.date()].update(
                        {
                            f"{int(i.properties['landsat:wrs_path']):03d}{int(i.properties['landsat:wrs_row']):03d}_cloud": i.properties[
                                "landsat:cloud_cover_land"
                            ]
                        }
                    )

        # determine third quartile for each tile
        df_tile = pd.DataFrame.from_dict(date_dict, orient="index")
        column_list = df_tile.columns.to_list()
        for c in range(len(column_list)):
            df_tile.loc[
                df_tile[column_list[c]] > self.min_cloud_value, column_list[c]
            ] = np.nan

        # arrange by cloud coverage average
        df_tile["avg_cloud"] = df_tile.mean(axis=1)
        df_tile = df_tile.sort_values(by="avg_cloud")
        log(f"Updated average cloud coverage: {df_tile.avg_cloud.mean()}")

        # create list of dates within normal distribution
        if self.compute_month_fallback:
            # List contains all dates with at least one tile within cloud coverage limits in order to allow month fallback [Heavier processing]
            date_list = df_tile.index.to_list()
        else:
            # List contains only dates with ALL tiles within cloud coverage limits [Faster processing]
            date_list = df_tile.dropna().index.to_list()
        log(f"Available dates: {len(date_list)}")

        # Return values
        self.date_list = date_list
        self.df_tile = df_tile

    def link_dict(self) -> Dict:
        """
        Extract download URLs for satellite imagery assets.

        Creates a mapping between dates and signed URLs for downloading
        specific satellite bands from Planetary Computer.

        Returns:
            Dictionary with structure: {date: {band_name: [url_list]}}

        Sets:
            Sets self.assets_hrefs attribute
        """
        assets_hrefs = {}

        for i in self.items:
            # only takes into account dates that are in the date list
            if i.datetime.date() not in self.date_list:
                continue
            # if date already in dictionary, append link to list
            if i.datetime.date() in list(assets_hrefs.keys()):
                for b in self.band_name_list:
                    assets_hrefs[i.datetime.date()][b].append(
                        pc.sign(self._find_asset_by_band_common_name(i, b).href)
                    )
            # if date not in dictionary, create new dictionary entry
            else:
                assets_hrefs[i.datetime.date()] = {}
                for b in self.band_name_list:
                    assets_hrefs[i.datetime.date()].update({b: []})
                    assets_hrefs[i.datetime.date()][b].append(
                        pc.sign(self._find_asset_by_band_common_name(i, b).href)
                    )

        self.assets_hrefs = assets_hrefs

    def _find_asset_by_band_common_name(self, item, common_name):
        """
        Search for a band's common name.

        Function that searches for a specific band within a SpatioTemporal Asset Catalog (STAC) and returns the corresponding asset.

        Raises:
            KeyError: If common_name is not found

        Returns:
            asset (object): Asset with the common name
        """

        for asset in item.assets.values():
            asset_bands = eo.ext(asset).bands
            if asset_bands and asset_bands[0].common_name == common_name:
                return asset
        raise KeyError(f"{common_name} band not found")

    def df_date_links(self) -> pd.DataFrame:
        """
        Create inventory DataFrame tracking data availability by month.

        Converts the asset URL dictionary into a structured DataFrame
        that tracks data availability for each month in the analysis period.

        Returns:
            DataFrame with columns:
            - year: Year of the time period
            - month: Month of the time period
            - data_id: Binary indicator (1=data available, 0=missing)
            - able_to_download: Placeholder for download success status

        Sets:
            Sets self.missing_months attribute
        """
        # dictionary to dataframe
        df_dates = (
            pd.DataFrame.from_dict(self.assets_hrefs, orient="Index")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        df_dates["date"] = pd.to_datetime(df_dates["date"]).dt.date
        df_dates["year"] = df_dates.apply(lambda row: row["date"].year, axis=1)
        df_dates["month"] = df_dates.apply(lambda row: row["date"].month, axis=1)

        df_dates_filtered = pd.DataFrame()

        # keep only one data point by month
        for y in df_dates["year"].unique():
            for m in df_dates.loc[df_dates["year"] == y, "month"].unique():
                df_dates_filtered = pd.concat(
                    [
                        df_dates_filtered,
                        df_dates.loc[
                            (df_dates["year"] == y) & (df_dates["month"] == m)
                        ].sample(1),
                    ],
                    ignore_index=True,
                )

        # create full range time dataframe
        df_tmp_dates = pd.DataFrame()  # temporary date dataframe
        df_tmp_dates["date"] = pd.date_range(
            start=self.start_date, end=self.end_date, freq=self.freq
        )  # create date range
        # extract year and month from date
        df_tmp_dates["year"] = df_tmp_dates.apply(lambda row: row["date"].year, axis=1)
        df_tmp_dates["month"] = df_tmp_dates.apply(
            lambda row: row["date"].month, axis=1
        )

        # remove date column for merge
        df_tmp_dates.drop(columns=["date"], inplace=True)

        df_complete_dates = df_tmp_dates.merge(
            df_dates_filtered,
            left_on=["year", "month"],
            right_on=["year", "month"],
            how="left",
        )

        # remove date
        df_complete_dates.drop(columns="date", inplace=True)
        df_complete_dates.sort_values(by=["year", "month"], inplace=True)

        # create binary column for available (1) or missing data (0)
        band_name_list = list(self.band_name_dict.keys())
        idx = df_complete_dates[band_name_list[0]].isna()
        df_complete_dates["data_id"] = 0
        df_complete_dates.loc[~idx, "data_id"] = 1

        df_complete_dates.drop(columns=band_name_list, inplace=True)

        # calculate missing months
        missing_months = len(df_complete_dates.loc[df_complete_dates.data_id == 0])

        # create empty able_to_download to avoid crash
        df_complete_dates["able_to_download"] = np.nan
        df_complete_dates["download_method"] = "none"

        self.missing_months = missing_months

        return df_complete_dates

    def create_raster_by_month(
        self,
        df_raster_inventory: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Download and process satellite rasters for each month.

        Main processing loop that downloads satellite imagery, creates mosaics,
        calculates indices, and performs quality control for each
        month in the analysis period.

        Args:
            df_raster_inventory: DataFrame tracking data availability by month

        Returns:
            Updated DataFrame with processing results for each month

        Side Effects:
            Creates raster files in tmp_dir_name directory
            Updates CSV file tracking processing progress
        """
        df_raster_inventory["able_to_download"] = np.nan

        log("\n Starting raster analysis")

        df_file_dir = self._define_processing_directory(df_raster_inventory)

        for i in tqdm(range(len(df_raster_inventory)), position=0, leave=True):
            # read dataframe in each iteration in case of code crash
            df_raster = pd.read_csv(
                df_file_dir,
                dtype={"download_method": "string"},  # force string dtype
            )
            df_raster["download_method"] = df_raster["download_method"].fillna("none")

            # checks if raster was previously downloaded and if it is available for processing
            if self._check_preexisting_files(i, df_file_dir, df_raster):
                continue
            # preprocess dates for download
            self.raster_download_date_preprocessing()

            # mosaic raster iterations (while loop tries defined times to process all available rasters (dates) in a month)
            self.download_and_process_raster_by_date(i, df_file_dir, df_raster)

        df_raster_inventory = pd.read_csv(df_file_dir, index_col=False)

        return df_raster_inventory

    def download_and_process_raster_by_date(
        self, i: int, df_file_path: Path, df_raster: pd.DataFrame
    ) -> None:
        """
         Download and process raster data for a specific month with retry logic.

         Attempts to download, mosaic, and process satellite data for the current
         month (self.month_, self.year_) with multiple retry attempts and quality control.

         Args:
             i: Row index in the raster inventory DataFrame
             df_file_path: Path to the inventory CSV file
             df_raster: Current state of the raster inventory DataFrame

        Side Effects:
             Modifies df_raster according to checker

        """

        # set checker to False
        checker = False
        # set current iteration
        self.iter_count = 1
        # create skip date list used to analyze null values in raster
        self.skip_date_list = []

        while self.iter_count <= self.MAX_RETRY_ATTEMPTS:
            # gather links for the date range from planetary computer
            self.gather_items()

            # gather links from dates that are within date_list
            self.link_dict()

            # log amount of available dates for the month compared to all available tiles
            if self.compute_month_fallback:
                month_tiles = []
                for item in self.items:
                    # if item's date is in assets_hrefs keys, check for unique tiles
                    if item.datetime.date() in list(self.assets_hrefs.keys()):
                        # For sentinel-2-l2a, gather unique mgrs_tile values
                        if self.satellite == "sentinel-2-l2a":
                            item_tile = item.properties["s2:mgrs_tile"]
                            if item_tile not in month_tiles:
                                month_tiles.append(item_tile)
                        # For landsat-c2-l2, gather unique wrs_path + wrs_row values
                        elif self.satellite == "landsat-c2-l2":
                            item_tile = (
                                item.properties["landsat:wrs_path"]
                                + item.properties["landsat:wrs_row"]
                            )
                            if item_tile not in month_tiles:
                                month_tiles.append(item_tile)
                if len(self.aoi_tiles) > len(month_tiles):
                    log(
                        f"NOTE: Insufficient tiles to cover area of interest. Needed: {len(self.aoi_tiles)}, available: {len(month_tiles)}."
                    )
                    log(
                        f"NOTE: Available tiles: {month_tiles}. Missing tiles: {list(set(self.aoi_tiles) - set(month_tiles))}."
                    )
                else:
                    log(
                        f"NOTE: Month has all available tiles within area of interest: {month_tiles}."
                    )

            # ------------------------------ LINKS ANALISYS A - ORDERED ACCORDING TO CLOUD COVERAGE [PREFERRED] ------------------------------
            # Create list of links ordered according to cloud coverage
            ordered_bandlink_dicts = []
            for data_position in range(len(self.dates_ordered)):
                ordered_bandlink_dicts.append(
                    self.assets_hrefs[self.dates_ordered[data_position]]
                )
            # Process ordered dates
            ordered_links_try = 0
            for bands_links in ordered_bandlink_dicts:
                log(
                    f"{self.dates_ordered[ordered_links_try]} - ITERATION {self.iter_count} - DATE {ordered_links_try + 1}/{len(ordered_bandlink_dicts)}."
                )
                checker = self.links_iteration(
                    checker,
                    bands_links,
                    specific_date=(True, self.dates_ordered[ordered_links_try]),
                )
                # If succeded current date, stop ordered dates iterations
                if checker:
                    break
                # Else, try next date
                ordered_links_try += 1
            # If succeded by any date, stop month's while loop (Doesn't try whole month's available links)
            if checker:
                download_method = "specific_date"
                break

            # ------------------------------ LINKS ANALISYS B - WHOLE MONTH'S AVAILABLE LINKS [BACKUP, month_fallback] ------------------------------
            if self.compute_month_fallback:
                log(
                    f"{self.year_}-{self.month_:02d} - MONTH ITERATION {self.iter_count}."
                )

                # --- GATHER UPDATED LINKS - Since links expire after some time, they are gathered at each iteration
                # gather items for the date range from planetary computer
                self.gather_items()

                # --- FIND BEST DATES PER TILE - From all month's available items, find the date with lowest cloud coverage for each tile
                # Re-create df_tile (tiles with cloud pct dataframe) for currently explored dates
                self.available_datasets()
                df_tile_current = self.df_tile.copy()
                # Drop 'avg_cloud' column
                df_tile_current.drop(columns=["avg_cloud"], inplace=True)
                # Drop all tile columns with no data (where mean is nan) and list the rest
                df_tile_current = df_tile_current.drop(
                    columns=df_tile_current.columns[
                        df_tile_current.mean(skipna=True).isna()
                    ]
                )
                tiles_lst = df_tile_current.columns.to_list()
                # Reset index to place date as a column
                df_tile_current.reset_index(inplace=True)
                df_tile_current.rename(columns={"index": "date"}, inplace=True)
                # For each tile, find the date where the clouds percentage is lowest and append date to perform month's analysis
                best_dates_tiles = {}
                for tile in tiles_lst:
                    # Find date where tile has lowest cloud percentage
                    mincloud_idx = df_tile_current[tile].min()
                    mincloud_date = df_tile_current.loc[
                        df_tile_current[tile] == mincloud_idx
                    ]["date"].unique()[0]
                    log(
                        f"Tile {tile.replace('_cloud', '')} has lowest cloud coverage on date {mincloud_date}."
                    )
                    # Save date and tile in dictionary
                    if mincloud_date in list(best_dates_tiles.keys()):
                        # Append to existing list
                        tiles_lst = best_dates_tiles[mincloud_date]
                        tiles_lst.append(tile)
                        best_dates_tiles[mincloud_date] = tiles_lst
                    else:
                        # Inicialize list
                        best_dates_tiles[mincloud_date] = [tile]

                # --- FILTER ITEMS FROM BEST DATES - Creates list of items with best dates and tiles only
                # gather links from dates that are within dates_month_min_cloud
                dates_month_min_cloud = list(best_dates_tiles.keys())
                filtered_items = []
                for item in self.items:
                    # If item's date in filtered dates
                    if item.datetime.date() in dates_month_min_cloud:
                        # Check current item's tile
                        if self.satellite == "sentinel-2-l2a":
                            tile = item.properties["s2:mgrs_tile"]
                        elif self.satellite == "landsat-c2-l2":
                            tile = (
                                item.properties["landsat:wrs_path"]
                                + item.properties["landsat:wrs_row"]
                            )
                        tile = tile + "_cloud"
                        # If tile inside dict, append its item to filtered_items
                        if tile in best_dates_tiles[item.datetime.date()]:
                            filtered_items.append(item)
                            log(
                                f"Appended item for tile {tile.replace('_cloud', '')} on date {item.datetime.date()} to month fallback analysis."
                            )

                # --- CREATE LINKS DICTIONARY - Create dictionary of links with best dates and tiles only.
                # change date_list to dates_month_min_cloud to create assets_hrefs with best dates only
                self.date_list = dates_month_min_cloud
                # gather links from dates that are within dates_month_min_cloud
                self.link_dict()
                # recover complete date list if using compute_month_fallback, since the full date list is used in other parts of the code
                self.date_list = self.complete_date_list

                # --- PROCESS LINKS - Use links_iteration() function
                # Create one month dictionary with bands as keys and list of links as values
                best_links_by_band = {}
                for data_position in range(len(dates_month_min_cloud)):
                    current_link_dct = self.assets_hrefs[
                        dates_month_min_cloud[data_position]
                    ]
                    for band, links in current_link_dct.items():
                        if band not in best_links_by_band:
                            best_links_by_band[
                                band
                            ] = []  # Initialize list if band not in dictionary
                        best_links_by_band[band].extend(
                            links
                        )  # Append links to the list for the band
                # Process all available links for the month
                checker = self.links_iteration(
                    checker, bands_links=best_links_by_band, specific_date=(False, None)
                )
                # If succeded whole month, stop while loop
                if checker:
                    download_method = "month_fallback"
                    break

            # Try next iteration (If not reached MAX_RETRY_ATTEMPTS)
            self.iter_count += 1

        # Current month's iteration finished, update df_raster according to checker
        if checker:
            log(
                f"Processed month {self.month_}/{self.year_}. Updating df_raster and moving to next month."
            )
            df_raster.loc[df_raster.index == i, "able_to_download"] = 1
            df_raster.loc[df_raster.index == i, "download_method"] = download_method
            df_raster.to_csv(df_file_path, index=False)
        else:
            log(
                f"Could not process month {self.month_}/{self.year_}. Updating df_raster and moving to next month."
            )
            df_raster.loc[df_raster.index == i, "data_id"] = 0
            df_raster.loc[df_raster.index == i, "able_to_download"] = 0
            df_raster.loc[df_raster.index == i, "download_method"] = "could_not_process"
            df_raster.to_csv(df_file_path, index=False)
            if self.compute_unavailable_dates:
                self._check_available_data(df_raster)  # test for missing months

    def links_iteration(self, checker, bands_links, specific_date) -> bool:
        """
        Saves processed rasters to a local directory by iterating over a list of links for each band,
        creating a mosaic, calculating the specified raster index and interpolating missing values (pixels).

        The function recieves a list of links for each band and a specific date (if any).

        Args:
            checker (bool): Boolean to check if processing succeeded.
            bands_links (dict): Dictionary with the links to the assets for each band.
            specific_date (tupple): Tupple with a boolean and a date to attempt.

        Sets:
            Sets self.skip_date_list attribute

        Returns:
            True if processing succeeded, False if all attempts failed

        Side Effects:
            Cleans up temporary files after each attempt
        """

        # If attempting to process satellite data from a specific date:
        if specific_date[0]:
            # Retrieve specified date from tupple
            date_attempt = specific_date[1]
            # Skip date if date in skip_date_list
            if date_attempt in self.skip_date_list:
                log(
                    f"{date_attempt} - ITERATION {self.iter_count} - Skipped date because previously it did not pass null test."
                )
                return checker
            # Else, log current date attempt
            log(f"Skip list:{self.skip_date_list}")
            log(
                f"Mosaic date {date_attempt.day}"
                + f"/{date_attempt.month}"
                + f"/{date_attempt.year} - iteration:{self.iter_count}"
            )
        try:
            # Mosaic process
            rasters_arrays = func_timeout(
                self.download_time_limit, self.mosaic_process, args=(bands_links,)
            )
            out_meta = rasters_arrays[list(rasters_arrays.keys())[0]][2]

            # Calculate raster index
            raster_idx = self.calculate_raster_index(rasters_arrays)
            log(f"Calculated {self.index_analysis}")
            del rasters_arrays

            # Interpolation process
            log(f"Starting interpolation")
            raster_idx[raster_idx == self.BAND_NODATA_VALUE] = (
                np.nan
            )  # change zero values to nan
            raster_idx[raster_idx == self.TEMPERATURE_NODATA_VALUE] = (
                np.nan
            )  # change zero values to nan (only for temperature)
            raster_idx[np.isinf(raster_idx)] = np.nan  # change inf values to nan
            raster_idx[np.isnan(raster_idx)] = (
                np.nan
            )  # change nan values to nan (to avoid errors)
            raster_idx = raster_idx.astype(
                "float32"
            )  # change data type to float32 to avoid fillnodata error

            log(f"Interpolating {np.isnan(raster_idx).sum()} nan values")
            raster_fill = fillnodata(
                raster_idx,
                mask=~np.isnan(raster_idx),
                max_search_distance=self.DEFAULT_MAX_SEARCH_DISTANCE,
                smoothing_iterations=self.DEFAULT_SMOOTHING_ITERATIONS,
            )
            log(
                f"Finished interpolation to fill na - {np.isnan(raster_fill).sum()} nan"
            )

            # save filled raster to temporary directory for further processing
            output_raster_path = (
                self.processing_raster_dir / f"{self.index_analysis}.tif"
            )
            self.save_output_raster(raster_fill, output_raster_path, out_meta)

            log("Starting null test")
            raster_file = rasterio.open(
                f"{self.processing_raster_dir}/{self.index_analysis}.tif"
            )
            gdf_raster_test = self.gdf_raster_test.to_crs(raster_file.crs)

            try:
                # test for nan values within study area
                self._raster_nan_test(gdf_raster_test, raster_file)
                log("Passed null test")

                # save raster to processing database
                index_raster_dir = (
                    self.tmp_dir_name
                    / f"{self.area_of_analysis_name}_{self.index_analysis}_{self.month_}_{self.year_}.tif"
                )
                self.save_output_raster(raster_fill, index_raster_dir, out_meta)
                log(f"Finished saving {self.index_analysis} raster")

                # log success
                if specific_date[0]:
                    log(f"{date_attempt} - ITERATION {self.iter_count} - SUCCESS.")
                else:
                    log(f"ALL TILES IN MONTH - ITERATION {self.iter_count} - SUCCESS.")
                clear_directory(self.processing_raster_dir)

                checker = True
                return checker

            except:
                # log null test failure
                if specific_date[0]:
                    log(
                        f"{date_attempt} - ITERATION {self.iter_count} - ERROR: FAILED NULL TEST ON DATE."
                    )
                    self.skip_date_list.append(date_attempt)
                else:
                    log(
                        f"ALL TILES IN MONTH - ITERATION {self.iter_count} - ERROR: FAILED NULL TEST ON ALL TILES."
                    )
                clear_directory(self.processing_raster_dir)
                return checker

        except:
            # log other failure
            log(
                f"ERROR IN ITERATION {self.iter_count}. Posible causes: links expired, mosaic raster failed, mosaic interpolation failed."
            )
            clear_directory(self.processing_raster_dir)
            return checker

    def mosaic_process(
        self,
        raster_bands: Dict[str, List],
    ) -> Dict:
        """
        Create mosaics from multiple satellite tiles for each band.

        Downloads and merges satellite tiles for each spectral band,
        with optional upscaling and cropping to area of interest.

        Args:
            raster_bands: Dictionary mapping band names to lists of download URLs

        Returns:
            Dictionary containing processed raster arrays, transforms, and metadata
            for each band: {band_name: [array, transform, metadata]}
        """

        raster_array = {}

        band_names_list = list(self.band_name_dict.keys())

        for b in band_names_list:
            log(f"Starting mosaic for {b}")
            raster_array[b] = [
                self.mosaic_raster(raster_bands[b], upscale=self.band_name_dict[b][0])
            ]
            # mosaic_raster creates a tuple which has to be unpacked
            raster_array[b] = [
                raster_array[b][0][0],
                raster_array[b][0][1],
                raster_array[b][0][2],
            ]
            raster_array[b][0] = raster_array[b][0].astype("float32")

            raster_array[b][2].update(
                {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "height": raster_array[b][0].shape[1],
                    "width": raster_array[b][0].shape[2],
                    "transform": raster_array[b][1],
                    "crs": self.projection_crs,
                }
            )

            log(f"Starting save: {b}")

            output_raster_path = self.processing_raster_dir / f"{b}.tiff"
            self.save_output_raster(
                raster_array[b][0], output_raster_path, raster_array[b][2]
            )

            raster_array[b][0] = [np.nan]
            log("Finished saving complete dataset")

            log("Starting crop")
            gdf_bb = self.gdf_bb
            with rasterio.open(output_raster_path) as src:
                gdf_bb = gdf_bb.to_crs(src.crs)
                shapes = [
                    gdf_bb.iloc[feature].geometry for feature in range(len(gdf_bb))
                ]
                raster_array[b][0], raster_array[b][1] = rasterio.mask.mask(
                    src, shapes, crop=True
                )
                raster_array[b][2] = src.meta
                raster_array[b][2].update(
                    {
                        "driver": "GTiff",
                        "dtype": "float32",
                        "height": raster_array[b][0].shape[1],
                        "width": raster_array[b][0].shape[2],
                        "transform": raster_array[b][1],
                        "crs": self.projection_crs,
                    }
                )
                src.close()

            output_raster_path = self.processing_raster_dir / f"{b}.tiff"
            self.save_output_raster(
                raster_array[b][0], output_raster_path, raster_array[b][2]
            )

            raster_array[b][0] = raster_array[b][0].astype("float32")

            log(f"Finished croping: {b}")

            log(f"Finished processing {b}")

        return raster_array

    def mosaic_raster(
        self, raster_asset_list: List[str], upscale: bool = False
    ) -> Tuple[np.ndarray, Any, Dict]:
        """
        Merge multiple raster tiles into a single mosaic.

        Args:
            raster_asset_list: List of raster file URLs or paths to merge
            upscale: Whether to upscale the mosaic by 2x using bilinear resampling

        Returns:
            Tuple containing:
            - mosaic: Merged raster array
            - out_trans: Transformation matrix for the mosaic
            - meta: Metadata dictionary for the mosaic
        """

        src_files_to_mosaic = []

        # Raster reprojection
        tmp_files = []
        for assets in raster_asset_list:
            with rasterio.open(assets) as src:
                # Reproject if necessary
                if src.crs != self.projection_crs:
                    log(f"mosaic_raster() - Reprojecting tile.")
                    # Crear transform and new metadata
                    transform, width, height = calculate_default_transform(
                        src.crs, self.projection_crs, src.width, src.height, *src.bounds
                    )
                    kwargs = src.meta.copy()
                    kwargs.update(
                        {
                            "crs": self.projection_crs,
                            "transform": transform,
                            "width": width,
                            "height": height,
                        }
                    )

                    # Save temporary reprojected file
                    tmp_file = tempfile.NamedTemporaryFile(
                        suffix=".tif", delete=False
                    ).name
                    with rasterio.open(tmp_file, "w", **kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=self.projection_crs,
                                resampling=Resampling.nearest,
                            )
                    tmp_files.append(tmp_file)
                    src_files_to_mosaic.append(rasterio.open(tmp_file))
                else:
                    src_files_to_mosaic.append(rasterio.open(assets))

        # Merge raster tiles
        log(f"mosaic_raster() - Merging {len(src_files_to_mosaic)} tiles.")
        mosaic, out_trans = merge(src_files_to_mosaic, method="first")
        log(f"mosaic_raster() - Merged {len(src_files_to_mosaic)} tiles.")

        meta = src.meta

        if upscale:
            # save raster
            out_meta = src.meta

            out_meta.update(
                {
                    "driver": "GTiff",
                    "dtype": "float32",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "crs": self.projection_crs,
                }
            )
            # write raster
            output_raster_path = self.processing_raster_dir / "mosaic_upscale.tif"
            self.save_output_raster(mosaic, output_raster_path, out_meta)

            # read and upscale
            with rasterio.open(
                self.processing_raster_dir + "/mosaic_upscale.tif", "r"
            ) as ds:
                upscale_factor = 1 / 2

                mosaic = ds.read(
                    out_shape=(
                        ds.count,
                        int(ds.height * upscale_factor),
                        int(ds.width * upscale_factor),
                    ),
                    resampling=Resampling.bilinear,
                )
                out_trans = ds.transform * ds.transform.scale(
                    (ds.width / mosaic.shape[-1]), (ds.height / mosaic.shape[-2])
                )
                meta = ds.meta

                meta.update(
                    {
                        "driver": "GTiff",
                        "dtype": "float32",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        "crs": self.projection_crs,
                    }
                )

            ds.close()
        src.close()

        return mosaic, out_trans, meta

    def calculate_raster_index(self, raster_arrays: Dict) -> np.ndarray:
        """
        Calculate specified index from spectral bands.

        Applies the mathematical equation defined in index_equation to compute
        specified indices (e.g., NDVI, EVI) from spectral band arrays.

        Args:
            raster_arrays: Dictionary containing band arrays and metadata
                          Format: {band_name: [array, transform, metadata]}

        Returns:
            Calculated index as numpy array
            If no equation provided, returns the first band array.
        """

        if len(self.index_equation) == 0:
            # if there is no equation the raster array is the result
            raster_idx = raster_arrays[list(raster_arrays.keys())[0]]
            return raster_idx

        raster_equation = self.index_equation[0]

        for rb in raster_arrays.keys():
            raster_equation = raster_equation.replace(rb, f"ra['{rb}'][0]")
        # create global variable in order to use it in exec as global
        global ra
        ra = raster_arrays
        exec(f"raster_index = {raster_equation}", globals())

        del ra

        return raster_index

    def _define_processing_directory(self, df_raster_inventory: pd.DataFrame) -> Path:
        """
        Set up directory structure for processing and create inventory file.

        Args:
            df_raster_inventory: DataFrame to save as processing inventory

        Returns:
            Path to the inventory CSV file

        Side Effects:
            Creates directory structure and saves initial inventory file
            Sets self.tmp_dir_name and self.processing_raster_dir attributes
        """

        # Create area-specific directory
        self.tmp_dir_name = self.tmp_dir / self.area_of_analysis_name
        self.tmp_dir_name.mkdir(exist_ok=True)  # Create if doesn't exist

        # Create DataFrame file path
        df_file_path = (
            self.tmp_dir_name
            / f"{self.index_analysis}_{self.area_of_analysis_name}_dataframe.csv"
        )

        # Save DataFrame if it doesn't exist
        if not df_file_path.exists():
            df_raster_inventory.to_csv(df_file_path, index=False)

        # Create temporary processing directory
        self.processing_raster_dir = self.tmp_dir_name / "temporary_files"
        self.processing_raster_dir.mkdir(exist_ok=True)  # Create if doesn't exist

        return df_file_path

    def _check_preexisting_files(
        self, i: int, df_file_path: Path, df_raster: pd.DataFrame
    ) -> bool:
        """
        Check if raster file already exists for current month/year.

        Args:
            i: Row index in the raster inventory DataFrame
            df_file_path: Path to the inventory CSV file
            df_raster: Current state of the raster inventory DataFrame

        Returns:
            True if processing should be skipped (file exists or no data available)
            False if processing should proceed

        Side Effects:
            Sets self.month_ and self.year_ attributes
            Updates inventory file if pre-existing raster found
        """
        # Extract month and year
        self.month_ = df_raster.loc[df_raster.index == i].month.values[0]
        self.year_ = df_raster.loc[df_raster.index == i].year.values[0]

        # Create raster file path
        raster_filename = f"{self.area_of_analysis_name}_{self.index_analysis}_{self.month_}_{self.year_}.tif"
        raster_path = self.tmp_dir_name / raster_filename

        # Check if raster already exists
        if raster_path.exists():
            df_raster.loc[i, "data_id"] = 1
            df_raster.to_csv(df_file_path, index=False)
            log(
                f" - {self.area_of_analysis_name} - Raster for {self.month_}/{self.year_} already downloaded. Skipping to next month."
            )
            return True

        # Check if month is available
        if df_raster.iloc[i].data_id == 0:
            if df_raster.iloc[i].download_method != "could_not_process":
                df_raster.loc[i, "download_method"] = "no_links_available"
                df_raster.to_csv(df_file_path, index=False)
            log(
                f" - {self.area_of_analysis_name} - Raster for {self.month_}/{self.year_} not available. Skipping to next month."
            )
            return True

        return False

    def raster_download_date_preprocessing(self) -> None:
        """
        Prepare date filtering for current month's processing.

        Sets up date arrays and time ranges for downloading data
        specific to the current month being processed.

        Sets:
            Sets self.dates_ordered attribute with filtered dates
        """

        log(f"\n Starting new analysis for {self.month_}/{self.year_}")

        # gather links for raster images
        sample_date = datetime(self.year_, self.month_, 1)
        first_day = sample_date + relativedelta(day=1)
        last_day = sample_date + relativedelta(day=31)

        # creates time range for a specific month
        time_of_interest = [
            f"{self.year_}-{self.month_:02d}-{first_day.day:02d}/{self.year_}"
            + f"-{self.month_:02d}-{last_day.day:02d}"
        ]

        # dates according to cloud coverage
        date_order = [
            True if (d.month == self.month_) and (d.year == self.year_) else False
            for d in self.date_list
        ]
        date_array = np.array(self.date_list)
        date_filter = np.array(date_order)
        dates_ordered = date_array[date_filter]

        self.dates_ordered = dates_ordered
        self.time_of_interest = time_of_interest

    def save_output_raster(
        self,
        single_raster_array: np.ndarray,
        output_raster_path: Path,
        output_meta: Dict,
    ) -> None:
        """
        Save raster array to file with metadata.

        Args:
            single_raster_array: Raster data array to save
            output_raster_path: Path where the raster file will be saved
            output_meta: Rasterio metadata dictionary for the output file
        """

        with rasterio.open(output_raster_path, "w", **output_meta) as dest:
            dest.write(single_raster_array)

            dest.close()

    def _raster_nan_test(self, gdf, raster_file):
        """
        The function performs a test to check for Not-a-Number values in a raster file.

        Arguments:
            gdf (geopandas.GeoDataFrame): Pass in the geodataframe containing geometries that we want to check for nan values
            raster_file (rasterio.io.DatasetReader): Specify the raster file to be used in the function

        Raises:
            An exception if needed.
        """

        gdf["test"] = gdf.geometry.apply(
            lambda geom: clean_mask(geom, raster_file, outside_value=np.nan)
        ).apply(np.ma.mean)

        log(f"There are {gdf.test.isna().sum()} null data values")

        if gdf["test"].isna().sum() > 0:
            raise NanValues("NaN values are still present after processing")

    def raster_interpolation(self, df_raster_inventory: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing raster data using temporal interpolation.

        Fills gaps in missing monthly data using three interpolation strategies:
        - Case 1: Missing data at beginning - copy from next available
        - Case 2: Missing data at end - copy from previous available
        - Case 3: Missing data in middle - linear interpolation between endpoints

        Args:
            df_raster_inventory: DataFrame tracking data availability

        Returns:
            Updated DataFrame with interpolation status

        Side Effects:
            Creates interpolated raster files for missing months
            Updates interpolate column in tracking DataFrame
        """

        log(
            f"Interpolating {len(df_raster_inventory.loc[df_raster_inventory.data_id == 0])}"
        )

        self._check_available_data(df_raster_inventory)

        df_raster_inventory["interpolate"] = 0

        for row in range(len(df_raster_inventory)):
            if df_raster_inventory.iloc[row].data_id == 0:
                # Set starting row to previus row (Unless it is the first row)
                start = row - 1
                if start == -1:
                    start = 0

                # Try setting finish row to the first ocurrance (.idmax()) of all following rows ([row:,:])
                # where data is not equal (ne) to cero (rows with downloaded images).
                # Meaning, the next row with a downloaded image.
                try:
                    finish = df_raster_inventory.iloc[row:, :].data_id.ne(0).idxmax()
                # Except (Zero next rows with a downloaded image), finish row is last row.
                except:
                    finish = len(df_raster_inventory)

                log(f"Row start:{start} - row finish:{finish}")

                df_subset = df_raster_inventory.iloc[start : finish + 1]
                if df_subset.loc[df_subset.index == start].data_id.values[0] == 0:
                    log("Entering missing data case 1 - first value missing")

                    case = "Case-1"

                    available_raster, meta = self._read_available_raster(
                        df_subset, start, finish, case
                    )

                    log("Read last raster data")

                    cont = 0
                    while df_subset.iloc[cont].data_id == 0:
                        df_raster_inventory = self.create_missing_raster(
                            df_raster_inventory,
                            df_subset,
                            start,
                            cont,
                            available_raster,
                            meta,
                        )

                        cont += 1

                elif df_subset.loc[df_subset.index == finish].data_id.values[0] == 0:
                    log("Entering missing data case 2 - last value missing")

                    case = "Case-2"

                    available_raster, meta = self._read_available_raster(
                        df_subset, start, finish, case
                    )

                    log("Read first raster data")

                    cont = 1
                    while cont < len(df_subset):
                        df_raster_inventory = self.create_missing_raster(
                            df_raster_inventory,
                            df_subset,
                            start,
                            cont,
                            available_raster,
                            meta,
                        )

                        cont += 1

                else:
                    log("Entering missing data case 3  - mid point missing")

                    case = "Case-3"

                    available_raster_start, meta = self._read_available_raster(
                        df_subset, start, finish, case + "-start"
                    )

                    available_raster_finish, _ = self._read_available_raster(
                        df_subset, start, finish, case + "-finish"
                    )

                    missing_len = len(df_subset.loc[df_subset.data_id == 0])
                    slope = 1 / (missing_len + 1)
                    slope_increment = slope

                    cont = 1

                    log(f"Preparing data for interpolation with slope {slope}")

                    # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
                    arr = np.r_["0,3", available_raster_start, available_raster_finish]
                    # define the grid coordinates where you want to interpolate
                    dim_row = available_raster_start.shape[1]
                    dim_col = available_raster_start.shape[2]

                    X, Y = np.meshgrid(np.arange(dim_row), np.arange(dim_col))

                    while round(slope, 4) < 1:
                        log(
                            f"Starting interpolation for iteration {cont} with position {slope}"
                        )
                        # switch order of col and row
                        coordinates = np.ones((dim_col, dim_row)) * slope, X, Y

                        inter_raster = ndimage.map_coordinates(
                            arr, coordinates, order=1
                        ).T
                        inter_raster = inter_raster.reshape(
                            (1, inter_raster.shape[0], inter_raster.shape[1])
                        )

                        df_raster_inventory = self.create_missing_raster(
                            df_raster_inventory,
                            df_subset,
                            start,
                            cont,
                            inter_raster,
                            meta,
                        )

                        cont += 1
                        slope = slope + slope_increment

        df_file_dir = (
            self.tmp_dir_name
            / f"{self.index_analysis}_{self.area_of_analysis_name}_dataframe.csv"
        )
        df_raster_inventory.to_csv(df_file_dir, index=False)

        return df_raster_inventory

    def _read_available_raster(
        self, df_subset: pd.DataFrame, start: int, finish: int, case: str
    ) -> np.ndarray:
        """
        Read existing raster data for interpolation reference.

        Args:
            df_subset: Subset of inventory DataFrame for current interpolation window
            start: Starting index in the subset
            finish: Ending index in the subset
            case: Interpolation case identifier ("Case-1", "Case-2", "Case-3-start", "Case-3-finish")

        Returns:
            Raster array data from the reference file
        """

        available_raster_id = None

        if case == "Case-1":
            available_raster_id = finish
        elif case == "Case-2":
            available_raster_id = start
        elif case == "Case-3-start":
            available_raster_id = start
        elif case == "Case-3-finish":
            available_raster_id = finish

        available_month = df_subset.loc[
            df_subset.index == available_raster_id
        ].month.values[0]
        available_year = df_subset.loc[
            df_subset.index == available_raster_id
        ].year.values[0]
        raster_path = (
            self.tmp_dir_name
            / f"{self.area_of_analysis_name}_{self.index_analysis}_{available_month}_{available_year}.tif"
        )
        raster_file = rasterio.open(raster_path)
        meta = raster_file.meta
        available_raster = raster_file.read()

        return available_raster, meta

    def create_missing_raster(
        self,
        df_raster_inventory: pd.DataFrame,
        df_subset: pd.DataFrame,
        start: int,
        cont: int,
        available_raster: np.ndarray,
        meta: dict,
    ) -> pd.DataFrame:
        """
        Create interpolated raster file for missing month.

        Args:
            df_raster_inventory: Full inventory DataFrame
            df_subset: Subset DataFrame for current interpolation window
            start: Starting index for the interpolation window
            cont: Current position in the interpolation sequence
            available_raster: Reference raster data for interpolation

        Returns:
            Updated inventory DataFrame with interpolation status

        Side Effects:
            Creates new raster file for the missing month
            Updates data_id and interpolate columns
        """

        missing_month = int(df_subset.iloc[cont].month)
        missing_year = int(df_subset.iloc[cont].year)
        raster_path = (
            self.tmp_dir_name
            / f"{self.area_of_analysis_name}_{self.index_analysis}_{missing_month}_{missing_year}.tif"
        )

        with rasterio.open(raster_path, "w", **meta) as dest:
            dest.write(available_raster)

            dest.close()

            log("Finished creating raster")
            df_raster_inventory.loc[
                df_raster_inventory.index == start + cont, "data_id"
            ] = 1
            df_raster_inventory.loc[
                df_raster_inventory.index == start + cont, "interpolate"
            ] = 1

        return df_raster_inventory

    def __repr__(self):
        """Provide a helpful string representation of the class."""
        return (
            f"PCRasterData(index='{self.index_analysis}', "
            f"area='{self.area_of_analysis_name}', "
            f"satellite='{self.satellite}', "
            f"date_range='{self.start_date}' to '{self.end_date}')"
        )

    def list_output_rasters(self) -> List[Path]:
        """
        List all output raster files.
        """
        if not self.tmp_dir_name or not self.tmp_dir_name.exists():
            return []

        # Find all .tif files matching our pattern
        pattern = f"{self.area_of_analysis_name}_{self.index_analysis}_*.tif"
        return list(self.tmp_dir_name.glob(pattern))


def clean_mask(geom, dataset="", outside_value=0, **mask_kw):
    """
    The mask in this function is used to extract the values from a raster dataset that fall
    within a given geometry of interest.

    Arguments:
        geom (geometry): Geometric figure that will be used to mask the raster dataset.
        dataset (rasterio DatasetReader): The raster dataset that will be masked by the
        inputted geometry. If no value is provided, then it defaults to an empty string
        and returns only the masked array of values from within the inputted geometry
        without any metadata.
        outside_value (float | np.nan): Value assigned when the geometry falls completely outside the raster extent.
                                        Defaults to 0. Use np.nan to flag these cases as errors.
        mask_kw (dict): A dictionary of arguments passed to create the mask.

    Returns:
        masked (np.array): Returns values from within the inputted geometry.
    """

    mask_kw.setdefault("crop", True)
    mask_kw.setdefault("all_touched", True)
    mask_kw.setdefault("filled", False)
    try:
        masked, _ = rasterio.mask.mask(dataset=dataset, shapes=(geom,), **mask_kw)
    except:
        masked = np.array([outside_value])

    return masked


class RasterToPolygon:
    """
    A class for extracting and analyzing raster data within polygon boundaries.

    This class processes raster time series data by extracting values that fall within
    specified polygons, calculating various statistics, and providing both summary
    statistics and detailed time series data.

    Attributes:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing polygon geometries
        feature_unique_id (str): Column name for unique polygon identifiers
        df_raster_inventory (pd.DataFrame): DataFrame with raster file inventory (year, month columns)
        index_analysis (str): Column name/identifier for the raster analysis index
        area_of_analysis_name (str): Name identifier for the analysis area
        tmp_dir (Path): Directory path where raster files are stored
        max_workers (int): Maximum number of worker processes for parallel computation. Defaults to 8
        max_processing_func_time (int): Timeout in seconds for individual processing tasks. Defaults to 300
    """

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        feature_unique_id: str,
        df_raster_inventory: pd.DataFrame,
        index_analysis: str,
        area_of_analysis_name: str,
        tmp_dir: Union[str, Path],
        **kwargs,
    ) -> None:
        """
        Initialize the RasterToPolygon processor.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing polygon geometries for analysis
            feature_unique_id (str): Column name that uniquely identifies each polygon
            df_raster_inventory (pd.DataFrame): DataFrame containing raster file metadata
                                                Must have 'year' and 'month' columns
            index_analysis (str): Name/identifier of the raster index being analyzed
                                (used in raster filenames)
            area_of_analysis_name (str): Descriptive name for the analysis area
            tmp_dir (Union[str, Path]): Path to directory containing raster files
            **kwargs: Optional parameters:
                - max_workers (int): Maximum worker processes (default: 8)
                - max_processing_func_time (int): Task timeout in seconds (default: 300)

        Raises:
            ValueError: If required columns are missing from input DataFrames
            FileNotFoundError: If tmp_dir does not exist
        """

        self.gdf = gdf
        self.feature_unique_id = feature_unique_id
        self.df_raster_inventory = df_raster_inventory
        self.index_analysis = index_analysis
        self.area_of_analysis_name = area_of_analysis_name
        self.tmp_dir = tmp_dir
        self.tmp_dir_name = f"{self.tmp_dir}/{self.area_of_analysis_name}"

        # === DEFAULT VALUES FOR OPTIONAL PARAMETERS ===
        defaults = {"max_workers": 8, "max_processing_func_time": 300}

        # === SET OPTIONAL PARAMETERS ===
        self.max_workers: int = kwargs.get("max_workers", defaults["max_workers"])
        self.max_processing_func_time: int = kwargs.get(
            "max_processing_func_time", defaults["max_processing_func_time"]
        )

    def raster_summary(self) -> Tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Generate raster statistics summary for all polygons.

        This method integrates the complete analysis pipeline, extracting raster values
        for each polygon, calculating various statistical measures, and formatting the
        results into two complementary outputs.

        Returns:
            Tuple[gpd.GeoDataFrame, pd.DataFrame]: A tuple containing:
                - gdf_raster_analysis: GeoDataFrame with summary statistics per polygon
                    including mean, std, median, min, max, trend, and yearly values
                - gdf_raster_df: DataFrame with all individual raster values assigned
                    to their respective polygons (no geometry, includes temporal data)

        Raises:
            RuntimeError: If multiprocessing fails or no valid results are obtained
        """

        gdf_raster = self.raster_to_polygon_multiprocess()

        log("Assigned raster data to polygons")

        # gdf to store summary statistics
        gdf_raster_analysis = (
            self.gdf[[self.feature_unique_id, "geometry"]].drop_duplicates().copy()
        )

        # calculate min,max
        df_raster_minmax = self.calculate_raster_min_max(gdf_raster)
        log("Calculated min and max")

        # calculate centrality, disperssion and tendency
        df_group_data = self.calculate_raster_central_dispersion_tendency(
            gdf_raster, df_raster_minmax
        )
        log("Calculated centrality, dispersion and tendency")

        # format output
        gdf_raster_analysis = self.format_raster_output(
            gdf_raster_analysis, df_group_data
        )
        log("Formatted gdf output with raster values")

        # create yearly data column
        gdf_raster_analysis = self.calculate_raster_yearly_data(
            gdf_raster, gdf_raster_analysis
        )
        log("Created yearly data column")

        # remove geometry information
        gdf_raster_df = gdf_raster.drop(columns=["geometry"])

        # add city information
        gdf_raster_analysis["area_of_analysis_name"] = self.area_of_analysis_name
        gdf_raster_df["area_of_analysis_name"] = self.area_of_analysis_name

        log(f"df nan values: {gdf_raster_df[self.index_analysis].isna().sum()}")

        # turn all columns to lowercase
        gdf_raster_analysis.columns = map(str.lower, gdf_raster_analysis.columns)
        gdf_raster_df.columns = map(str.lower, gdf_raster_df.columns)

        return gdf_raster_analysis, gdf_raster_df

    def raster_to_polygon_multiprocess(self) -> pd.DataFrame:
        """
        Extract raster values for polygons using parallel processing.

        This method coordinates multiprocessing across years and months, distributing
        the computational load while maintaining data integrity and providing progress feedback.

        Returns:
            pd.DataFrame: Combined DataFrame containing extracted raster values
                            for all polygons across all time periods
        """

        # create empty geodataframe to save raster data
        gdf_raster = gpd.GeoDataFrame()

        years_list = list(self.df_raster_inventory.year.unique())
        all_results = []  # collect results before concatenating

        # determine the concurrent number of workers
        self.num_workers = min(
            mp.cpu_count() - 1, self.max_workers
        )  # Leave one CPU free, set max_workers

        log(f"Processing {len(years_list)} years using {self.num_workers} workers")

        for year in tqdm(years_list, desc="Processing years", position=0):
            # Prepare input data for this year
            input_list = [
                [year, month]
                for month in self.df_raster_inventory.loc[
                    self.df_raster_inventory.year == year
                ].month.unique()
            ]

            if not input_list:
                log(f"Warning: No months found for year {year}")
                continue

            # Process year with multiprocessing
            year_results = self._process_year_parallel(input_list)

            if year_results:
                all_results.extend(year_results)

        # Concatenate all results at once (more efficient)
        if all_results:
            gdf_raster = pd.concat(all_results, ignore_index=True, axis=0)
        else:
            log("Warning: No results obtained from processing")
            gdf_raster = pd.DataFrame()  # Return empty DataFrame

        return gdf_raster

    def _process_year_parallel(self, input_list: List[List[int]]) -> List[pd.DataFrame]:
        """
        Process monthly raster data for a single year using multiprocessing.

        Args:
            input_list (List[List[int]]): List of [year, month] pairs to process

        Returns:
            List[pd.DataFrame]: List of processed DataFrames for each month
        """

        year_results = []
        pool = None

        try:
            # Create pool with context manager for proper cleanup
            with Pool(processes=self.num_workers) as pool:
                # Submit all jobs
                async_results = [
                    pool.apply_async(self._wrap_mask_by_polygon, args)
                    for args in input_list
                ]

                # Collect results with progress bar
                for async_result in tqdm(
                    async_results,
                    desc=f"Processing months for year {input_list[0][0]}",
                    position=1,
                    leave=False,
                ):
                    try:
                        result = async_result.get(
                            timeout=self.max_processing_func_time
                        )  # 5 minute timeout per task
                        if result is not None and not result.empty:
                            year_results.append(result)
                    except Exception as e:
                        log(f"Error processing task: {e}")
                        continue

        except Exception as e:
            log(f"Error in multiprocessing pool: {e}")
            if pool is not None:
                pool.terminate()
                pool.join()

        return year_results

    def _wrap_mask_by_polygon(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Error-handling wrapper for the mask_by_polygon method.

        Args:
            year (int): Year of the raster data to process
            month (int): Month of the raster data to process

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if processing failed
        """
        try:
            return self.mask_by_polygon(year, month)
        except Exception as e:
            log(f"Error processing year {year}, month {month}: {e}")
            return None

    def mask_by_polygon(self, year: int, month: int) -> pd.DataFrame:
        """
        Extract raster values within polygon boundaries for a specific time period.

        This method opens a raster file, reprojects geometries to match the raster CRS,
        applies a masking function to extract values within each polygon, and returns
        the processed data with temporal metadata.

        Args:
            year (int): Year identifier for the raster file
            month (int): Month identifier for the raster file

        Returns:
            pd.DataFrame: DataFrame containing polygon IDs, extracted raster values,
                            and temporal information (year, month)
        """

        gdf_raster = self.gdf.copy()
        # read raster file
        raster_file_dir = f"{self.tmp_dir_name}/{self.area_of_analysis_name}_{self.index_analysis}_{month}_{year}.tif"

        with rasterio.open(raster_file_dir) as raster_file:
            gdf_raster = gdf_raster.to_crs(raster_file.crs)
            # Using the apply function to apply the clean_mask function to the geometry column of the geodataframe
            try:
                gdf_raster[self.index_analysis] = gdf_raster.geometry.apply(
                    lambda geom: clean_mask(geom, raster_file)
                ).apply(np.ma.mean)
            except:
                log(f"Error processing year {year}, month {month}: {str(e)}")
                gdf_raster[self.index_analysis] = np.nan
            # adds month and year columns to the hex_raster geodataframe
            gdf_raster["month"] = month
            gdf_raster["year"] = year

            gdf_raster = gdf_raster.to_crs("EPSG:4326")

            return gdf_raster

    def calculate_raster_min_max(self, gdf_raster: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temporal minimum and maximum values for each polygon.

        This method computes min/max values across years for each polygon,
        providing insights into the temporal range of the analyzed index.

        Args:
            gdf_raster (pd.DataFrame): Input DataFrame with temporal raster data

        Returns:
            pd.DataFrame: DataFrame with min/max statistics per polygon
        """

        df_raster_minmax = (
            gdf_raster[[self.feature_unique_id, self.index_analysis, "year"]]
            .groupby([self.feature_unique_id, "year"])
            .agg(["max", "min"], numeric_only=True)
        )

        df_raster_minmax.columns = ["_".join(col) for col in df_raster_minmax.columns]
        df_raster_minmax = df_raster_minmax.reset_index()
        df_raster_minmax = (
            df_raster_minmax[
                [
                    self.feature_unique_id,
                    f"{self.index_analysis}_max",
                    f"{self.index_analysis}_min",
                ]
            ]
            .groupby([self.feature_unique_id])
            .mean(numeric_only=True)
        )
        df_raster_minmax = df_raster_minmax.reset_index()

        return df_raster_minmax

    def calculate_raster_central_dispersion_tendency(
        self, gdf_raster: pd.DataFrame, df_raster_minmax: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate central tendency, dispersion, and temporal trend statistics.

        This method computes comprehensive statistical measures including measures
        of central tendency (mean, median), dispersion (standard deviation), and
        temporal trends using Mann-Kendall analysis.

        Args:
            gdf_raster (pd.DataFrame): Input DataFrame with raster time series data

        Returns:
            pd.DataFrame: DataFrame with statistical measures per polygon
        """
        df_group_data = (
            gdf_raster[[self.feature_unique_id, self.index_analysis]]
            .groupby(self.feature_unique_id)
            .agg(["mean", "std", "median", mk.sens_slope])
        )
        df_group_data.columns = ["_".join(col) for col in df_group_data.columns]
        df_group_data = df_group_data.reset_index().merge(
            df_raster_minmax, on=self.feature_unique_id
        )

        return df_group_data

    def format_raster_output(
        self, gdf_raster_analysis: gpd.GeoDataFrame, df_group_data: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """
        Format and structure the final analysis output.

        This method merges statistical data with geometries, creates derived
        metrics, and structures the output for easy interpretation and use.

        Args:
            gdf_raster_analysis (gpd.GeoDataFrame): Base GeoDataFrame with geometries
            df_group_data (pd.DataFrame): Statistical data to merge

        Returns:
            gpd.GeoDataFrame: Formatted GeoDataFrame with all analysis results
        """

        gdf_raster_analysis = gdf_raster_analysis.merge(
            df_group_data, on=self.feature_unique_id
        )
        gdf_raster_analysis[self.index_analysis + "_diff"] = (
            gdf_raster_analysis[self.index_analysis + "_max"]
            - gdf_raster_analysis[self.index_analysis + "_min"]
        )
        gdf_raster_analysis[self.index_analysis + "_tend"] = gdf_raster_analysis[
            f"{self.index_analysis}_sens_slope"
        ].apply(lambda x: x[0])
        gdf_raster_analysis = gdf_raster_analysis.drop(
            columns=[f"{self.index_analysis}_sens_slope"]
        )

        return gdf_raster_analysis

    def calculate_raster_yearly_data(
        self, gdf_raster: pd.DataFrame, gdf_raster_analysis: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Add yearly average values as separate columns to the analysis DataFrame.

        This method creates individual columns for each year's average value,
        enabling time series analysis and visualization of temporal patterns.

        Args:
            gdf_raster (pd.DataFrame): Source data with temporal information
            gdf_raster_analysis (gpd.GeoDataFrame): Target DataFrame for yearly data

        Returns:
            gpd.GeoDataFrame: Enhanced DataFrame with yearly value columns
        """

        for y in gdf_raster["year"].unique():
            gdf_tmp = (
                gdf_raster.loc[gdf_raster.year == y]
                .groupby(self.feature_unique_id)
                .agg({self.index_analysis: "mean"})
                .reset_index()
            )
            gdf_tmp = gdf_tmp.rename(
                columns={self.index_analysis: f"{self.index_analysis}_{y}"}
            )
            gdf_tmp = gdf_tmp[
                [self.feature_unique_id, f"{self.index_analysis}_{y}"]
            ].copy()
            gdf_raster_analysis = gdf_raster_analysis.merge(
                gdf_tmp, on=self.feature_unique_id, how="left"
            )
            del gdf_tmp

        return gdf_raster_analysis
