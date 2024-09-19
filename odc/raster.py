################################################################################
# Module: Raster
# Set of raster data treatment and analysis functions, mainly using Planetary Computer
# updated: 13/11/2023
################################################################################

from .utils import *
from .data import *
from tqdm import tqdm
import rasterio
from rasterio import windows
from rasterio import features
from rasterio import warp
import rasterio.mask
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.fill import fillnodata
from pystac.extensions.eo import EOExtension as eo
from datetime import datetime
from dateutil.relativedelta import relativedelta
import planetary_computer as pc
from scipy import stats as st
from rasterio.merge import merge
import pandas as pd
import numpy as np
from func_timeout import func_timeout
import pymannkendall as mk
import scipy.ndimage as ndimage
from pystac_client import Client
from multiprocessing import Pool

# Flags to ignore division by zero and invalid floating point operations
np.seterr(divide='ignore', invalid='ignore')

# A class is created it receives a message and has a function that returns it as a string
class AvailableData(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class NanValues(Exception):
    def __init__(self, message):
        self.message = message

def available_data_check(df_len, missing_months, pct_limit=50, window_limit=6):
    pct_missing = round(missing_months/len(df_len),2)*100
    log(f'Created DataFrame with {missing_months} ({pct_missing}%) missing months')
    if pct_missing >= pct_limit: 
        raise AvailableData('Missing more than 50 percent of data points')
    df_rol = df_len.rolling(window_limit).sum()
    if len(df_rol.loc[df_rol.data_id==0])>0:
        raise AvailableData('Multiple missing months together')
    del df_rol

def download_raster_from_pc(gdf, index_analysis, city, freq, start_date, end_date, 
                            tmp_dir, band_name_dict, query={}, satellite="sentinel-2-l2a",
                            projection_crs="EPSG:6372", compute_unavailable_dates=True):
    """
    Function that returns a raster with the data provided.
    Arguments:
        gdf (geopandas.GeoDataFrame): Area of interest
        index_analysis (str): Index of analysis
        city (str): City name
        freq (str): Frequency of raster analysis
        start_date (date): First date of raster data
        end_date (date): Last date of raster data
        tmp_dir (str): address of temporary directory where downloaded and processed 
        raster for a specific city will be saved. 
        band_name_list (list): List with multispectral band names for raster analysis
        satellite (str): satellite used to download imagery
        projection_crs (str): projection to be used when needed. Defaults to "EPSG:6372".
        compute_unavailable_dates (bool): Whether or not to consider unavailable dates (Raises errors when too many unavailable). Defaults to True.

    Raises:
        AvailableData: Object with a message

    Returns:
        df_len (pandas.DataFrame): Dataframe containing a summary of available and 
        processed data for city and the specified time range.
    """
    # Create area of interest coordinates from hexagons to download raster data    
    log('Extracting bounding coordinates from hexagons')
    # Create buffer around hexagons
    poly = gdf.to_crs(projection_crs).buffer(500)
    poly = poly.to_crs("EPSG:4326")
    poly = gpd.GeoDataFrame(geometry=poly).dissolve().geometry
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

    # Create time of interest (Creates a list for all to-be-analysed-months with structure [start_day/end_day,(...)])
    log('Defining time of interest')
    time_of_interest = create_time_of_interest(start_date, end_date, freq=freq)
    # Gather items for time and area of interest (Creates of list of available image items)
    log('Gathering items for time and area of interest')
    items = gather_items(time_of_interest, area_of_interest, query=query, satellite=satellite)
    log(f'Fetched {len(items)} items')

    log('Checking available tiles for area of interest')
    # df_clouds, date_list = arrange_items(items, satellite=satellite)
    date_list = available_datasets(items, satellite, query)
    # log(f"{len(date_list)} dates available with avg {round(df_clouds['avg_cloud'].mean(),2)}% clouds.")

    # Create dictionary from links (assets_hrefs is a dict. of dates and links with structure {available_date:{band_n:[link]}})
    band_name_list = list(band_name_dict.keys())[:-1]
    assets_hrefs = link_dict(band_name_list, items, date_list)
    log('Created dictionary from items')

    # Analyze available data according to raster properties (Creates df_len for the first time)
    df_len, missing_months = df_date_links(assets_hrefs, start_date, end_date, 
                                           band_name_list, freq)
    
    # Test for missing months, raises errors
    if compute_unavailable_dates:
        available_data_check(df_len, missing_months)

    # Raster cropping with bounding box from earlier
    bounding_box = gpd.GeoDataFrame(geometry=poly).envelope
    gdf_bb = gpd.GeoDataFrame(gpd.GeoSeries(bounding_box), columns=['geometry'])
    log('Created bounding box for raster cropping')

    # Create GeoDataFrame to test nan values in raster
    gdf_raster_test = gdf.to_crs(projection_crs).buffer(1)
    gdf_raster_test = gdf_raster_test.to_crs("EPSG:4326")
    gdf_raster_test = gpd.GeoDataFrame(geometry=gdf_raster_test).dissolve()

    # Raster creation - Download raster data by month
    log('Starting raster creation for specified time')
    df_len = create_raster_by_month(
        df_len, index_analysis, city, tmp_dir,
        band_name_dict,date_list, gdf_raster_test,
        gdf_bb, area_of_interest, satellite, query=query,
        compute_unavailable_dates=compute_unavailable_dates)
    log('Finished raster creation')

    # Calculate percentage of missing months
    missing_months = len(df_len.loc[df_len.data_id==0])
    log(f'Updated missing months to {missing_months} ({round(missing_months/len(df_len),2)*100}%)')
    
    if compute_unavailable_dates:
        # Starts raster interpolation by predicting points from existing values and updates missing months percentage
        log('Starting raster interpolation')
        df_len = raster_interpolation(df_len, city, tmp_dir, index_analysis)
        log('Finished raster interpolation')
        missing_months = len(df_len.loc[df_len.data_id==0])
        log(f'Updated missing months to {missing_months} ({round(missing_months/len(df_len),2)*100}%)')
    
    # returns final raster
    return df_len


def create_time_of_interest(start_date, end_date, freq='MS'):
    """
    Creates a time range used to download raster data going from start_date to 
    end_date and a specified frequency

    Arguments:
        start_date (date): First date of raster data
        end_date (date): Last date of raster data
        freq (str):Frequency of time range between start_date and end_date 

    Returns:
        time_of_interest (list): date range in specified format used by Planetary Computer api
    """
    df_tmp_dates = pd.DataFrame() # temporary date dataframe
    df_tmp_dates['date'] = pd.date_range(start = start_date,   
                                end = end_date, 
                                freq = freq)
    # extract month and year from date
    df_tmp_dates['month'] = df_tmp_dates.apply(lambda row: row['date'].month, axis=1)
    df_tmp_dates['year'] = df_tmp_dates.apply(lambda row: row['date'].year, axis=1)

    time_of_interest = []
    
    # create a time range by month
    for d in range(len(df_tmp_dates)):
        
        month = df_tmp_dates.loc[df_tmp_dates.index==d].month.values[0]
        year = df_tmp_dates.loc[df_tmp_dates.index==d].year.values[0]

        sample_date = datetime(year, month, 1)
        first_day = sample_date + relativedelta(day=1) # first day of the month
        last_day = sample_date + relativedelta(day=31) # last day of the month

        # append time range to time of interest list with planetary computer format
        time_of_interest.append(f"{year}-{month:02d}-{first_day.day:02d}/{year}"+
                                f"-{month:02d}-{last_day.day:02d}")
        
    # Returns array with time of interest
    return time_of_interest


def gather_items(time_of_interest, area_of_interest, query={}, satellite="sentinel-2-l2a"):
    """ 
    Items gathered in time and area of interest from planetary computer.

    Arguments:
        time_of_interest (list): Time range of interest
        area_of_interest (dict): Polygon, area of interest
        satellite (str): satellite used to download imagery

    Returns:
        items (np.array): items intersecting time and area of interest
    """
    # gather items from planetary computer by date and area of interest
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    items = []

    log(f'Gathering items for {satellite} with query: {query}')

    for t in time_of_interest:
        try:
            search = catalog.search(
                collections=[satellite],
                intersects=area_of_interest,
                datetime=t,
                query = query
            )

            # Check how many items were returned
            items.extend(list(search.items()))
        except:
            log('No items found')
            continue
    return items

def find_asset_by_band_common_name(item, common_name):
    """
    Filter that receives an item from a list and searches for a band with a common name.

    Arguments:
        item (object): Belongs to the gathered items
        common_name (str): Common name of the band  to be searched

    Raises:
        KeyError: If common_name is not found

    Returns:
        asset (str) : Asset with the common name
    """
    # gather links for each band
    for asset in item.assets.values():
        asset_bands = eo.ext(asset).bands
        if asset_bands and asset_bands[0].common_name == common_name:
            return asset
    raise KeyError(f"{common_name} band not found")


def link_dict(band_name_list, items, date_list):
    """
    The function creates a dictionary with the links to the assets.

    Arguments:
        band_name_list (list): list with multispectral band names for raster analysis
        items (list): items intersecting time and area of interest
        date_list (list): dates with available data

    Returns:
        assets_hrefs (dict): Dictionary with the links to the assets
    """
    assets_hrefs = {}

    for i in items:
        # only takes into account dates that are in the date list
        if i.datetime.date() not in date_list:
            continue
        # if date already in dictionary, append link to list
        if i.datetime.date() in list(assets_hrefs.keys()):
            for b in band_name_list:
                assets_hrefs[i.datetime.date()][b].append(pc.sign(find_asset_by_band_common_name(i,b).href))
        # if date not in dictionary, create new dictionary entry
        else:
            assets_hrefs[i.datetime.date()] = {}
            for b in band_name_list:
                assets_hrefs[i.datetime.date()].update({b:[]})
                assets_hrefs[i.datetime.date()][b].append(pc.sign(find_asset_by_band_common_name(i,b).href))
                
    return assets_hrefs

def df_date_links(assets_hrefs, start_date, end_date, band_name_list, freq='MS'):
    """
    The Function converts the dictionary of assets into a dataframe that contains date, month and year. 
    It merges the month and year, to then remove the date and extract the data_id 
    and remove columns containing band names. Aditionally it uses a function to count the missing months.

    Arguments:
        assets_hrefs (dict): Dictionary with the links to the assets
        start_date (date): First date of raster data
        end_date (date): Last date of raster data
        band_name_list (list): list with multispectral band names for raster analysis
        freq (str): Frequency of time range between start_date and end_date

    Returns:
        df_complete_dates (pandas.DataFrame): Dataframe with filtered dates
        missing_months (int): Number of missing months
    """
    # dictionary to dataframe
    df_dates = pd.DataFrame.from_dict(assets_hrefs, orient='Index').reset_index().rename(columns={'index':'date'})
    df_dates['date'] = pd.to_datetime(df_dates['date']).dt.date
    df_dates['year'] = df_dates.apply(lambda row: row['date'].year, axis=1)
    df_dates['month'] = df_dates.apply(lambda row: row['date'].month, axis=1)
    
    df_dates_filtered = pd.DataFrame()
    
    # keep only one data point by month
    for y in df_dates['year'].unique():
        for m in df_dates.loc[df_dates['year']==y,'month'].unique():
            df_dates_filtered = pd.concat([df_dates_filtered,
                                         df_dates.loc[(df_dates['year']==y)&
                                                      (df_dates['month']==m)].sample(1)],
                                          ignore_index=True)
    
    # create full range time dataframe
    df_tmp_dates = pd.DataFrame() # temporary date dataframe
    df_tmp_dates['date'] = pd.date_range(start = start_date,   
                               end = end_date, 
                               freq = freq) # create date range
    # extract year and month from date
    df_tmp_dates['year'] = df_tmp_dates.apply(lambda row: row['date'].year, axis=1)
    df_tmp_dates['month'] = df_tmp_dates.apply(lambda row: row['date'].month, axis=1)

    # remove date column for merge
    df_tmp_dates.drop(columns=['date'], inplace=True)

    df_complete_dates = df_tmp_dates.merge(df_dates_filtered, left_on=['year','month'],
                                          right_on=['year','month'], how='left')

    # remove date 
    df_complete_dates.drop(columns='date', inplace=True)
    df_complete_dates.sort_values(by=['year','month'], inplace=True)
    
    # create binary column for available (1) or missing data (0)
    idx = df_complete_dates[band_name_list[0]].isna()
    df_complete_dates['data_id'] = 0
    df_complete_dates.loc[~idx,'data_id'] = 1
    
    df_complete_dates.drop(columns=band_name_list, inplace=True)

    # calculate missing months
    missing_months = len(df_complete_dates.loc[df_complete_dates.data_id==0])
    
    # create empty able_to_download to avoid crash
    df_complete_dates['able_to_download'] = np.nan
    
    return df_complete_dates, missing_months


def arrange_items(items, satellite="sentinel-2-l2a"):
    """
    Previously function available_datasets. 
    Filters for dates where all tiles (rasters) that compose the area of interest are available.

    Arguments:
        items (np.array): items intersecting time and area of interest
        satellite (str): satellite used to download imagery

    Returns:
        date_list (list): List of dates with full image available
    """
    # test raster outliers by date    
    date_dict = {}

    # iterate over raster tiles by date
    for i in items:
        # check and add raster properties (ID to identify tiles and cloud coverage to order dates by cloud coverage) to dictionary by tile and date 
        # (items have different depending on satellite)
        
        # General explanation: date_dict[current date].update({current tile ID : current tile cloud coverage})
        
        if satellite == "sentinel-2-l2a":
            if i.datetime.date() in list(date_dict.keys()):
                # if date already exists in date_dict, update
                date_dict[i.datetime.date()].update({i.properties['s2:mgrs_tile']:i.properties['s2:high_proba_clouds_percentage']})
            else:
                # else, create and update
                date_dict[i.datetime.date()] = {}
                date_dict[i.datetime.date()].update({i.properties['s2:mgrs_tile']:i.properties['s2:high_proba_clouds_percentage']})

        elif satellite == "landsat-c2-l2":
            if i.datetime.date() in list(date_dict.keys()):
                # if date already exists in date_dict, update
                date_dict[i.datetime.date()].update({i.properties['landsat:wrs_row']:i.properties['landsat:cloud_cover_land']})
            else:
                 # else, create and update
                date_dict[i.datetime.date()] = {}
                date_dict[i.datetime.date()].update({i.properties['landsat:wrs_row']:i.properties['landsat:cloud_cover_land']})
                
    # Turn into DataFrame
    df_tile = pd.DataFrame.from_dict(date_dict, orient='index')

    # Drop rows where there are NaNs (Unavailable raster tiles)
    df_tile = df_tile.dropna()

    # Arrange by cloud coverage average
    df_tile['avg_cloud'] = df_tile.mean(axis=1)
    df_tile = df_tile.sort_values(by='avg_cloud')
    
    # Create list of dates
    date_list = df_tile.index.to_list()
    
    return df_tile, date_list

   
def available_datasets(items, satellite="sentinel-2-l2a", query={}, min_cloud_value=10):
    """
    Filters dates per quantile and finds available ones.

    Arguments:
        items (np.array): items intersecting time and area of interest
        satellite (str): satellite used to download imagery
        min_cloud_value (int): minimum cloud coverage value to be considered for quantile analysis

    Returns:
        date_list (list): List with available dates with filter
    """
    if query:
        if 'eo:cloud_cover' in list(query.keys()):
            min_cloud_value = query['eo:cloud_cover']['lt']

    # test raster outliers by date    
    date_dict = {}

    date_dict = {}
    # iterate over raster tiles by date
    for i in items:
        if satellite == "sentinel-2-l2a":
            # check and add raster properties to dictionary by tile and date
            # if date is within dictionary append properties from item to list
            if i.datetime.date() in list(date_dict.keys()):
                # gather cloud percentage, high_proba_clouds_percentage, no_data values and nodata_pixel_percentage
                # check if properties are within dictionary date keys
                if i.properties['s2:mgrs_tile']+'_cloud' in list(date_dict[i.datetime.date()].keys()):
                    date_dict[i.datetime.date()].update(
                        {i.properties['s2:mgrs_tile']+'_cloud':
                        i.properties['s2:high_proba_clouds_percentage']})
                    #date_dict[i.datetime.date()].update(
                    #    {i.properties['s2:mgrs_tile']+'_nodata':
                    #    i.properties['s2:nodata_pixel_percentage']})
                
                else:
                    date_dict[i.datetime.date()].update(
                        {i.properties['s2:mgrs_tile']+'_cloud':
                        i.properties['s2:high_proba_clouds_percentage']})
                    #date_dict[i.datetime.date()].update(
                    #    {i.properties['s2:mgrs_tile']+'_nodata':
                    #    i.properties['s2:nodata_pixel_percentage']})
            # create new date key and add properties to it
            else:
                date_dict[i.datetime.date()] = {}
                date_dict[i.datetime.date()].update(
                    {i.properties['s2:mgrs_tile']+'_cloud':
                    i.properties['s2:high_proba_clouds_percentage']})
                #date_dict[i.datetime.date()].update(
                #    {i.properties['s2:mgrs_tile']+'_nodata':
                #    i.properties['s2:nodata_pixel_percentage']})
        elif satellite == "landsat-c2-l2":
            # check and add raster properties to dictionary by tile and date
            # if date is within dictionary append properties from item to list
            if i.datetime.date() in list(date_dict.keys()):
                # gather cloud percentage, high_proba_clouds_percentage, no_data values and nodata_pixel_percentage
                # check if properties are within dictionary date keys
                if i.properties['landsat:wrs_row']+'_cloud' in list(date_dict[i.datetime.date()].keys()):
                    date_dict[i.datetime.date()].update(
                        {i.properties['landsat:wrs_row']+'_cloud':
                        i.properties['landsat:cloud_cover_land']})
                
                else:
                    date_dict[i.datetime.date()].update(
                        {i.properties['landsat:wrs_row']+'_cloud':
                        i.properties['landsat:cloud_cover_land']})
            # create new date key and add properties to it
            else:
                date_dict[i.datetime.date()] = {}
                date_dict[i.datetime.date()].update(
                    {i.properties['landsat:wrs_row']+'_cloud':
                    i.properties['landsat:cloud_cover_land']})
    
    # determine third quartile for each tile
    df_tile = pd.DataFrame.from_dict(date_dict, orient='index')
    q3 = [np.percentile(df_tile[c].dropna(),[75]) for c in df_tile.columns.to_list() if 'cloud' in c]
    q3 = [v[0] for v in q3]

    # check if q3 analysis is necessary
    ### UPDATE NECESSARY TO REMOVE THIS CONDITIONAL
    '''q3_test = [True if test>min_cloud_value else False for test in q3]
    if sum(q3_test)>0:
        log(f'Quantile filter dictionary by column: {dict(zip(df_tile.columns, q3))}')

        column_list = df_tile.columns.to_list()

        # filter dates by missing values or outliers according to cloud and no_data values
        for c in range(len(column_list)):
            df_tile.loc[df_tile[column_list[c]]>min_cloud_value,column_list[c]] = np.nan
    else:
        log('Fixed filter applied')
        column_list = df_tile.columns.to_list()

        # filter dates by missing values or outliers according to cloud and no_data values
        for c in range(len(column_list)):'''
    column_list = df_tile.columns.to_list()
    for c in range(len(column_list)):
        df_tile.loc[df_tile[column_list[c]]>min_cloud_value,column_list[c]] = np.nan

    # arrange by cloud coverage average
    df_tile['avg_cloud'] = df_tile.mean(axis=1)
    df_tile = df_tile.sort_values(by='avg_cloud')
    log(f'Updated average cloud coverage: {df_tile.avg_cloud.mean()}')

    # create list of dates within normal distribution and without missing values
    date_list = df_tile.dropna().index.to_list()

    log(f'Available dates: {len(date_list)}')
    log(f'Raster tiles per date: {len(df_tile.columns.to_list())}')

    return date_list


def mosaic_raster(raster_asset_list, tmp_dir='tmp/', upscale=False):
    """
    The mosaic_raster function takes a list of raster assets and merges them together.

        Arguments:
            raster_asset_list (list): A list of raster asset paths to be appended together.
            tmp_dir (str): The directory where temporary files will be stored during processing. Defaults to 'tmp/'.
            upscale (bool): Whether or not the mosaic is upscaled by 2x  using the formats of the conditional statement

        Returns:
            mosaic (np.array): merged raster data
            out_trans (str): transformation information for mosaic raster
            meta (dictionary): Metadata of the raster object    
    """

    src_files_to_mosaic = []

    for assets in raster_asset_list:
        src = rasterio.open(assets)
        src_files_to_mosaic.append(src)
        
    mosaic, out_trans = merge(src_files_to_mosaic) # mosaic raster
    
    meta = src.meta
    
    if upscale:
        # save raster
        out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "dtype": 'float32',
                         "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})
        # write raster
        with rasterio.open(tmp_dir+"mosaic_upscale.tif", "w", **out_meta) as dest:
            dest.write(mosaic)

            dest.close()
        # read and upscale
        with rasterio.open(tmp_dir+"mosaic_upscale.tif", "r") as ds:

            upscale_factor = 1/2

            mosaic = ds.read(
                        out_shape=(
                            ds.count,
                            int(ds.height * upscale_factor),
                            int(ds.width * upscale_factor)
                        ),
                        resampling=Resampling.bilinear
                    )
            out_trans = ds.transform * ds.transform.scale(
                                (ds.width / mosaic.shape[-1]),
                                (ds.height / mosaic.shape[-2])
                            )
            meta = ds.meta

            meta.update({"driver": "GTiff",
                            "dtype": 'float32',
                            "height": mosaic.shape[1],
                            "width": mosaic.shape[2],
                            "transform": out_trans})

        ds.close()
    src.close()
    
    return mosaic, out_trans, meta

def mosaic_process(links_band_1, links_band_2, band_name_dict, gdf_bb, tmp_dir=''):
    """
    The function takes in two lists of links to raster files, and a list of band names.
    It then mosaics the first list of links into one large array, and does the same for the second list.
    The function returns four objects: 
        1) The mosaic_band_array for band 1 (mosaic_band_2), 
        2) The mosaic_band array for band 2 (mosaic_band2), 
        3) A transformation matrix that can be used to transform coordinates from
        pixel space to map space (outtrans)  
        4) An object containing the metadata for the output file (out_meta)

    Arguments: 
        links_band_1 (list): Pass in the links for band 1
        links_band_2 (list): Get the output_transform and output_meta
        band_name_list (list): Name the output files
        tmp_dir (str): Specify a temporary directory to store the intermediate files

    Returns: 
        mosaic_band_1 (np.array): The mosaic array for band 1
        mosaic_band_2 (np.array): The mosaic array for band 2
        out_trans_band_2 (np.array): The transformation matrix for band 2
        out_meta (object): The metadata for the output file.      
    """
    log(f'Starting mosaic for {list(band_name_dict.keys())[0]}')
    mosaic_band_1, out_trans_band_1, out_meta_1= mosaic_raster(links_band_1, tmp_dir, 
                                                               upscale=band_name_dict[list(band_name_dict.keys())[0]][0])
    mosaic_band_1 = mosaic_band_1.astype('float16')

    out_meta_1.update({"driver": "GTiff",
                    "dtype": 'float32',
                    "height": mosaic_band_1.shape[1],
                    "width": mosaic_band_1.shape[2],
                    "transform": out_trans_band_1})

    log(f'Starting save: {list(band_name_dict.keys())[0]}')

    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[0]}.tif", "w", **out_meta_1) as dest:
        dest.write(mosaic_band_1)

        dest.close()
        
    del mosaic_band_1
    log('Finished saving complete dataset')
    
    log('Starting crop')
    
    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[0]}.tif") as src:
        gdf_bb = gdf_bb.to_crs(src.crs)
        shapes = [gdf_bb.iloc[feature].geometry for feature in range(len(gdf_bb))]
        mosaic_band_1, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                            "dtype": 'float32',
                            "height": mosaic_band_1.shape[1],
                            "width": mosaic_band_1.shape[2],
                            "transform": out_transform})
        src.close()


    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[0]}.tif", "w", **out_meta) as dest:
        dest.write(mosaic_band_1)

        dest.close()

    mosaic_band_1 = mosaic_band_1.astype('float16')

    log(f'Finished croping: {list(band_name_dict.keys())[0]}')

    log(f'Finished processing {list(band_name_dict.keys())[0]}')

    log(f'Starting mosaic for {list(band_name_dict.keys())[1]}')
    mosaic_band_2, out_trans_band_2, out_meta_2 = mosaic_raster(links_band_2, tmp_dir, 
                                                               upscale=band_name_dict[list(band_name_dict.keys())[1]][0])
    log(f'Finished processing {list(band_name_dict.keys())[1]}')
    mosaic_band_2 = mosaic_band_2.astype('float16')
    log('Transformed band arrays to float16')

    out_meta_2.update({"driver": "GTiff",
                    "dtype": 'float32',
                    "height": mosaic_band_2.shape[1],
                    "width": mosaic_band_2.shape[2],
                    "transform": out_trans_band_2})

    log(f'Starting save: {list(band_name_dict.keys())[1]}')

    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[1]}.tif", "w", **out_meta_2) as dest:
        dest.write(mosaic_band_2)

        dest.close()
        
    del mosaic_band_2
    log('Finished saving complete dataset')
    
    log('Starting crop')
    
    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[1]}.tif") as src:
        gdf_bb = gdf_bb.to_crs(src.crs)
        shapes = [gdf_bb.iloc[feature].geometry for feature in range(len(gdf_bb))]
        mosaic_band_2, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
        out_meta.update({"driver": "GTiff",
                            "dtype": 'float32',
                            "height": mosaic_band_2.shape[1],
                            "width": mosaic_band_2.shape[2],
                            "transform": out_transform})
        src.close()

    with rasterio.open(f"{tmp_dir}{list(band_name_dict.keys())[1]}.tif", "w", **out_meta) as dest:
        dest.write(mosaic_band_2)

        dest.close()

    mosaic_band_2 = mosaic_band_2.astype('float16')

    log(f'Finished croping: {list(band_name_dict.keys())[1]}')

    log(f'Finished processing {list(band_name_dict.keys())[1]}')
    

    return mosaic_band_1, mosaic_band_2, out_transform, out_meta

def raster_nan_test(gdf, raster_file):
    """
    The function performs a test to check for Not-a-Number values in a raster file.

    Arguments:
        gdf (geopandas.GeoDataFrame): Pass in the geodataframe containing geometries that we want to check for nan values
        raster_file (rasterio.io.DatasetReader): Specify the raster file to be used in the function

    Raises:
        An exception if needed.
    """
    
    gdf['test'] = gdf.geometry.apply(lambda geom: clean_mask(geom, raster_file)).apply(np.ma.mean)
    
    log(f'There are {gdf.test.isna().sum()} null data values')

    if gdf['test'].isna().sum() > 0:
        raise NanValues('NaN values are still present after processing')

def mosaic_process_v2(raster_bands, band_name_dict, gdf_bb, tmp_dir):
    
    raster_array = {}

    band_names_list = list(band_name_dict.keys())[:-1]
    
    for b in band_names_list:

        log(f'Starting mosaic for {b}')
        raster_array[b]= [mosaic_raster(raster_bands[b], tmp_dir, 
                                       upscale=band_name_dict[b][0])]
        # mosaic_raster creates a tuple which has to be unpacked
        raster_array[b] = [raster_array[b][0][0],
                  raster_array[b][0][1],
                  raster_array[b][0][2]]
        raster_array[b][0] = raster_array[b][0].astype('float16')
        # return mosaic, out_trans, meta

        raster_array[b][2].update({"driver": "GTiff",
                        "dtype": 'float32',
                        "height": raster_array[b][0].shape[1],
                        "width": raster_array[b][0].shape[2],
                        "transform": raster_array[b][1]})

        log(f'Starting save: {b}')

        with rasterio.open(f"{tmp_dir}{b}.tif", "w", **raster_array[b][2]) as dest:
            dest.write(raster_array[b][0])

            dest.close()
            
        raster_array[b][0] = [np.nan]
        log('Finished saving complete dataset')
        
        log('Starting crop')
        
        with rasterio.open(f"{tmp_dir}{b}.tif") as src:
            gdf_bb = gdf_bb.to_crs(src.crs)
            shapes = [gdf_bb.iloc[feature].geometry for feature in range(len(gdf_bb))]
            raster_array[b][0], raster_array[b][1] = rasterio.mask.mask(src, shapes, crop=True)
            raster_array[b][2] = src.meta
            raster_array[b][2].update({"driver": "GTiff",
                                "dtype": 'float32',
                                "height": raster_array[b][0].shape[1],
                                "width": raster_array[b][0].shape[2],
                                "transform": raster_array[b][1]})
            src.close()


        with rasterio.open(f"{tmp_dir}{b}.tif", "w", **raster_array[b][2]) as dest:
            dest.write(raster_array[b][0])

            dest.close()

        raster_array[b][0] = raster_array[b][0].astype('float16')

        log(f'Finished croping: {b}')

        log(f'Finished processing {b}')

    return raster_array


def create_raster_by_month(df_len, index_analysis, city, tmp_dir, 
                           band_name_dict, date_list, gdf_raster_test, gdf_bb, 
                           aoi, sat, query={}, time_exc_limit=1500,
                           compute_unavailable_dates=True):
    """
    The function is used to create a raster for each month of the year within the time range
    The function takes in a dataframe with the length of years and months, an index analysis, city name, 
    temporary directory path (tmp_dir), band name list (band_name_list), date list (date_list), 
    geodataframe bounding box(gdf_bb) and area of interest(aoi). 
    the function also performs raster analysis for each row of the DataFrame, downloads and processes 
    the raster data, calculates an index, crops the raster, performs interpolation, and 
    saves the processed rasters and corresponding metadata in the specified directory. 

    Arguments:
        df_len (pandas.DataFrame): Summary dataframe indicating available raster data for each month
        index_analysis (str): Define the index analysis
        city (str): Save the raster files based on the city name
        tmp_dir (str): Save the raster files in a temporary directory
        band_name_list (list): Define the bands that will be used in the analysis
        date_list (list): Define the dates that are used to download the images
        gdf_bb (geopandas.GeoDataFrame): Crop the raster to a specific area of interest
        aoi (str): Define the area of interest
        sat (str): Define the satellite used to gather data
        time_exc_limit (int): Set the time limit for downloading a raster
        compute_unavailable_dates (bool): Whether or not to consider unavailable dates (Raises errors when too many unavailable). Defaults to True.

    Returns:
        df_len (pandas.DataFrame): Summary dataframe indicating available raster data for each month.  
    """
    df_len['able_to_download'] = np.nan
    band_name_list = list(band_name_dict.keys())[:-1]

    log('\n Starting raster analysis')

    # check if file exists, for example in case of code crash
    df_file_dir = tmp_dir+index_analysis+f'_{city}_dataframe.csv'
    if os.path.exists(df_file_dir) == False: # Or folder, will return true or false
        df_len.to_csv(df_file_dir, index=False)
    # create folder to store temporary raster files by iteration
    tmp_raster_dir = tmp_dir+'temporary_files/'
    if os.path.exists(tmp_raster_dir) == False: # Or folder, will return true or false
        os.mkdir(tmp_raster_dir)

    for i in tqdm(range(len(df_len)), position=0, leave=True):
        
        df_raster = pd.read_csv(df_file_dir, index_col=False)

        # binary id - checks if month could be processed
        checker = 0

	# gather month and year from df to save raster
        month_ = df_raster.loc[df_raster.index==i].month.values[0]
        year_ = df_raster.loc[df_raster.index==i].year.values[0]
        
        # check if raster already exists
        if f'{city}_{index_analysis}_{month_}_{year_}.tif' in os.listdir(tmp_dir):
            df_raster.loc[i,'data_id'] = 11
            df_raster.to_csv(df_file_dir, index=False)
            continue
        
        # check if month is available
        if df_raster.iloc[i].data_id==0:
            continue

        log(f'\n Starting new analysis for {month_}/{year_}')
        
        # gather links for raster images
        sample_date = datetime(year_, month_, 1)
        first_day = sample_date + relativedelta(day=1)
        last_day = sample_date + relativedelta(day=31)

        # creates time range for a specific month
        time_of_interest = [f"{year_}-{month_:02d}-{first_day.day:02d}/{year_}"+
                            f"-{month_:02d}-{last_day.day:02d}"]
    
        # create dataframe
        #df_links = pd.DataFrame.from_dict(assets_hrefs, 
        #                                orient='Index').reset_index().rename(columns={'index':'date'})

        # dates according to cloud coverage
        date_order = [True if (d.month == month_) and (d.year == year_) else False for d in date_list]
        date_array = np.array(date_list)
        date_filter = np.array(date_order)
        dates_ordered = date_array[date_filter]
        
        # mosaic raster iterations (while loop tries 5 times to process all available rasters (dates) in a month)
        iter_count = 1
        # create skip date list used to analyze null values in raster
        skip_date_list = []
        
        while iter_count <= 5:
            # gather links for the date range from planetary computer
            items = gather_items(time_of_interest, aoi, query=query, satellite=sat)
            
            # gather links from dates that are within date_list
            assets_hrefs = link_dict(band_name_list, items, date_list)

            #for data_link in range(len(df_links)):
            for data_link in range(len(dates_ordered)):
                 
                # Skip date if in skip_date_list
                if dates_ordered[data_link] in skip_date_list:
                    log(f'Skipped {dates_ordered[data_link]} - iteration:{iter_count} because it did not pass null test.')
                    continue
                 
                # log(data_link)
                log(f'Skip list:{skip_date_list}')
                log(dates_ordered[data_link])
                log(f'Mosaic date {dates_ordered[data_link].day}'+
                            f'/{dates_ordered[data_link].month}'+
                            f'/{dates_ordered[data_link].year} - iteration:{iter_count}')
                
                # check if date contains null values within study area
                #if df_links.iloc[data_link]['date'] in skip_date_list:

                try:
                    #links_band_1 = df_links.iloc[data_link][list(band_name_dict.keys())[0]]
                    #links_band_2 = df_links.iloc[data_link][list(band_name_dict.keys())[1]]
                    bands_links = assets_hrefs[dates_ordered[data_link]]

                    rasters_arrays = func_timeout(time_exc_limit, mosaic_process_v2,
                                                                                args=(bands_links,
                                                                                      band_name_dict, gdf_bb, tmp_raster_dir))
                    out_meta = rasters_arrays[list(rasters_arrays.keys())[0]][2]

                    # calculate raster index
                    raster_idx = calculate_raster_index(band_name_dict, rasters_arrays)
                    log(f'Calculated {index_analysis}')
                    del rasters_arrays

                    log(f'Starting interpolation')

                    raster_idx[raster_idx == 0 ] = np.nan # change zero values to nan
                    # only for temperature
                    raster_idx[raster_idx == -124.25 ] = np.nan # change zero values to nan
                    raster_idx = raster_idx.astype('float32') # change data type to float32 to avoid fillnodata error

                    log(f'Interpolating {np.isnan(raster_idx).sum()} nan values')
                    raster_fill = fillnodata(raster_idx, mask=~np.isnan(raster_idx),
                                        max_search_distance=50, smoothing_iterations=0)
                    log(f'Finished interpolation to fill na - {np.isnan(raster_fill).sum()} nan')

                    with rasterio.open(f"{tmp_raster_dir}{index_analysis}.tif",'w', **out_meta) as dest:
                            dest.write(raster_fill)

                            dest.close()

                    log('Starting null test')

                    raster_file = rasterio.open(f"{tmp_raster_dir}{index_analysis}.tif")

                    gdf_raster_test = gdf_raster_test.to_crs(raster_file.crs)

                    try:
                        # test for nan values within study area
                        raster_nan_test(gdf_raster_test,raster_file)

                        log('Passed null test')
                        
                        # save raster to processing database
                        with rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif",'w', **out_meta) as dest:
                            dest.write(raster_fill)

                            dest.close()
                        log(f'Finished saving {index_analysis} raster')

                        checker = 1
                        iter_count = 6
                        delete_files_from_folder(tmp_raster_dir)
                        break
                    except:
                        log('Failed null test')
                        skip_date_list.append(dates_ordered[data_link])
                        delete_files_from_folder(tmp_raster_dir)

                except:
                    log(f'Error in iteration {iter_count}')
                    delete_files_from_folder(tmp_raster_dir)
                    continue
            iter_count = iter_count + 1
                
        if checker==0:
            df_raster.loc[df_raster.index==i,'data_id']=0
            df_raster.loc[df_raster.index==i,'able_to_download']=0
            df_raster.to_csv(df_file_dir, index=False)
            if compute_unavailable_dates:
                available_data_check(df_raster, len(df_raster.loc[df_raster.data_id==0])) # test for missing months
            continue

    df_len = pd.read_csv(df_file_dir, index_col=False)

    return df_len

def calculate_raster_index(band_name_dict, raster_arrays):
    """
    The function calculates the raster index according to a user equation. 
    If no equation is provided, the raster_array is returned.

    Args:
        band_name_dict (dict): dictionary containing the band names and the equation to be used
        raster_arrays (dict): dictionary containing the rasters numpy arrays

    Returns:
        np.array: resulting numpy array for the raster index
    """
    raster_equation = band_name_dict['eq'][0]

    if len(band_name_dict['eq']) == 0:
        # if there is no equation the raster array is the result
        raster_idx = raster_arrays[list(raster_arrays.keys())[0]]
        return raster_idx

    for rb in raster_arrays.keys():
        raster_equation = raster_equation.replace(rb,f"ra['{rb}'][0]")
    # create global variable in order to use it in exec as global
    global ra
    ra = raster_arrays
    exec(f"raster_index = {raster_equation}", globals())
    
    del ra

    return raster_index


def raster_interpolation(df_len, city, tmp_dir, index_analysis):
    """
    This function interpolates missing raster data by time windows, filling the gaps in unavailable months. 
      
     Arguments:
        df_len (pandas.DataFrame): Pass the dataframe containing the information of each raster file
        city (str): Name the raster files
        tmp_dir (str): Specify the directory where the raster files are stored
        index_analysis (str) : Select the index to be analyzed

    Returns:
        df_len (pandas.DataFrame): Returns the updated dataframe from the cvs document that arranged
        all the information of the raster files.
    """

    log(f'Interpolating {len(df_len.loc[df_len.data_id==0])}')

    available_data_check(df_len, len(df_len.loc[df_len.data_id==0]))

    df_len['interpolate'] = 0
    
    for row in range(len(df_len)):
    
        if df_len.iloc[row].data_id == 0:
            # Set starting row to previus row (Unless it is the first row)
            start = row - 1
            if start == -1:
                start = 0
            
            # Try setting finish row to the first ocurrance (.idmax()) of all following rows ([row:,:])
            # where data is not equal (ne) to cero (rows with downloaded images).
            # Meaning, the next row with a downloaded image.
            try:
                finish = df_len.iloc[row:,:].data_id.ne(0).idxmax()
            # Except (Zero next rows with a downloaded image), finish row is last row.
            except:
                finish = len(df_len)
                
            log(f'Row start:{start} - row finish:{finish}')
                    
            df_subset = df_len.iloc[start:finish+1]
            if df_subset.loc[df_subset.index==start].data_id.values[0] == 0:
                
                log('Entering missing data case 1 - first value missing')
                
                month_ = df_subset.loc[df_subset.index==finish].month.values[0]
                year_ = df_subset.loc[df_subset.index==finish].year.values[0]
                raster_file = rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif")
                meta = raster_file.meta
                raster_last = raster_file.read()
                
                log('Read last raster data')
                
                cont = 0
                while df_subset.iloc[cont].data_id == 0:
                    month_ = int(df_subset.iloc[cont].month)
                    year_ = int(df_subset.iloc[cont].year)
                    with rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif",'w', **meta) as dest:
                        dest.write(raster_last)

                        dest.close()
                        
                        log('Finished creating raster')
                        df_len.loc[df_len.index==start+cont,'data_id'] = 1
                        df_len.loc[df_len.index==start+cont,'interpolate'] = 1
                    
                    cont += 1
                
            elif df_subset.loc[df_subset.index==finish].data_id.values[0] == 0:
                
                log('Entering missing data case 2 - last value missing')
                
                month_ = df_subset.loc[df_subset.index==start].month.values[0]
                year_ = df_subset.loc[df_subset.index==start].year.values[0]
                raster_file = rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif")
                meta = raster_file.meta
                raster_first = raster_file.read()
                
                log('Read first raster data')
                
                cont = 1
                while cont < len(df_subset):
                    month_ = int(df_subset.iloc[cont].month)
                    year_ = int(df_subset.iloc[cont].year)
                    with rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif",'w', **meta) as dest:
                        dest.write(raster_first)

                        dest.close()
                        
                        log('Finished creating raster')
                        df_len.loc[df_len.index==start+cont,'data_id'] = 1
                        df_len.loc[df_len.index==start+cont,'interpolate'] = 1
                    
                    cont += 1
                
            else:
                
                log('Entering missing data case 3  - mid point missing')
                
                month_ = df_subset.loc[df_subset.index==start].month.values[0]
                year_ = df_subset.loc[df_subset.index==start].year.values[0]
                raster_file = rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif")
                raster_first = raster_file.read()
                
                month_ = df_subset.loc[df_subset.index==finish].month.values[0]
                year_ = df_subset.loc[df_subset.index==finish].year.values[0]
                raster_file = rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif")
                meta = raster_file.meta
                raster_last = raster_file.read()
                
                missing_len = len(df_subset.loc[df_subset.data_id==0])
                slope = 1 / (missing_len + 1)
                slope_increment = slope
                
                cont = 1
                
                log(f'Preparing data for interpolation with slope {slope}')
                
                # rejoin arr1, arr2 into a single array of shape (2, 10, 10)
                arr = np.r_['0,3', raster_first, raster_last]
                # define the grid coordinates where you want to interpolate
                dim_row = raster_first.shape[1]
                dim_col = raster_first.shape[2]

                X, Y = np.meshgrid(np.arange(dim_row), np.arange(dim_col))
                
                while round(slope,4) < 1:
                    
                    log(f'Starting interpolation for iteration {cont} with position {slope}')
                    # switch order of col and row
                    coordinates = np.ones((dim_col, dim_row))*slope, X, Y
                    
                    inter_raster = ndimage.map_coordinates(arr, coordinates, order=1).T
                    inter_raster = inter_raster.reshape((1,inter_raster.shape[0],inter_raster.shape[1]))
                    
                    month_ = int(df_subset.iloc[cont].month)
                    year_ = int(df_subset.iloc[cont].year)
                    with rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month_}_{year_}.tif",'w', **meta) as dest:
                        dest.write(inter_raster)

                        dest.close()
                        
                        log(f'Finished creating raster')
                        
                        df_len.loc[df_len.index==start+cont,'data_id'] = 1
                        df_len.loc[df_len.index==start+cont,'interpolate'] = 1
                    
                    cont += 1
                    slope = slope + slope_increment

    df_file_dir = tmp_dir+index_analysis+f'_{city}_dataframe.csv'
    df_len.to_csv(df_file_dir,index=False)

    return df_len

def clean_mask(geom, dataset='', **mask_kw):
    """
    The mask in this function is used to extract the values from a raster dataset that fall 
    within a given geometry of interest.

    Arguments:
        geom (geometry): Geometric figure that will be used to mask the raster dataset.
        dataset (rasterio DatasetReader): The raster dataset that will be masked by the 
        inputted geometry. If no value is provided, then it defaults to an empty string 
        and returns only the masked array of values from within the inputted geometry 
        without any metadata.
        mask_kw (dict): A dictionary of arguments passed to create the mask.

    Returns:
        masked (np.array): Returns values from within the inputted geometry.
    """
    
    mask_kw.setdefault('crop', True)
    mask_kw.setdefault('all_touched', True)
    mask_kw.setdefault('filled', False)
    masked, _ = rasterio.mask.mask(dataset=dataset, shapes=(geom,),
                                  **mask_kw)
    return masked


def mask_by_hexagon(hex_gdf,year,month,city,index_analysis,tmp_dir):
    """"
    The function takes a hexagon GeoDataFrame, year, month, city name and index analysis as input.
    It then opens the raster file for that specific month and year in the tmp_dir directory. 
    It applies a mask to the raster file 

    Arguments:
        hex_gdf (geopandas.GeoDataFrame): Creates a copy of the hexagon geodataframe
        year (int): Specify the year that will be used
        month (int): Month index that will be used
        city (str): Specify the city for which we want to calculate the ndmi
        index_analysis (str): Specify which index analysis to use
        tmp_dir (str): Specify the directory where the raster files are stored

    Returns:
        hex_raster (np.array): A hexagon raster with the index analysis added as a column
    """
    hex_raster = hex_gdf.copy()
    # read ndmi file
    raster_file = rasterio.open(f"{tmp_dir}{city}_{index_analysis}_{month}_{year}.tif")

    hex_raster = hex_raster.to_crs(raster_file.crs) 
    # Using the apply function to apply the clean_mask function to the geometry column of the hex_gdf geodataframe
    try:

        hex_raster[index_analysis] = hex_raster.geometry.apply(lambda geom: clean_mask(geom, raster_file)).apply(np.ma.mean)
    except:
        hex_raster[index_analysis] = np.nan
    #adds month and year columns to the hex_raster geodataframe
    hex_raster['month'] = month
    hex_raster['year'] = year

    hex_raster = hex_raster.to_crs("EPSG:4326")

    return hex_raster


def raster_to_hex_multi(hex_gdf, df_len, index_analysis, city, raster_dir):
    """
    The function takes a  geodataframe, containing dates for available datasets,
    the index analysis a specified multispectral band index, and the city name as inputs. 
    It then creates an empty geodataframe to to save index_analysis by date. The function loops through 
    each year in df_len and for each month in that year, to mask each and every raster 
    to its corresponding hexagon.

    Arguments:
        hex_gdf (geopandas.GeoDataFrame): Pass the hexagon geodataframe to the function
        df_len (int): Determine the number of years and months that are in the data
        index_analysis (str): Specify which index analysis to use
        city (str): Specify the city of interest
        raster_dir (str): Specify the directory where the raster files are stored

    Returns: 
    hex_raster (geopandas.GeoDataFrame): A geodataframe with the hexagon id.
    """
    
    # create empty geodataframe to save ndmi by date

    hex_raster = gpd.GeoDataFrame()

    years_list = list(df_len.year.unique())

    for i in tqdm(range(len(years_list)),position=0,leave=True):
        y = years_list[i]
        input_list = [[hex_gdf,y,month,city,index_analysis,raster_dir] for month in list(df_len.loc[df_len.year==y].month.unique())]
        pbar = tqdm(total=len(input_list))
        pool = Pool()
        hex_res = pd.concat(pool.starmap(mask_by_hexagon,input_list))
        pool.close()
        hex_raster = pd.concat([hex_raster, hex_res], 
            ignore_index = True, axis = 0)
        del hex_res
        
    return hex_raster

def raster_to_hex(hex_gdf, df_len, r, index_analysis, city, raster_dir):
    """
    The function takes a hexagonal grid, a dataframe of dates, the name of the satellite imagery index
    to then return it into a geodataframe a  mean value for each hexagon in a grid for each date
    to offer a better classification in a csv file.

    Arguments:
        hex_gdf (geopandas.GeoDataFrame): Pass the hexagonal grid to the function
        df_len (int): Iterate through the dataframe containing the dates of each image
        r (int): Specify the resolution of the hexagons
        index_analysis (str): Select the index to be analyzed
        city (str): Specify the city to be analyzed
        raster_dir (str): Specify the directory where the raster files are stored

    Returns:
        hextmp (geopandas.GeoDataFrame): A geodataframe with the mean value of each index by hexagon and date
    """
    
    # create empty geodataframe to save ndmi by date
    hex_raster = gpd.GeoDataFrame()

    for d in tqdm(range(len(df_len)),position=0,leave=True):

        month_ = df_len.loc[df_len.index==d].month.values[0]
        year_ = df_len.loc[df_len.index==d].year.values[0]

        hex_tmp = hex_gdf.copy()

        if df_len.iloc[d].data_id==1:

            # read index_analysis file
            raster_file = rasterio.open(f"{raster_dir}{city}_{index_analysis}_{month_}_{year_}.tif")

            hex_tmp = hex_tmp.to_crs(raster_file.crs)

            try:
                hex_tmp[index_analysis] = hex_tmp.geometry.apply(lambda geom: clean_mask(geom, raster_file)).apply(np.ma.mean)
            except:
                hex_tmp[index_analysis] = np.nan

        else:
            hex_tmp[index_analysis] = np.nan

        hex_tmp['month'] = month_
        hex_tmp['year'] = year_

        hex_tmp = hex_tmp.to_crs("EPSG:4326")

        # concatenate into single geodataframe
        hex_raster = pd.concat([hex_raster, hex_tmp], 
            ignore_index = True, axis = 0)

        del hex_tmp
        
    return hex_raster


def raster_to_hex_analysis(hex_gdf, df_len, index_analysis, tmp_dir, city, res):
    """
    The function groups the raster by hexagons and calculates 
    summary statistics for each one of them.
    The function returns a dataframe and a geodataframe: one with summary statistics for each of the hexagons
    (hexagon id, mean value of index analysis per year) and another with all values 
    from the raster assigned to their respective hexagon.

    Arguments:
        hex_gdf (geopandas.GeoDataFrame): Gets the hexagons
        df_len (pandas.DataFrame): Divides the dataframe into chunks to be processed in parallel
        index_analysis (str): Specify the column name of the index we want to analyze
        tmp_dir (str): Store the raster files in a temporary directory
        city (str): city information from the dataframes
        res (int): hexagon resolution used to filter the dataframe by resolution

    Returns:
        hex_raster_analysis (geopandas.GeoDataFrame): Has the summary statistics for each of the hexagons
        hex_raster_df (pandas.DataFrame): Has all values from the raster assigned to their respective hexagon
    """
    
    # group raster by hex

    log('Starting raster to hexagons')
    hex_gdf = hex_gdf.copy()
    hex_raster = raster_to_hex_multi(hex_gdf, df_len, index_analysis, city, tmp_dir)
    log('Assigned raster data to hexagons')
    
    # summary statistics
    hex_raster_analysis = hex_gdf[['hex_id','geometry','res']].drop_duplicates().copy()
    
    hex_raster_minmax = hex_raster[['hex_id',index_analysis,'year']].groupby(['hex_id','year']).agg(['max','min'])
    hex_raster_minmax.columns = ['_'.join(col) for col in hex_raster_minmax.columns]
    hex_raster_minmax = hex_raster_minmax.reset_index()
    hex_raster_minmax = hex_raster_minmax[['hex_id',f'{index_analysis}_max',f'{index_analysis}_min']].groupby(['hex_id']).mean()
    hex_raster_minmax = hex_raster_minmax.reset_index()

    hex_group_data = hex_raster[['hex_id',index_analysis]].groupby('hex_id').agg(['mean','std',
                                                                                'median',mk.sens_slope])
    hex_group_data.columns = ['_'.join(col) for col in hex_group_data.columns]
    hex_group_data = hex_group_data.reset_index().merge(hex_raster_minmax, on='hex_id')
    
    hex_raster_analysis = hex_raster_analysis.merge(hex_group_data, on='hex_id')
    hex_raster_analysis[index_analysis+'_diff'] = hex_raster_analysis[index_analysis+'_max'] - hex_raster_analysis[index_analysis+'_min']
    hex_raster_analysis[index_analysis+'_tend'] = hex_raster_analysis[f'{index_analysis}_sens_slope'].apply(lambda x: x[0])
    hex_raster_analysis = hex_raster_analysis.drop(columns=[f'{index_analysis}_sens_slope'])
    
    # remove geometry information
    hex_raster_df = hex_raster.drop(columns=['geometry'])
    
    # add city information
    hex_raster_df['city'] = city
    hex_raster_analysis['city'] = city

    log(f'df nan values: {hex_raster_df[index_analysis].isna().sum()}')

    return hex_raster_analysis, hex_raster_df
