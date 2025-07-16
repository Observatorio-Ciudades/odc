import geopandas as gpd
import pandas as pd
from data import *


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
