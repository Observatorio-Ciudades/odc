################################################################################
# Module: Visualization
# General visualization functions, mainly focused on kepler configurations
# updated: 10/08/2023
################################################################################

import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Wrap title text inside plots
from textwrap import wrap
# Import cm, colors and colorbar to create colormap legends manually
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from matplotlib.colors import TwoSlopeNorm #Tendency colorbar with both positive and negative values
# Import logo image
import matplotlib.image as mpimg
# Place logo image above plot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from .data import *

def hex_plot(ax, gdf_data, gdf_boundary, gdf_edges, column , title,save_png=False, save_pdf=False,show=False, name='plot',dpi=300,transparent=True, close_figure=True):
	"""
	Plot hexbin geoDataFrames to create the accesibility plots.

	Arguments:
		ax (matplotlib.axes): ax to use in the plot
		gdf_data (geopandas.GeoDataFrame): geoDataFrame with the data to be plotted
		gdf_boundary (geopandas.GeoDataFrame): geoDataFrame with the boundary to use 
		gdf_edges (geopandas.GeoDataFrame): geoDataFrame with the edges (streets)
		column (geopandas.GeoDataFrame) column to plot from the gdf_data geoDataFrame
		title (str): string with the title to use in the plot

	Keyword Arguments:
		save_png (bool): save the plot in png or not (default: {False})
		save_pdf (bool): save the plot in pdf or not (default: {False})
		show (bool): show the plot or not (default: {False})
		name (str): name for the plot to be saved if save=True (default: {plot})
		dpi (int) resolution to use (default: {300})
		transparent (bool): save with transparency or not (default: {True})
	"""
	divider = make_axes_locatable(ax) # Creates a divider for the axes
	cax = divider.append_axes("bottom", size="5%", pad=0.1) # Creates an axes for the colorbar (cax=cax). The size is size% of the main ax, while there's a spacing (pad) between the main ax and the colorbar

	# Plot data
	if len(gdf_data[gdf_data[column]<=0]) > 0:
		gdf_data[gdf_data[column]<=0].plot(ax=ax,color='#2b2b2b', alpha=0.95, linewidth=0.1, edgecolor='k', zorder=0)
	gdf_data[gdf_data[column]>0].plot(ax=ax,column=column, cmap='magma_r',vmax=1000,zorder=1,legend=True,cax=cax,legend_kwds={'label':'Distancia (m)','orientation': "horizontal"})
	# Plot boundary
	gdf_boundary.boundary.plot(ax=ax,color='#f8f8f8',zorder=2,linestyle='--',linewidth=0.5)
	# Plot edges
	if len(gdf_edges[(gdf_edges['highway']=='motorway') | (gdf_edges['highway']=='motorway_link')]) > 0:
		gdf_edges[(gdf_edges['highway']=='motorway') | (gdf_edges['highway']=='motorway_link')].plot(ax=ax,color='#898989',alpha=0.5,linewidth=2.5,zorder=3)
	if len(gdf_edges[(gdf_edges['highway']=='primary') | (gdf_edges['highway']=='primary_link')]) > 0:
		gdf_edges[(gdf_edges['highway']=='primary') | (gdf_edges['highway']=='primary_link')].plot(ax=ax,color='#898989',alpha=0.5,linewidth=1.5,zorder=3)
	# Format plot
	ax.set_title(f'{title}',fontdict={'fontsize':30})
	ax.axis('off')
	# Save or show plot
	if save_png:
		plt.savefig('../output/figures/{}.png'.format(name),dpi=dpi,transparent=transparent)
	if save_pdf:
		plt.savefig('../output/figures/{}.pdf'.format(name))
	if close_figure:
		plt.close()
	if show:
		plt.show()


def kepler_config():
	"""
	Create configuration dictionary data for kepler maps

	Returns:
		config, config_index (dict): dictionaries with to types of configurations depending on request	
	"""

	config = {'version': 'v1', 'config': 
	{'visState': {'filters': [], 'layers': [{'id': 'jsx1yd', 'type': 'geojson', 'config': 
	{'dataId': 'Análisis de hexágono', 'label': 'Análisis de hexágono', 'color': [231, 159, 213], 
	'columns': {'geojson': 'geometry'}, 'isVisible': True, 
	'visConfig': {'opacity': 0.85, 'strokeOpacity': 0.05, 'thickness': 0.5, 'strokeColor': [28, 27, 27], 
	'colorRange': {'name': 'Custom Palette', 'type': 'custom', 'category': 'Custom', 
	'colors': ['#00939c','#85c4c8','#feeee8','#ec9370','#c22e00']}, 
	'strokeColorRange': {'name': 'Global Warming', 'type': 'sequential', 'category': 'Uber', 
	'colors': ['#5A1846', '#900C3F', '#C70039', '#E3611C', '#F1920E', '#FFC300']}, 
	'radius': 10, 'sizeRange': [0, 10], 'radiusRange': [0, 50], 'heightRange': [0, 500], 
	'elevationScale': 5, 'stroked': True, 'filled': True, 'enable3d': False, 'wireframe': False}, 
	'hidden': False, 'textLabel': [{'field': None, 'color': [255, 255, 255], 'size': 18, 'offset': [0, 0], 'anchor': 'start', 'alignment': 'center'}]}, 
	'visualChannels': {'colorField': {'name': 'dist_farmacia', 'type': 'real'}, 'colorScale': 'quantile',
    'sizeField': None, 'sizeScale': 'linear', 'strokeColorField': None, 'strokeColorScale': 'quantile', 
	'heightField': {'name': 'dist_farmacia', 'type': 'real'}, 'heightScale': 'linear', 'radiusField': None, 'radiusScale': 'linear'}}], 
	'interactionConfig': {'tooltip': {'fieldsToShow': {'Análisis de hexágono': []}, 
	'compareMode': False, 'compareType': 'absolute', 'enabled': True}, 'brush': {'size': 0.5, 'enabled': False}, 
	'geocoder': {'enabled': False}, 'coordinate': {'enabled': False}}, 'layerBlending': 'normal', 'splitMaps': [], 
	'animationConfig': {'currentTime': None, 'speed': 1}}, 'mapState': {'bearing': 0, 'dragRotate': False, 
	'latitude': 32.42395273933573, 'longitude': -114.38059308118848, 'pitch': 0, 'zoom': 8.515158481972351, 
	'isSplit': False}, 'mapStyle': {'styleType': 'muted_night', 'topLayerGroups': {}, 
	'visibleLayerGroups': {'label': True, 'road': True, 'border': False, 'building': True, 'water': True, 'land': True, '3d building': False}, 
	'threeDBuildingColor': [9.665468314072013, 17.18305478057247, 31.1442867897876], 'mapStyles': {}},
	}}

	config_idx = {
		"version": "v1",
		"config": {
			"visState": {
				"filters": [],
				"layers": [
				{
					"id": "jsx1yd",
					"type": "geojson",
					"config": {
						"dataId": "Análisis de hexágono",
						"label": "Análisis de hexágono",
						"color": [
							231,
							159,
							213
						],
						"columns": {
							"geojson": "geometry"
						},
						"isVisible": True,
						"visConfig": {
							"opacity": 0.35,
							"strokeOpacity": 0.05,
							"thickness": 0.5,
							"strokeColor": [
								28,
								27,
								27
							],
							"colorRange": {
								"name": "Custom Palette",
								"type": "custom",
								"category": "Custom",
								"colors": [
									"#2C51BE",
									"#7A0DA6",
									"#CF1750",
									"#FD7900",
									"#FAE300"
								],
								"reversed": True
							},
							"strokeColorRange": {
								"name": "Global Warming",
								"type": "sequential",
								"category": "Uber",
								"colors": [
									"#5A1846",
									"#900C3F",
									"#C70039",
									"#E3611C",
									"#F1920E",
									"#FFC300"
								]
							},
							"radius": 10,
							"sizeRange": [
								0,
								10
							],
							"radiusRange": [
								0,
								50
							],
							"heightRange": [
								0,
								500
							],
							"elevationScale": 5,
							"stroked": True,
							"filled": True,
							"enable3d": False,
							"wireframe": False
						},
						"hidden": False,
						"textLabel": [
							{
								"field": None,
								"color": [
									255,
									255,
									255
								],
								"size": 18,
								"offset": [
									0,
									0
								],
								"anchor": "start",
								"alignment": "center"
							}
						]
					},
					"visualChannels": {
						"colorField": {
							"name": "bins_idx_accessibility",
							"type": "string"
						},
						"colorScale": "ordinal",
						"sizeField": None,
						"sizeScale": "linear",
						"strokeColorField": None,
						"strokeColorScale": "quantile",
						"heightField": {
							"name": "dist_farmacia",
							"type": "real"
						},
						"heightScale": "linear",
						"radiusField": None,
						"radiusScale": "linear"
					}
				}
				],
				"interactionConfig": {
					"tooltip": {
						"fieldsToShow": {
							"data": [
								{
									"name": "idx_accessibility",
									"format": None
								}
							]
						},
						"compareMode": False,
						"compareType": "absolute",
						"enabled": True
					},
					"brush": {
						"size": 0.5,
						"enabled": False
					},
					"geocoder": {
						"enabled": False
					},
					"coordinate": {
						"enabled": False
					}
				},
				"layerBlending": "normal",
				"splitMaps": [],
				"animationConfig": {
					"currentTime": None,
					"speed": 1
				}
			},
			"mapState": {
				"bearing": 0,
				"dragRotate": False,
				"latitude": 21.73127093107669,
				"longitude": -102.36507230084396,
				"pitch": 0,
				"zoom": 9.61578119574967,
				"isSplit": False
			},
			"mapStyle": {
				"styleType": "muted_night",
				"topLayerGroups": {},
				"visibleLayerGroups": {
					"label": True,
					"road": True,
					"border": False,
					"building": True,
					"water": True,
					"land": True,
					"3d building": False
				},
				"threeDBuildingColor": [
					9.665468314072013,
					17.18305478057247,
					31.1442867897876
				],
				"mapStyles": {}
			}
		}
	}

	return config, config_idx


def square_bounds(ax, bounds_gdf, boundary_expansion=[0.05,0.05]):
    """
    Formats a ax's boundaries in order to set squared bounds around a given GeoDataFrame's geometry.

    This function takes an ax and modifies the its boundaries to set a squared boundary
    around the geometry of a given GeoDataFrame. These bounds can be expanded or contracted 
    using values from boundary_expansion. This expansion is applied symmetrically, keeping the 
    bounds_gdf centered.
    
    Parameters
    ----------
    ax: matplotlib.axes
        ax to be modified using ax.set_xlim() and ax.set_ylim().
    bounds_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the geometry from which the initial bounding box is created.
    boundary_expansion: list, default [0.05,0.05].
        List with two numeric values that represent percentages used to expand the squared-boundary plot symmetrically.
        If no expansion is required, use [0,0], but note that the bounds will be very tight around the bounds_gdf.
        The first digit expands the x-axis and the second digit expands the y-axis.
        e.g. [0.05,0.10] would expand by 5% the horizontal bounds (2.5% to the left, 2.5% to the right)
        and by 10% the vertical bounds (5% to the bottom, 5% to the top).

    Returns
    -------
    None, modifies ax.

    """
    # Calculate bounds_gdf's current bounding box
    minx, miny, maxx, maxy = bounds_gdf.total_bounds
    
    # Find bounding box's larger size and its proportions relative to the shorter side
    x_dist = abs(maxx-minx)
    y_dist = abs(maxy-miny)
    larger_side = max(x_dist,y_dist)
    smaller_side = min(x_dist,y_dist)
    #ax_proportions = larger_side/smaller_side #To be used in later function modifications

    # Turn into squared bounds by adding the difference between the larger_side and the smaller_side to the smaller_side
    size_diff = larger_side-smaller_side
    # If it is already a square, no need to calculate
    if size_diff == 0:
        square_minx = minx
        square_maxx = maxx
        square_miny = miny
        square_maxy = maxy
    else:
        # If x_dist is the larger, enlarge <y> side
        if x_dist>y_dist:
            square_minx = minx #<x> is larger size
            square_maxx = maxx #<x> is larger size
            square_miny = miny-(size_diff/2) #<y> is enlarged
            square_maxy = maxy+(size_diff/2) #<y> is enlarged
        # Else, y_dist is the larger, enlarge <x> side
        else:
            square_minx = minx-(size_diff/2) #<x> is enlarged
            square_maxx = maxx+(size_diff/2) #<x> is enlarged
            square_miny = miny #<y> is larger size
            square_maxy = maxy #<y> is larger size

    # Grow squared bound by specified values in boundary_expansion
    expanded_square_minx = square_minx - (larger_side*boundary_expansion[0])/2
    expanded_square_maxx = square_maxx + (larger_side*boundary_expansion[0])/2
    expanded_square_miny = square_miny - (larger_side*boundary_expansion[1])/2
    expanded_square_maxy = square_maxy + (larger_side*boundary_expansion[1])/2

    # Adjust ax limits accordingly
    ax.set_xlim(expanded_square_minx, expanded_square_maxx)
    ax.set_ylim(expanded_square_miny, expanded_square_maxy)


def observatory_plot_format(ax, plot_title, legend_title, legend_type, cmap_args=[], grid=False):
    """
    Formats the plot to the observatory's style:

    This function positions the title on the top left corner of the plot, formats the legend title in bold, 
    and places OdC's logo on the bottom right corner of the plot.

    NOTE: Use this function after plotting all elements on it.
        Since matplotlib updates the layout size and proportions to the elements inserted onto the map, 
        this function must be used as a final plot formatting, after plotting all map elements.
    
    Parameters
    ----------
    ax: matplotlib.axes
        ax to be fit to the observatory's style.
    plot_title: str
        Text to be set as main plot title.
    legend_title: str
        Text to be set as legend title.
    legend_type: str
        Must be either 'categorized' or 'colorbar'. Edits the legend according to its type.
        For 'categorized' the legend is edited using matplotlib.legend.Legend.
        For 'colorbar' the legend is edited using matplotlib.colorbar.ColorbarBase.
    cmap_agrs: list, default [].
        Necessary if legend_type='colormap', this argument must contain two values used to create the colorbar:
        - The first element must be the previously generated colormap (which can be obtained using plt.get_cmap(''))
        - The second element must be the previously generated norm colors (which can be obtained using colors.Normalize())
    grid: bool, default False.
        If true turns on coordinates grid.

    Returns
    -------
    None, modifies ax.
    
    """
    ##########################################################################################
    # STEP 1: AX DIMENSIONS
    # Calculate current ax width, height and larger size (Relevant values used in title text wrapping, legent text sizing and image zoom calculation)
    fig = ax.figure
    fig.canvas.draw()  # Opens canvas in order to measure real size
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    ax_width_px = bbox.width
    ax_height_px = bbox.height
    ax_largersize_px = max(ax_width_px, ax_height_px)

    ##########################################################################################
    # STEP 2: MAIN TITLE FORMAT
    # MAIN TITLE FORMAT - Wrap title text
    # Calculate fontsize relative to ax width (Previously set fontsize = 12.5)
    fontsize = int(ax_largersize_px / 60)
    # Calculate aprox. number of characters that fit inside half the ax size (In order to wrap text when it reaches half the ax size)
    char_width_px = fontsize * 0.6  # Average size that a character occupies (in pixels)
    max_chars = int((ax_width_px * 0.40) / char_width_px)# (Chose 0.40 instead of 0.50 because the text doesn't start exactly over the left axis)
    # Wrap tittle
    wrapped_title = "\n".join(wrap(plot_title, width=max_chars))
    
    # MAIN TITLE FORMAT - Set title
    ax.text(0.03, 0.98, wrapped_title,
            fontsize=fontsize, 
            fontweight='bold',
            ha='left', va='top', # Text allignment
            transform=ax.transAxes, #ax.tramsAxes sets position relative to axes instead of coordinates
            #bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3') # Create semi-transparent box around text [Canceled]
           )
    
    ##########################################################################################
    # STEP 3: LEGEND TITLE FORMAT   
    # LEGEND TITLE FORMAT - Edit legend text size
    legend_fontsize = int(ax_width_px / 75)
    if legend_type == 'categorized':
        # Modify categorized legend, which is a matplotlib.legend.Legend and can be accessed using ax.get_legend()
        legend = ax.get_legend()
        legend.set_title(legend_title)
        legend.get_title().set_fontsize(legend_fontsize)
        legend.get_title().set_fontweight('bold')
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
    elif legend_type == 'colorbar':
        # Create a divider for the provided ax
        divider = make_axes_locatable(ax)
        # Create an axe for the colorbar (The size will be the "size%" of the main ax, while there's a spacing (pad) between the main ax and the colorbar)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        # Load previously created colorbar and norm
        cmap = cmap_args[0]
        norm = cmap_args[1]
        # Create colorbar manually
        cb = colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm, orientation='horizontal')
        # Set colorbar label manually
        cb.set_label(legend_title, fontsize=legend_fontsize, fontweight='bold')
        cb.ax.tick_params(labelsize=legend_fontsize)
        
    ##########################################################################################
    # STEP 4: GRID FORMAT
    # Turn on grid if required
    if grid:
        ax.grid(color='grey', linestyle='--', linewidth=0.25) 

    ##########################################################################################
    # STEP 5: OBSERVATORY'S LOGO
    # Find current path to project
    from pathlib import Path
    current_path = Path().resolve()
    for parent in current_path.parents:
        if parent.name == "odc":
            project_root = parent
            break
    # Read image from dir starting in project_root
    img_dir = str(project_root)+'/data/external/logo_odc.png'
    img = mpimg.imread(img_dir)
    # Calculate image zoom a) Get current image extent
    img_width_px = img.shape[1]
    # Calculate image zoom c) Calculate required zoom so that image occupies ~10% of larger ax size
    target_fraction = 0.05
    img_zoom = (ax_largersize_px * target_fraction) / img_width_px
    # Insert image on bottom right corner with specified zoom
    # (0,1) --> Upper left
    # (1,1) --> Upper right
    # (0,0) --> Lower left
    # (1,0) --> Lower right
    img_position = (0.98, 0.03)
    img_box = OffsetImage(img, zoom=img_zoom)
    ab = AnnotationBbox(img_box, img_position, frameon=False,
                        xycoords='axes fraction',  # Set coordinates relative to axis (e.g. (0,1))
                        box_alignment=img_position)  # Align image
    # Add image to ax
    ax.add_artist(ab)


def plot_hex_proximity(data_gdf, location_name, ax,
                       column='mean_time',
                       plot_osmnx_edges = (False, ''),
                       plot_boundary = (False, ''),
                       adjust_to = ('',[0.05,0.05]),
                       save_png = (False, '../output/figures/plot.png'),
                       png_transparency = False,
                       png_dpi = 300,
                       save_pdf = (False, '../output/figures/plot.pdf'),
                      ):
    """
    Creates a plot showing proximity analysis data.

    This function can be used to plot a classical proximity analysis (time to amenities), the amenity count (how many amenities 
    can be found on a given walking time) or a sigmodial analysis. The function defaults to mean proximity analysis.
    
    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the proximity analysis data from OdC.
    location_name: str
        Text containing the location (e.g. city name), used to be added in the main plot title.        
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default 'mean_time'.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed and it must contain 'min', 'idx', 'max' or 'time' in order to select the analysis.
        'min' refers to 'minutes' (e.g. column 'cultural_15min', which indicates the average number of kindergardens on a 15 minutes walk by hex)
        'idx' refers to 'sigmodial index' (e.g. column 'idx_preescolar', which indicates the proximity index (using a sigmodial function) for kindergardens)
        'max' refers to 'maximum time' (e.g. columns 'max_preescolar' which indicates time (minutes) data and is categorized in time bins)
        'time' is used in 'min_time', 'mean_time', 'median_time' or 'max_time', and indicates statistical summaries of available amenities.
    plot_osmnx_edges: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1]
        The gdf must contain a boundary (polygon geometry) since the outline will be plotted.
    adjust_to: tupple, default ('',[0.05,0.05]).
        Argument to be passed to function square_bounds().Tupple containing string on position [0]. 
        Passing 'boundary' adjustes zoom to boundary if provided, passing'edges' adjustes zoom to edges if provided, other strings adjustes zoom to data_gdf. 
        The image boundaries will be adjusted to form a square centered on the previously specified gdf.
        Position [1] must contain a list with two numbers. The first will expand the x axis and the second will expand the y axis.
        Example: ('boundary',[0.10,0.05]) sets the bounds of the image as a square around the boundary_gdf from plot_boundary[1] 
        plus a 10% expansion on the x-axis and a 5% expansion on the y-axis.
    save_png: tupple, default (False, '../output/figures/plot.png').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the png is saved.
    png_transparency: bool, default False.
        If True, saves the png with transparency.
    png_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots proximity analysis.
    
    """
    # --------------- GENERAL PLOT STYLE
    data_linestyle = '-'
    data_linewidth = 0.35
    data_edgecolor = 'white'
    
    # --------------- FOR AMENITY AVAILABILITY (COUNT) DATA
    # (e.g. column 'cultural_15min', which indicates the average number of kindergardens on a 15 minutes walk by hex)
    if 'min' in column:
        # --- TITLE
        # Extract amenity name from column name
        amenity_name = column.split('_')[0]
        # Extract time used for availability calculation (normaly 15 minutes) from column name
        time_amount = column.split('_')[1]
        # Create plot title
        plot_title = f"Availability of {amenity_name} amenities on a {time_amount} walk in {location_name.capitalize()}."
        # --- LEGEND - COLOR BAR
        legend_type='colorbar'
        # Define cmap and normalization
        cmap = plt.get_cmap('viridis')
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(ax=ax, column=column, cmap=cmap, legend=False,
                      linestyle=data_linestyle, linewidth=data_linewidth,
                      edgecolor=data_edgecolor, zorder=1)
        # Plot proximity data using the viridis color palette directly 
        # (Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax)
        #data_gdf.plot(ax=ax,column=column,cmap='viridis',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)         
    
    # --------------- FOR PROXIMITY INDEX (IDX) DATA
    # (e.g. column 'idx_preescolar', which indicates the proximity index (using a sigmodial function) for kindergardens)
    elif 'idx' in column:
        # --- TITLE
        # Set plot title
        if column == 'idx_sum': #All-amenities index
            plot_title = f"Proximity index (Sigmodial) for all amenities in {location_name.capitalize()}."
        else: # Index to specific amenity
            # Extract amenity name from column name
            amenity_name = column.split('_')[1]
            plot_title = f"Proximity index (Sigmodial) for {amenity_name} amenities in {location_name.capitalize()}."
        # --- LEGEND - COLOR BAR
        legend_type='colorbar'
        # Define cmap and normalization
        cmap = plt.get_cmap('magma')
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(ax=ax, column=column, cmap=cmap, legend=False,
                      linestyle=data_linestyle, linewidth=data_linewidth,
                      edgecolor=data_edgecolor, zorder=1)
        # Plot proximity data using the magma color palette directly
        # (Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax)
        #data_gdf.plot(ax=ax,column=column,cmap='magma',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)
    
    # --------------- FOR PROXIMITY (TIME) DATA
    # (e.g. columns 'max_preescolar' or 'median_time', which indicates average time (minutes) data and is categorized in time bins)
    elif ('max' in column) or ('time' in column):
        # --- TITLE
        # Set plot title
        if 'time' in column: #Time to all amenities
            # Extract statistical selected (Can be mean_time, median_time or max_time) from column name
            statistical = column.split('_')[0]
            plot_title = f"Proximity analysis ({statistical} time) to all amenities in {location_name.capitalize()}."
        else: # Time to specific amenity
            # Extract amenity name from column name
            amenity_name = column.split('_')[1]
            plot_title = f"Time to {amenity_name} amenities in {location_name.capitalize()}."
        # --- LEGEND - CATEGORIZED
        legend_type='categorized'
        # Categorize time data in time bins
        data_gdf.loc[data_gdf[column]>=60 , f'{column}_cat'] = '60 or more minutes'
        data_gdf.loc[(data_gdf[column]>=45 )&
                     (data_gdf[column]<60), f'{column}_cat'] = '45 minutes to 60 minutes'
        data_gdf.loc[(data_gdf[column]>=30)&
                     (data_gdf[column]<45), f'{column}_cat'] = '30 minutes to 45 minutes'
        data_gdf.loc[(data_gdf[column]>=15)&
                     (data_gdf[column]<30), f'{column}_cat'] = '15 minutes to 30 minutes'
        data_gdf.loc[(data_gdf[column]<15), f'{column}_cat'] = '15 minutes or less'
        # Order data
        categories = ['15 minutes or less','15 minutes to 30 minutes','30 minutes to 45 minutes','45 minutes to 60 minutes','60 or more minutes']
        data_gdf[f'{column}_cat'] = pd.Categorical(data_gdf[f'{column}_cat'], categories=categories, ordered=True)
        # --- PLOT
        # Plot proximity data using the viridis color palette reversed
        data_gdf.plot(ax=ax,column=f'{column}_cat',cmap='viridis_r',legend=True,legend_kwds={'loc':'lower left'},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)

    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(ax=ax,color='#bebebe',linestyle='--',linewidth=0.75,zorder=2)
    # Plot edges if available
    if plot_osmnx_edges[0]:
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main)
        edges_shown_a = ['trunk','trunk_link','motorway','motorway_link']
        edges_gdf_main = edges_gdf[edges_gdf['highway'].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main=True
            edges_gdf_main.plot(ax=ax,color='#000000',alpha=0.5,linewidth=1.0,zorder=3)
        # Plot edges (Primary and secondary)
        edges_shown_b = ['primary','primary_link']#,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf['highway'].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary=True
            edges_gdf_primary.plot(ax=ax,color='#000000',alpha=0.5,linewidth=0.50,zorder=3)
    
	# --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == 'boundary') and (plot_boundary[0]):
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == 'edges') and (plot_osmnx_edges[0]):
        if edges_main and edges_primary:
            square_bounds(ax, pd.concat([edges_gdf_main,edges_gdf_primary]), adjust_to[1])
        elif edges_main:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    if legend_type=='colorbar':
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[cmap,norm],
                                grid=False
                                )
    else:
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[],
                                grid=False
                                )
    
    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        plt.savefig(save_png[1],dpi=png_dpi,transparent=png_transparency)
    if save_pdf[0]:
        plt.savefig(save_pdf[1])


def plot_hex_ndvi(data_gdf, location_name, ax,
                  column='ndvi_mean',
                  plot_osmnx_edges = (False, ''),
                  plot_boundary = (False, ''),
                  adjust_to = ('',[0.05,0.05]),
                  save_png = (False, '../output/figures/plot.png'),
                  png_transparency = False,
                  png_dpi = 300,
                  save_pdf = (False, '../output/figures/plot.pdf'),
                 ):
    """
    Creates a plot showing ndvi analysis data.

    This function can be used to plot NDVI analysis results (data divided in 5 vegetation categories) or NDVI tendency given the available years.
    The function defaults to mean NDVI analysis results.
    
    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    location_name: str
        Text containing the location (e.g. city name), used to be added in the main plot title.        
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default 'ndvi_mean'.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed and it must be 'ndvi_tend' or 'ndvi_{year}' (e.g. 'ndvi_2019') or a statistical summary column.
        Passing 'ndvi_tend' returns a tendency plot using 'ndvi_tend' column
        Passing 'ndvi_{year}' or other statistical summary column (e.g. 'ndvi_mean') assumes the column has NDVI values ranging from -1 to 1 and
        categorizes that information in 5 vegetation categories.
    plot_osmnx_edges: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1]
        The gdf must contain a boundary (polygon geometry) since the outline will be plotted.
    adjust_to: tupple, default ('',[0.05,0.05]).
        Argument to be passed to function square_bounds().Tupple containing string on position [0]. 
        Passing 'boundary' adjustes zoom to boundary if provided, passing'edges' adjustes zoom to edges if provided, other strings adjustes zoom to data_gdf. 
        The image boundaries will be adjusted to form a square centered on the previously specified gdf.
        Position [1] must contain a list with two numbers. The first will expand the x axis and the second will expand the y axis.
        Example: ('boundary',[0.10,0.05]) sets the bounds of the image as a square around the boundary_gdf from plot_boundary[1] 
        plus a 10% expansion on the x-axis and a 5% expansion on the y-axis.
    save_png: tupple, default (False, '../output/figures/plot.png').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the png is saved.
    png_transparency: bool, default False.
        If True, saves the png with transparency.
    png_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots proximity analysis.
    
    """
    # --------------- GENERAL PLOT STYLE
    data_linestyle = '-'
    data_linewidth = 0.35
    data_edgecolor = 'white'
    
    # --------------- FOR TENDENCY (ndvi_tend)
    # (ndvi_tend indicates the data's tendency analysed in all available years)
    if column=='ndvi_tend':
        # --- TITLE
        # Create plot title
        plot_title = f"Tendency of NDVI data in {location_name.capitalize()}."
        # --- LEGEND - COLOR BAR
        legend_type='colorbar'
        # Define cmap and normalization
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        if vmin < 0 < vmax: #Regular case, there are negative and positive values in tendency
            cmap = plt.get_cmap('RdYlGn')
            # Create symmetrical norm scale (Prevents presenting small vmins or vmaxs with dark colors)
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif vmin < vmax < 0: #Non-regular case, there are only negative values in tendency
            cmap = plt.get_cmap('Reds_r')
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else: #Non-regular case, there are only positive values in tendency
            cmap = plt.get_cmap('Greens')
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(ax=ax, column=column,cmap=cmap,vmin=vmin,vmax=vmax,legend=False,
                      linestyle=data_linestyle, linewidth=data_linewidth,
                      edgecolor=data_edgecolor, zorder=1)
        # Plot proximity data using the viridis color palette directly 
        # (Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax)
        #data_gdf.plot(ax=ax,column=column,cmap='viridis',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)         

    # --------------- FOR NDVI VALUES
    # (e.g. columns 'max_preescolar' or 'median_time', which indicates average time (minutes) data and is categorized in time bins)
    else:
        # --- TITLE
        # Set plot title according to column used
        # If the fifth value starts with '2', its a year value. e.g. 'ndvi_2018'
        # Else, its a statistical value, e.g. 'ndvi_mean'
        if column[5] == '2':
            # Extract selected year
            year = column.split('_')[1]
            plot_title = f"NDVI values by hex in {year} in {location_name.capitalize()}."
        else:
            # Extract selected statistical name
            statistical = column.split('_')[1]
            plot_title = f"{statistical.capitalize()} NDVI values by hex in {location_name.capitalize()}."
        
        # --- LEGEND - CATEGORIZED
        legend_type='categorized'
        # Categorize time data in time bins
        data_gdf[f'{column}_cat'] = np.nan
        data_gdf.loc[data_gdf[column]>=0.6 , f'{column}_cat'] = 'Alta densidad vegetal'
        data_gdf.loc[(data_gdf[column]>=0.4 )&
                     (data_gdf[column]<0.6), f'{column}_cat'] = 'Moderada densidad vegetal'
        data_gdf.loc[(data_gdf[column]>=0.2)&
                     (data_gdf[column]<0.4), f'{column}_cat'] = 'Mínima densidad vegetal'
        data_gdf.loc[(data_gdf[column]>=0.1)&
                     (data_gdf[column]<0.2), f'{column}_cat'] = 'Suelo'
        data_gdf.loc[(data_gdf[column]<0.1), f'{column}_cat'] = 'Suelo artificial/Agua/Piedra'
        # Order data
        categories = ['Suelo artificial/Agua/Piedra', 'Suelo', 'Mínima densidad vegetal', 'Moderada densidad vegetal', 'Alta densidad vegetal']
        data_gdf[f'{column}_cat'] = pd.Categorical(data_gdf[f'{column}_cat'], categories=categories, ordered=True)
        # --- PLOT
        # Plot proximity data using the viridis color palette reversed
        data_gdf.plot(ax=ax,column=f'{column}_cat',cmap='YlGn',legend=True,legend_kwds={'loc':'lower left'},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)
    
    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(ax=ax,color='#bebebe',linestyle='--',linewidth=0.75,zorder=2)
    # Plot edges if available
    if plot_osmnx_edges[0]:
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main)
        edges_shown_a = ['trunk','trunk_link','motorway','motorway_link']
        edges_gdf_main = edges_gdf[edges_gdf['highway'].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main=True
            edges_gdf_main.plot(ax=ax,color='#000000',alpha=0.5,linewidth=1.0,zorder=3)
        # Plot edges (Primary and secondary)
        edges_shown_b = ['primary','primary_link']#,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf['highway'].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary=True
            edges_gdf_primary.plot(ax=ax,color='#000000',alpha=0.5,linewidth=0.50,zorder=3)
    
	# --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == 'boundary') and (plot_boundary[0]):
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == 'edges') and (plot_osmnx_edges[0]):
        if edges_main and edges_primary:
            square_bounds(ax, pd.concat([edges_gdf_main,edges_gdf_primary]), adjust_to[1])
        elif edges_main:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    if legend_type=='colorbar':
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[cmap,norm],
                                grid=False
                                )
    else:
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[],
                                grid=False
                                )
    
    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        plt.savefig(save_png[1],dpi=png_dpi,transparent=png_transparency)
    if save_pdf[0]:
        plt.savefig(save_pdf[1])


def plot_hex_temperature(data_gdf, location_name, ax,
                         column='',
                         plot_osmnx_edges = (False, ''),
                         plot_boundary = (False, ''),
                         adjust_to = ('',[0.05,0.05]),
                         save_png = (False, '../output/figures/plot.png'),
                         png_transparency = False,
                         png_dpi = 300,
                         save_pdf = (False, '../output/figures/plot.pdf'),
                        ):
    """
    Creates a plot showing temperature analysis data.

    This function can be used to plot temperature analysis results (data showing hotter or colder areas relative to the overall mean, divided in 7 categories)
    or temperature tendency given the available years. The function defaults to temperature analysis results.
    
    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    location_name: str
        Text containing the location (e.g. city name), used to be added in the main plot title.        
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default ''.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed.
        Passing 'temperature_tend' returns a tendency plot using 'temperature_tend' column
        NOTE: Passing ANY other value assumes that there is a column named 'temperature_mean' from which the difference relative to the overall mean is calculated.
    plot_osmnx_edges: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, '').
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1]
        The gdf must contain a boundary (polygon geometry) since the outline will be plotted.
    adjust_to: tupple, default ('',[0.05,0.05]).
        Argument to be passed to function square_bounds().Tupple containing string on position [0]. 
        Passing 'boundary' adjustes zoom to boundary if provided, passing'edges' adjustes zoom to edges if provided, other strings adjustes zoom to data_gdf. 
        The image boundaries will be adjusted to form a square centered on the previously specified gdf.
        Position [1] must contain a list with two numbers. The first will expand the x axis and the second will expand the y axis.
        Example: ('boundary',[0.10,0.05]) sets the bounds of the image as a square around the boundary_gdf from plot_boundary[1] 
        plus a 10% expansion on the x-axis and a 5% expansion on the y-axis.
    save_png: tupple, default (False, '../output/figures/plot.png').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the png is saved.
    png_transparency: bool, default False.
        If True, saves the png with transparency.
    png_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1] 
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots proximity analysis.
    
    """
    # --------------- GENERAL PLOT STYLE
    data_linestyle = '-'
    data_linewidth = 0.35
    data_edgecolor = 'white'
    
    # --------------- FOR TENDENCY (temperature_tend)
    # (temperature_tend indicates the data's tendency analysed in all available years)
    if column=='temperature_tend':
        # --- TITLE
        # Create plot title
        plot_title = f"Tendency of temperature data in {location_name.capitalize()}."
        # --- LEGEND - COLOR BAR
        legend_type='colorbar'
        # Define cmap and normalization
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        # Define norm case
        if vmin < 0 < vmax: #Regular case, there are negative and positive values in tendency
            cmap = plt.get_cmap('RdBu_r')
            # Create symmetrical norm scale (Prevents presenting small vmins or vmaxs with dark colors)
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif vmin < vmax < 0: #Non-regular case, there are only negative values in tendency
            cmap = plt.get_cmap('Blues_r')
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else: #Non-regular case, there are only positive values in tendency
            cmap = plt.get_cmap('Reds')
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(ax=ax, column=column, cmap=cmap, vmin=vmin, vmax=vmax, legend=False,
                      linestyle=data_linestyle, linewidth=data_linewidth,
                      edgecolor=data_edgecolor, zorder=1) 

    # --------------- FOR ANOMALY (temperature_anomaly)
    # (temperature_anomaly indicates the difference between the temperature_mean of each hex and the overall (city) mean.
    else:
        # Rewrite column, no other values are permitted.
        column='temperature_anomaly'
        # --- TITLE
        # Set plot title according to column used
        plot_title = f"Temperature difference between each hex's mean and overall mean in {location_name.capitalize()}."
        # --- LEGEND - CATEGORIZED
        legend_type='categorized'
        # Calculate anomaly (difference between mean in each hex and city's mean)
        mean_city_temperature = data_gdf.temperature_mean.mean()
        data_gdf['temperature_anomaly'] = data_gdf['temperature_mean'] - mean_city_temperature
        # Categorize anomaly
        classif_bins = [-100,-3.5,-1.5,-0.5,0.5,1.5,3.5,100]
        data_gdf['anomaly_class'] = pd.cut(data_gdf['temperature_anomaly'],
                                           bins=classif_bins,
                                           labels=[-3, -2, -1, 0, 1, 2, 3],
                                           include_lowest=True).astype(int)
        classes_dict = {3:"1. More temperature",
                        2:"2.",
                        1:"3.",
                        0:"4. Near overall mean",
                        -1:"5.",
                        -2:"6.",
                        -3:f"7. Less temperature"
                       }
        data_gdf['anomaly_bins'] = data_gdf['anomaly_class'].map(classes_dict)
        # Define order and convert col into ordered category
        categories = list(classes_dict.values())
        data_gdf['anomaly_bins'] = pd.Categorical(data_gdf['anomaly_bins'],
                                                  categories=categories,
                                                  ordered=True)
        # Force categorical order
        data_gdf.sort_values(by='anomaly_bins', inplace=True)
        # --- PLOT
        # Plot temperature anomaly data using the viridis color palette reversed
        data_gdf.plot(ax=ax,column='anomaly_bins',cmap='RdBu',legend=True,legend_kwds={'loc':'lower left'},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)
    
    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(ax=ax,color='#bebebe',linestyle='--',linewidth=0.75,zorder=2)
    # Plot edges if available
    if plot_osmnx_edges[0]:
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main)
        edges_shown_a = ['trunk','trunk_link','motorway','motorway_link']
        edges_gdf_main = edges_gdf[edges_gdf['highway'].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main=True
            edges_gdf_main.plot(ax=ax,color='#000000',alpha=0.5,linewidth=1.0,zorder=3)
        # Plot edges (Primary and secondary)
        edges_shown_b = ['primary','primary_link']#,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf['highway'].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary=True
            edges_gdf_primary.plot(ax=ax,color='#000000',alpha=0.5,linewidth=0.50,zorder=3)
    
	# --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == 'boundary') and (plot_boundary[0]):
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == 'edges') and (plot_osmnx_edges[0]):
        if edges_main and edges_primary:
            square_bounds(ax, pd.concat([edges_gdf_main,edges_gdf_primary]), adjust_to[1])
        elif edges_main:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    if legend_type=='colorbar':
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[cmap,norm],
                                grid=False
                                )
    else:
        observatory_plot_format(ax=ax,
                                plot_title=plot_title,
                                legend_title = f'Column: {column}.',
                                legend_type=legend_type,
                                cmap_args=[],
                                grid=False
                                )
    
    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        plt.savefig(save_png[1],dpi=png_dpi,transparent=png_transparency)
    if save_pdf[0]:
        plt.savefig(save_pdf[1])