################################################################################
# Module: Visualization
# General visualization functions, mainly focused on kepler configurations
# updated: 10/08/2023
################################################################################

import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("bottom", size="5%", pad=0.1)
	gdf_data[gdf_data[column]<=0].plot(ax=ax,color='#2b2b2b', alpha=0.95, linewidth=0.1, edgecolor='k', zorder=0)
	gdf_data[gdf_data[column]>0].plot(ax=ax,column=column, cmap='magma_r',vmax=1000,zorder=1,legend=True,cax=cax,legend_kwds={'label':'Distancia (m)','orientation': "horizontal"})
	gdf_boundary.boundary.plot(ax=ax,color='#f8f8f8',zorder=2,linestyle='--',linewidth=0.5)
	gdf_edges[(gdf_edges['highway']=='motorway') | (gdf_edges['highway']=='motorway_link')].plot(ax=ax,color='#898989',alpha=0.5,linewidth=2.5,zorder=3)
	gdf_edges[(gdf_edges['highway']=='primary') | (gdf_edges['highway']=='primary_link')].plot(ax=ax,color='#898989',alpha=0.5,linewidth=1.5,zorder=3)
	ax.set_title(f'{title}',fontdict={'fontsize':30})
	ax.axis('off')
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