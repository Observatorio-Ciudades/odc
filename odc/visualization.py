################################################################################
# Module: Visualization
# General visualization functions for specific plotting and Kepler configurations
# updated: 15/08/2025
################################################################################

# Wrap title text inside plots
from textwrap import wrap

import geopandas as gpd

# Import cm, colors and colorbar to create colormap legends manually
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as colors

# Import logo image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# For function documentation
from matplotlib.axes import Axes
from matplotlib.colors import (
    TwoSlopeNorm,  # Tendency colorbar with both positive and negative values
)

# Place logo image above plot
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .data import *
from .utils import *


def square_bounds(
    ax: Axes, bounds_gdf: gpd.GeoDataFrame, boundary_expansion=[0.05, 0.05]
) -> None:
    """
    Formats an ax's boundaries in order to set squared bounds around a given GeoDataFrame's geometry inside a plot.

    This function takes an ax and modifies the its boundaries to set a squared boundary
    around the geometry of a given GeoDataFrame. These bounds can be expanded or contracted
    using values from boundary_expansion. This expansion is applied symmetrically, keeping the
    bounds_gdf centered.

    Parameters
    ----------
    ax: matplotlib.axes
        ax to be modified using ax.set_xlim() and ax.set_ylim().
    bounds_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the geometry from which an initial bounding box is created.
    boundary_expansion: list, default [0.05,0.05].
        List with two numeric values that represent percentages used to expand the squared-boundary plot symmetrically.
        If no expansion is required, use [0,0], but note that the bounds will be very tight around the bounds_gdf.
        The first digit expands the x-axis and the second digit expands the y-axis.
        e.g. [0.05,0.10] would expand by 5% the horizontal bounds (2.5% to the left, 2.5% to the right)
        and by 10% the vertical bounds (5% to the bottom, 5% to the top), thereby plotting a rectangle with a larger x-axis than y-axis.

    Returns
    -------
    None, modifies ax.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.
    ValueError
        If boundary_expansion is not a list with two numeric values.

    """
    # Input validation
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if bounds_gdf is None or not isinstance(bounds_gdf, gpd.GeoDataFrame):
        raise TypeError("bounds_gdf must be a geopandas.GeoDataFrame instance.")
    if (
        not isinstance(boundary_expansion, list)
        or len(boundary_expansion) != 2
        or (not all(isinstance(i, (int, float)) for i in boundary_expansion))
    ):
        raise ValueError("boundary_expansion must be a list with two numeric values.")

    # Calculate bounds_gdf's current bounding box
    minx, miny, maxx, maxy = bounds_gdf.total_bounds

    # Find bounding box's size difference (larger side minus smaller side)
    # in order to create a centered square bounding box
    x_dist = abs(maxx - minx)
    y_dist = abs(maxy - miny)
    larger_side = max(x_dist, y_dist)
    smaller_side = min(x_dist, y_dist)
    size_diff = larger_side - smaller_side

    # Find squeared bounding limits
    if size_diff == 0:  # If it is already a square, no need to calculate expansion
        square_minx = minx
        square_maxx = maxx
        square_miny = miny
        square_maxy = maxy
    else:  # Else, enlarge shorter side
        if x_dist > y_dist:  # (Here <x> is larger size and <y> is enlarged)
            square_minx = minx
            square_maxx = maxx
            square_miny = miny - (size_diff / 2)
            square_maxy = maxy + (size_diff / 2)
        else:  # (Here <x> is enlarged and <y> is larger size)
            square_minx = minx - (size_diff / 2)
            square_maxx = maxx + (size_diff / 2)
            square_miny = miny
            square_maxy = maxy

    # Use additional boundary_expansion values to expand the square bounds as requested
    expanded_square_minx = square_minx - (larger_side * boundary_expansion[0]) / 2
    expanded_square_maxx = square_maxx + (larger_side * boundary_expansion[0]) / 2
    expanded_square_miny = square_miny - (larger_side * boundary_expansion[1]) / 2
    expanded_square_maxy = square_maxy + (larger_side * boundary_expansion[1]) / 2

    # Adjust ax limits accordingly
    ax.set_xlim(expanded_square_minx, expanded_square_maxx)
    ax.set_ylim(expanded_square_miny, expanded_square_maxy)


def observatory_plot_format(
    ax: Axes,
    plot_title: str,
    legend_title: str,
    legend_type: str,
    cmap_args=[],
    grid=False,
) -> None:
    """
    Formats the plot to the observatory's style.

    This function positions the title on the top left corner of the plot, formats the legend title in bold,
    and places OdC's logo on the bottom right corner of the plot.

    NOTE: This function must be used after plotting all elements on it a given ax.
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
        - The first element must be the previously generated colormap (type plt.colormaps.Colormap, which can be obtained using plt.get_cmap(''))
        - The second element must be the previously generated norm colors (type colors.Normalize, which can be obtained using colors.Normalize())
    grid: bool, default False.
        If true turns on coordinates grid.

    Returns
    -------
    None, modifies ax.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.
    ValueError
        If legend_type is not 'categorized' or 'colorbar', or if cmap_args does not contain the required elements for a colorbar.

    """
    # Input validation
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(plot_title, str):
        raise TypeError("plot_title must be a string.")
    if not isinstance(legend_title, str):
        raise TypeError("legend_title must be a string.")
    if legend_type not in ["categorized", "colorbar"]:
        raise ValueError("legend_type must be either 'categorized' or 'colorbar'.")
    if legend_type == "colorbar" and (
        len(cmap_args) != 2
        or not isinstance(cmap_args[0], colors.Colormap)
        or not isinstance(cmap_args[1], colors.Normalize)
    ):
        raise ValueError(
            "If using legent_type='colorbar', cmap_args must contain two elements: a colormap and a norm. The first element must be a matplotlib colormap and the second a matplotlib colors.Normalize instance."
        )
    if not isinstance(grid, bool):
        raise TypeError("grid must be a boolean value (True or False).")

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
    max_chars = int(
        (ax_width_px * 0.40) / char_width_px
    )  # (Chose 0.40 instead of 0.50 because the text doesn't start exactly over the left axis)
    # Wrap tittle
    wrapped_title = "\n".join(wrap(plot_title, width=max_chars))

    # MAIN TITLE FORMAT - Set title
    ax.text(
        0.03,
        0.98,
        wrapped_title,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="top",  # Text allignment
        transform=ax.transAxes,  # ax.tramsAxes sets position relative to axes instead of coordinates
        # bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, boxstyle='round,pad=0.3') # Create semi-transparent box around text [Canceled]
    )

    ##########################################################################################
    # STEP 3: LEGEND TITLE FORMAT
    # LEGEND TITLE FORMAT - Edit legend text size
    legend_fontsize = int(ax_width_px / 75)
    if legend_type == "categorized":
        # Modify categorized legend, which is a matplotlib.legend.Legend and can be accessed using ax.get_legend()
        legend = ax.get_legend()
        legend.set_title(legend_title)
        legend.get_title().set_fontsize(legend_fontsize)
        legend.get_title().set_fontweight("bold")
        for text in legend.get_texts():
            text.set_fontsize(legend_fontsize)
    elif legend_type == "colorbar":
        # Create colorbar manually, which is a matplotlib.colorbar.ColorbarBase
        # Create a divider for the provided ax
        divider = make_axes_locatable(ax)
        # Create an axe for the colorbar (The size will be the "size%" of the main ax, while there's a spacing (pad) between the main ax and the colorbar)
        cax = divider.append_axes("bottom", size="5%", pad=0.1)
        # Load previously created colorbar and norm
        cmap = cmap_args[0]
        norm = cmap_args[1]
        # Create colorbar manually
        cb = colorbar.ColorbarBase(
            ax=cax, cmap=cmap, norm=norm, orientation="horizontal"
        )
        # Set colorbar label manually
        cb.set_label(legend_title, fontsize=legend_fontsize, fontweight="bold")
        cb.ax.tick_params(labelsize=legend_fontsize)

    ##########################################################################################
    # STEP 4: GRID FORMAT
    # Turn on grid if required
    if grid:
        ax.grid(color="grey", linestyle="--", linewidth=0.25)

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
    img_dir = str(project_root) + "/data/external/logo_odc.png"
    img = mpimg.imread(img_dir)
    # Calculate image zoom a) Get current image extent
    img_width_px = img.shape[1]
    # Calculate image zoom b) Calculate required zoom so that image occupies ~10% of larger ax size
    target_fraction = 0.05
    img_zoom = (ax_largersize_px * target_fraction) / img_width_px
    # Insert image on bottom right corner with specified zoom
    # (0,1) --> Upper left
    # (1,1) --> Upper right
    # (0,0) --> Lower left
    # (1,0) --> Lower right
    img_position = (0.98, 0.03)
    img_box = OffsetImage(img, zoom=img_zoom)
    ab = AnnotationBbox(
        img_box,
        img_position,
        frameon=False,
        xycoords="axes fraction",  # Set coordinates relative to axis (e.g. (0,1))
        box_alignment=img_position,
    )  # Align image
    # Add image to ax
    ax.add_artist(ab)


def plot_proximity(
    data_gdf: gpd.GeoDataFrame,
    ax: Axes,
    column="mean_time",
    location_name="",
    plot_osmnx_edges=(False, gpd.GeoDataFrame),
    plot_boundary=(False, gpd.GeoDataFrame),
    adjust_to=("", [0.05, 0.05]),
    save_png=(False, "../output/figures/plot.png"),
    output_transparency=False,
    output_dpi=300,
    save_pdf=(False, "../output/figures/plot.pdf"),
) -> None:
    """
    Creates a plot showing proximity analysis data.

    This function can be used to plot a classical proximity analysis (time to amenities), the amenity count (how many amenities
    can be found on a given walking time) or a sigmodial analysis. The function defaults to mean proximity analysis (using the 'mean_time' column).

    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the proximity analysis data from OdC.
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default 'mean_time'.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed and it must contain 'min', 'idx', 'max' or 'time' in order to select the analysis.
        'min' refers to 'minutes' (e.g. column 'cultural_15min', which indicates the average number of kindergardens on a 15 minutes walk)
        'idx' refers to 'sigmodial index' (e.g. column 'idx_preescolar', which indicates the proximity index (using a sigmodial function) for kindergardens)
        'max' refers to 'maximum time' (e.g. columns 'max_preescolar' which indicates time (minutes) data and is categorized in time bins)
        'time' is used in 'min_time', 'mean_time', 'median_time' or 'max_time', and indicates statistical summaries of available amenities.
    location_name: str, default ''.
        Text containing the location (e.g. city name), used to be added in the main plot title.
        If not provided, the title will not contain a location name.
    plot_osmnx_edges: tupple, default (False, gpd.GeoDataFrame).
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, gpd.GeoDataFrame).
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
    output_transparency: bool, default False.
        If True, saves the output with transparency.
    output_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1]
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots proximity analysis.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.

    """
    # Input validation
    log("plot_proximity() - Validating inputs.")
    if not isinstance(data_gdf, gpd.GeoDataFrame):
        raise TypeError("data_gdf must be a geopandas.GeoDataFrame instance.")
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(column, str):
        raise TypeError("column must be a string.")
    if not isinstance(location_name, str):
        raise TypeError("location_name must be a string.")
    # Input validation for plot_osmnx_edges tupple
    if not isinstance(plot_osmnx_edges, tuple) or len(plot_osmnx_edges) != 2:
        raise TypeError(
            "plot_osmnx_edges must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_osmnx_edges[0]:
        if not isinstance(plot_osmnx_edges[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_osmnx_edges[0] is True, plot_osmnx_edges[1] must be a GeoDataFrame."
            )
        if not all(
            plot_osmnx_edges[1].geometry.type.isin(["LineString", "MultiLineString"])
        ):
            raise TypeError(
                "The plot_osmnx_edges[1] GeoDataFrame must contain only LineString or MultiLineString geometries."
            )
    # Input validation for plot_boundary tupple
    if not isinstance(plot_boundary, tuple) or len(plot_boundary) != 2:
        raise TypeError(
            "plot_boundary must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_boundary[0]:
        if not isinstance(plot_boundary[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_boundary[0] is True, plot_boundary[1] must be a GeoDataFrame."
            )
        if not all(plot_boundary[1].geometry.type.isin(["Polygon", "MultiPolygon"])):
            raise TypeError(
                "The plot_boundary[1] GeoDataFrame must contain only Polygon or MultiPolygon geometries."
            )
    if (
        not isinstance(adjust_to, tuple)
        or len(adjust_to) != 2
        or (not all(isinstance(i, (int, float)) for i in adjust_to[1]))
    ):
        raise TypeError(
            "adjust_to must be a tuple with two elements: a string (either 'boundary', 'edges' or '') and a list with two numeric values."
        )
    if (
        not isinstance(save_png, tuple)
        or len(save_png) != 2
        or not isinstance(save_png[1], str)
    ):
        raise TypeError(
            "save_png must be a tuple with two elements: a boolean and a string (file path)."
        )
    if not isinstance(output_transparency, bool):
        raise TypeError("output_transparency must be a boolean value (True or False).")
    if not isinstance(output_dpi, int):
        raise TypeError("output_dpi must be an integer value.")
    if (
        not isinstance(save_pdf, tuple)
        or len(save_pdf) != 2
        or not isinstance(save_pdf[1], str)
    ):
        raise TypeError(
            "save_pdf must be a tuple with two elements: a boolean and a string (file path)."
        )

    # --------------- DATA PLOT STYLE
    data_linestyle = "-"
    data_linewidth = 0.35
    data_edgecolor = "white"

    # --------------- FOR AMENITY AVAILABILITY (COUNT) DATA
    # (e.g. column 'cultural_15min', which indicates the average number of kindergardens on a 15 minutes walk)
    if "min" in column:
        log(
            "plot_proximity() - Identified 'min' in column. Creating plot for amenity availability (count) data."
        )
        # --- TITLE
        # Extract amenity name from column name
        amenity_name = column.split("_")[0]
        # Extract time used for availability calculation (normaly 15 minutes) from column name
        time_amount = column.split("_")[1]
        # Create plot title
        plot_title = f"Availability of {amenity_name} amenities on a {time_amount} walk"  # Whithout period at the end to add location name if provided
        # --- LEGEND - COLOR BAR
        legend_type = "colorbar"
        # Define cmap and normalization
        cmap = plt.get_cmap("viridis")
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            legend=False,
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )
        # Plot proximity data using the viridis color palette directly
        # [Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax]
        # data_gdf.plot(ax=ax,column=column,cmap='viridis',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)

    # --------------- FOR PROXIMITY INDEX (IDX) DATA
    # (e.g. column 'idx_preescolar', which indicates the proximity index (using a sigmodial function) for kindergardens)
    elif "idx" in column:
        log(
            "plot_proximity() - Identified 'idx' in column. Creating plot for proximity index (sigmodial) data."
        )
        # --- TITLE
        # Set plot title
        if column == "idx_sum":  # All-amenities index
            plot_title = f"Proximity index (Sigmodial) for all amenities"  # Whithout period at the end to add location name if provided
        else:  # Index to specific amenity
            # Extract amenity name from column name
            amenity_name = column.split("_")[1]
            plot_title = f"Proximity index (Sigmodial) for {amenity_name} amenities"  # Whithout period at the end to add location name if provided
        # --- LEGEND - COLOR BAR
        legend_type = "colorbar"
        # Define cmap and normalization
        cmap = plt.get_cmap("magma")
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            legend=False,
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )
        # Plot proximity data using the magma color palette directly
        # [Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax]
        # data_gdf.plot(ax=ax,column=column,cmap='magma',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)

    # --------------- FOR PROXIMITY (TIME) DATA
    # (e.g. columns 'max_preescolar' or 'median_time', which indicates average time (minutes) data and is categorized in time bins)
    elif ("max" in column) or ("time" in column):
        log(
            "plot_proximity() - Identified 'time' or 'max' in column. Creating plot for proximity (time) data."
        )
        # --- TITLE
        # Set plot title
        if "time" in column:  # Time to all amenities
            # Extract statistical selected (Can be mean_time, median_time or max_time) from column name
            statistical = column.split("_")[0]
            if statistical == "time":  # e.g. 'time_schools', a specific amenity
                plot_title = f"Proximity analysis (time) to all amenities"  # Whithout period at the end to add location name if provided
            else:
                plot_title = f"Proximity analysis ({statistical} time) to all amenities"  # Whithout period at the end to add location name if provided
        else:  # Time to specific amenity
            # Extract amenity name from column name
            amenity_name = column.split("_")[1]
            plot_title = f"Time to {amenity_name} amenities"  # Whithout period at the end to add location name if provided
        # --- LEGEND - CATEGORIZED
        legend_type = "categorized"
        # Categorize time data in time bins
        data_gdf.loc[f"{column}_cat"] = ""
        data_gdf.loc[data_gdf[column] >= 60, f"{column}_cat"] = "60 or more minutes"
        data_gdf.loc[
            (data_gdf[column] >= 45) & (data_gdf[column] < 60), f"{column}_cat"
        ] = "45 minutes to 60 minutes"
        data_gdf.loc[
            (data_gdf[column] >= 30) & (data_gdf[column] < 45), f"{column}_cat"
        ] = "30 minutes to 45 minutes"
        data_gdf.loc[
            (data_gdf[column] >= 15) & (data_gdf[column] < 30), f"{column}_cat"
        ] = "15 minutes to 30 minutes"
        data_gdf.loc[(data_gdf[column] < 15), f"{column}_cat"] = "15 minutes or less"
        # Order data
        categories = [
            "15 minutes or less",
            "15 minutes to 30 minutes",
            "30 minutes to 45 minutes",
            "45 minutes to 60 minutes",
            "60 or more minutes",
        ]
        data_gdf[f"{column}_cat"] = pd.Categorical(
            data_gdf[f"{column}_cat"], categories=categories, ordered=True
        )
        # --- PLOT
        # Plot proximity data using the viridis color palette reversed
        data_gdf.plot(
            ax=ax,
            column=f"{column}_cat",
            cmap="viridis_r",
            legend=True,
            legend_kwds={"loc": "lower left"},
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )

    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        log("plot_proximity() - Plotting area of interest's boundary provided.")
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(
            ax=ax, color="#bebebe", linestyle="--", linewidth=0.75, zorder=2
        )
    # Plot edges if available
    if plot_osmnx_edges[0]:
        log("plot_proximity() - Plotting Open Street Map edges provided.")
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main, relevant on a regional scale)
        edges_shown_a = ["trunk", "trunk_link", "motorway", "motorway_link"]
        edges_gdf_main = edges_gdf[edges_gdf["highway"].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main_available = True
            edges_gdf_main.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=1.0, zorder=3
            )
        # Plot edges (Other relevant on a city scale)
        edges_shown_b = ["primary", "primary_link"]  # ,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf["highway"].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary_available = True
            edges_gdf_primary.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=0.50, zorder=3
            )

    # --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == "boundary") and (plot_boundary[0]):
        log("plot_proximity() - Adjusting ax size to boundary provided.")
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == "edges") and (plot_osmnx_edges[0]):
        log("plot_proximity() - Adjusting ax size to edges provided.")
        if edges_main_available and edges_primary_available:
            square_bounds(
                ax, pd.concat([edges_gdf_main, edges_gdf_primary]), adjust_to[1]
            )
        elif edges_main_available:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary_available:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            log(
                "plot_proximity() - No edges available to adjust ax size to. Using data_gdf instead."
            )
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        log("plot_proximity() - Adjusting ax size to data_gdf.")
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    # Add location name to title if provided
    if location_name == "":
        plot_title = plot_title + "."
    else:
        plot_title = plot_title + f" in {location_name.capitalize()}."
    # Call observatory_plot_format() function
    log("plot_proximity() - Formatting plot to observatory's style.")
    if legend_type == "colorbar":
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[cmap, norm],
            grid=False,
        )
    else:
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[],
            grid=False,
        )

    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        log(f"plot_proximity() - Saving plot as PNG in {save_png[1]}.")
        plt.savefig(save_png[1], dpi=output_dpi, transparent=output_transparency)
    if save_pdf[0]:
        log(f"plot_proximity() - Saving plot as PDF in {save_pdf[1]}.")
        pdf_name = save_pdf[1]
        if pdf_name != ".pdf":
            pdf_name = pdf_name + ".pdf"
        plt.savefig(
            pdf_name, format="pdf", dpi=output_dpi, transparent=output_transparency
        )


def plot_ndvi(
    data_gdf: gpd.GeoDataFrame,
    ax: Axes,
    column="ndvi_mean",
    location_name="",
    plot_osmnx_edges=(False, gpd.GeoDataFrame),
    plot_boundary=(False, gpd.GeoDataFrame),
    adjust_to=("", [0.05, 0.05]),
    save_png=(False, "../output/figures/plot.png"),
    output_transparency=False,
    output_dpi=300,
    save_pdf=(False, "../output/figures/plot.pdf"),
) -> None:
    """
    Creates a plot showing ndvi analysis data.

    This function can be used to plot NDVI analysis results (data divided in 5 vegetation categories) or NDVI tendency given the available years.
    The function defaults to mean NDVI analysis results (using the 'ndvi_mean' column).

    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default 'ndvi_mean'.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed and it must be 'ndvi_tend' or 'ndvi_{year}' (e.g. 'ndvi_2019') or a statistical summary column.
        Passing 'ndvi_tend' returns a tendency plot using 'ndvi_tend' column
        Passing 'ndvi_{year}' or other statistical summary column (e.g. 'ndvi_mean') assumes the column has NDVI values ranging from -1 to 1 and
        categorizes that information in 5 vegetation categories.
    location_name: str, default ''.
        Text containing the location (e.g. city name), used to be added in the main plot title.
        If not provided, the title will not contain a location name.
    plot_osmnx_edges: tupple, default (False, gpd.GeoDataFrame).
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, gpd.GeoDataFrame).
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
    output_transparency: bool, default False.
        If True, saves the output with transparency.
    output_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1]
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots ndvi analysis.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.

    """
    # Input validation
    log("plot_proximity() - Validating inputs.")
    if not isinstance(data_gdf, gpd.GeoDataFrame):
        raise TypeError("data_gdf must be a geopandas.GeoDataFrame instance.")
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(column, str):
        raise TypeError("column must be a string.")
    if not isinstance(location_name, str):
        raise TypeError("location_name must be a string.")
    # Input validation for plot_osmnx_edges tupple
    if not isinstance(plot_osmnx_edges, tuple) or len(plot_osmnx_edges) != 2:
        raise TypeError(
            "plot_osmnx_edges must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_osmnx_edges[0]:
        if not isinstance(plot_osmnx_edges[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_osmnx_edges[0] is True, plot_osmnx_edges[1] must be a GeoDataFrame."
            )
        if not all(
            plot_osmnx_edges[1].geometry.type.isin(["LineString", "MultiLineString"])
        ):
            raise TypeError(
                "The plot_osmnx_edges[1] GeoDataFrame must contain only LineString or MultiLineString geometries."
            )
    # Input validation for plot_boundary tupple
    if not isinstance(plot_boundary, tuple) or len(plot_boundary) != 2:
        raise TypeError(
            "plot_boundary must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_boundary[0]:
        if not isinstance(plot_boundary[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_boundary[0] is True, plot_boundary[1] must be a GeoDataFrame."
            )
        if not all(plot_boundary[1].geometry.type.isin(["Polygon", "MultiPolygon"])):
            raise TypeError(
                "The plot_boundary[1] GeoDataFrame must contain only Polygon or MultiPolygon geometries."
            )
    if (
        not isinstance(adjust_to, tuple)
        or len(adjust_to) != 2
        or (not all(isinstance(i, (int, float)) for i in adjust_to[1]))
    ):
        raise TypeError(
            "adjust_to must be a tuple with two elements: a string (either 'boundary', 'edges' or '') and a list with two numeric values."
        )
    if (
        not isinstance(save_png, tuple)
        or len(save_png) != 2
        or not isinstance(save_png[1], str)
    ):
        raise TypeError(
            "save_png must be a tuple with two elements: a boolean and a string (file path)."
        )
    if not isinstance(output_transparency, bool):
        raise TypeError("output_transparency must be a boolean value (True or False).")
    if not isinstance(output_dpi, int):
        raise TypeError("output_dpi must be an integer value.")
    if (
        not isinstance(save_pdf, tuple)
        or len(save_pdf) != 2
        or not isinstance(save_pdf[1], str)
    ):
        raise TypeError(
            "save_pdf must be a tuple with two elements: a boolean and a string (file path)."
        )

    # --------------- DATA PLOT STYLE
    data_linestyle = "-"
    data_linewidth = 0.35
    data_edgecolor = "white"

    # --------------- FOR TENDENCY (ndvi_tend)
    # (ndvi_tend indicates the data's tendency analysed in all available years)
    if column == "ndvi_tend":
        log(
            "plot_ndvi() - Identified 'ndvi_tend' in column. Creating plot for NDVI tendency data."
        )
        # --- TITLE
        # Create plot title
        plot_title = f"Tendency of NDVI data"  # Whithout period at the end to add location name if provided
        # --- LEGEND - COLOR BAR
        legend_type = "colorbar"
        # Define cmap and normalization
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        if (
            vmin < 0 < vmax
        ):  # Regular case, there are negative and positive values in tendency
            cmap = plt.get_cmap("RdYlGn")
            # Create symmetrical norm scale (Prevents presenting small vmins or vmaxs with dark colors)
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif (
            vmin < vmax < 0
        ):  # Non-regular case, there are only negative values in tendency
            cmap = plt.get_cmap("Reds_r")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:  # Non-regular case, there are only positive values in tendency
            cmap = plt.get_cmap("Greens")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=False,
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )
        # Plot proximity data using the viridis color palette directly
        # [Canceled since setting legend=True doesn't allow for label size and formatting modification. Would require to divide ax previously to get cax]
        # data_gdf.plot(ax=ax,column=column,cmap='viridis',legend=True,cax=cax,legend_kwds={'label':f"Column: {column}.",'orientation': "horizontal"},linestyle=data_linestyle,linewidth=data_linewidth,edgecolor=data_edgecolor,zorder=1)

    # --------------- FOR NDVI VALUES
    # (e.g. columns 'max_preescolar' or 'median_time', which indicates average time (minutes) data and is categorized in time bins)
    else:
        log("plot_ndvi() - Creating plot for NDVI values data.")
        # --- TITLE
        # Set plot title according to column used
        # If the fifth value starts with '2', its a year value. e.g. 'ndvi_2018'
        # Else, its a statistical value, e.g. 'ndvi_mean'
        if column[5] == "2":
            # Extract selected year
            year = column.split("_")[1]
            plot_title = f"NDVI values in {year}"  # Whithout period at the end to add location name if provided
        else:
            # Extract selected statistical name
            statistical = column.split("_")[1]
            plot_title = f"{statistical.capitalize()} NDVI values"  # Whithout period at the end to add location name if provided

        # --- LEGEND - CATEGORIZED
        legend_type = "categorized"
        # Categorize time data in time bins
        data_gdf[f"{column}_cat"] = ""
        data_gdf.loc[data_gdf[column] >= 0.6, f"{column}_cat"] = (
            "High vegetation density"
        )
        data_gdf.loc[
            (data_gdf[column] >= 0.4) & (data_gdf[column] < 0.6), f"{column}_cat"
        ] = "Moderate vegetation density"
        data_gdf.loc[
            (data_gdf[column] >= 0.2) & (data_gdf[column] < 0.4), f"{column}_cat"
        ] = "Low vegetation density"
        data_gdf.loc[
            (data_gdf[column] >= 0.1) & (data_gdf[column] < 0.2), f"{column}_cat"
        ] = "Bare soil"
        data_gdf.loc[(data_gdf[column] < 0.1), f"{column}_cat"] = (
            "Artificial surface/Water/Rock"
        )
        # Order data
        categories = [
            "Artificial surface/Water/Rock",
            "Bare soil",
            "Low vegetation density",
            "Moderate vegetation density",
            "High vegetation density",
        ]
        data_gdf[f"{column}_cat"] = pd.Categorical(
            data_gdf[f"{column}_cat"], categories=categories, ordered=True
        )
        # --- PLOT
        # Plot proximity data using the viridis color palette reversed
        data_gdf.plot(
            ax=ax,
            column=f"{column}_cat",
            cmap="YlGn",
            legend=True,
            legend_kwds={"loc": "lower left"},
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )

    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        log("plot_ndvi() - Plotting area of interest's boundary provided.")
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(
            ax=ax, color="#bebebe", linestyle="--", linewidth=0.75, zorder=2
        )
    # Plot edges if available
    if plot_osmnx_edges[0]:
        log("plot_ndvi() - Plotting Open Street Map edges provided.")
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main, relevant on a regional scale)
        edges_shown_a = ["trunk", "trunk_link", "motorway", "motorway_link"]
        edges_gdf_main = edges_gdf[edges_gdf["highway"].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main_available = True
            edges_gdf_main.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=1.0, zorder=3
            )
        # Plot edges (Other relevant on a city scale)
        edges_shown_b = ["primary", "primary_link"]  # ,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf["highway"].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary_available = True
            edges_gdf_primary.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=0.50, zorder=3
            )

    # --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == "boundary") and (plot_boundary[0]):
        log("plot_ndvi() - Adjusting ax size to boundary provided.")
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == "edges") and (plot_osmnx_edges[0]):
        log("plot_ndvi() - Adjusting ax size to edges provided.")
        if edges_main_available and edges_primary_available:
            square_bounds(
                ax, pd.concat([edges_gdf_main, edges_gdf_primary]), adjust_to[1]
            )
        elif edges_main_available:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary_available:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            log(
                "plot_ndvi() - No edges available to adjust ax size to. Using data_gdf instead."
            )
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        log("plot_ndvi() - Adjusting ax size to data_gdf.")
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    # Add location name to title if provided
    if location_name == "":
        plot_title = plot_title + "."
    else:
        plot_title = plot_title + f" in {location_name.capitalize()}."
    # Call observatory_plot_format() function
    log("plot_ndvi() - Formatting plot to observatory's style.")
    if legend_type == "colorbar":
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[cmap, norm],
            grid=False,
        )
    else:
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[],
            grid=False,
        )

    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        log(f"plot_ndvi() - Saving plot as PNG in {save_png[1]}.")
        plt.savefig(save_png[1], dpi=output_dpi, transparent=output_transparency)
    if save_pdf[0]:
        log(f"plot_ndvi() - Saving plot as PDF in {save_pdf[1]}.")
        pdf_name = save_pdf[1]
        if pdf_name != ".pdf":
            pdf_name = pdf_name + ".pdf"
        plt.savefig(
            pdf_name, format="pdf", dpi=output_dpi, transparent=output_transparency
        )


def plot_temperature(
    data_gdf: gpd.GeoDataFrame,
    ax: Axes,
    column="",
    location_name="",
    plot_osmnx_edges=(False, gpd.GeoDataFrame),
    plot_boundary=(False, gpd.GeoDataFrame),
    adjust_to=("", [0.05, 0.05]),
    save_png=(False, "../output/figures/plot.png"),
    output_transparency=False,
    output_dpi=300,
    save_pdf=(False, "../output/figures/plot.pdf"),
) -> None:
    """
    Creates a plot showing temperature analysis data.

    This function can be used to plot temperature analysis results (data showing hotter or colder areas relative to the overall mean, divided in 7 categories)
    or temperature tendency given the available years. The function defaults to temperature analysis results (using the 'temperature_mean' column).

    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    ax: matplotlib.axes
        ax to use in the plot.
    column: str, default ''.
        Name of the column with the data to plot from the data_gdf GeoDataFrame.
        This name defines the analysis to be performed.
        Passing 'temperature_tend' returns a tendency plot using 'temperature_tend' column
        NOTE: Passing ANY other value assumes that there is a column named 'temperature_mean' from which the difference relative to the overall mean is calculated.
    location_name: str, default ''.
        Text containing the location (e.g. city name), used to be added in the main plot title.
        If not provided, the title will not contain a location name.
    plot_osmnx_edges: tupple, default (False, gpd.GeoDataFrame).
        Tuple containing boolean on position [0]. If true, a gdf can be specified on position [1].
        The gdf must contain edges (line geometry) from Open Street Map with a column named 'highway' since the edges are shown according 'highway' values.
    plot_boundary: tupple, default (False, gpd.GeoDataFrame).
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
    output_transparency: bool, default False.
        If True, saves the output with transparency.
    output_dpi: int, default 300.
        Sets the resolution to be used to save the png.
    save_pdf: tupple, default (False, '../output/figures/plot.pdf').
        Tupple containing boolean on position [0]. If true, a string can be specified on position [1]
        indicating the directory and file name where the pdf is saved.

    Returns
    -------
    None, plots temperature analysis.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.

    """
    # Input validation
    log("plot_temperature() - Validating inputs.")
    if not isinstance(data_gdf, gpd.GeoDataFrame):
        raise TypeError("data_gdf must be a geopandas.GeoDataFrame instance.")
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(column, str):
        raise TypeError("column must be a string.")
    if not isinstance(location_name, str):
        raise TypeError("location_name must be a string.")
    # Input validation for plot_osmnx_edges tupple
    if not isinstance(plot_osmnx_edges, tuple) or len(plot_osmnx_edges) != 2:
        raise TypeError(
            "plot_osmnx_edges must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_osmnx_edges[0]:
        if not isinstance(plot_osmnx_edges[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_osmnx_edges[0] is True, plot_osmnx_edges[1] must be a GeoDataFrame."
            )
        if not all(
            plot_osmnx_edges[1].geometry.type.isin(["LineString", "MultiLineString"])
        ):
            raise TypeError(
                "The plot_osmnx_edges[1] GeoDataFrame must contain only LineString or MultiLineString geometries."
            )
    # Input validation for plot_boundary tupple
    if not isinstance(plot_boundary, tuple) or len(plot_boundary) != 2:
        raise TypeError(
            "plot_boundary must be a tuple with two elements: a boolean and a GeoDataFrame."
        )
    if plot_boundary[0]:
        if not isinstance(plot_boundary[1], gpd.GeoDataFrame):
            raise TypeError(
                "If plot_boundary[0] is True, plot_boundary[1] must be a GeoDataFrame."
            )
        if not all(plot_boundary[1].geometry.type.isin(["Polygon", "MultiPolygon"])):
            raise TypeError(
                "The plot_boundary[1] GeoDataFrame must contain only Polygon or MultiPolygon geometries."
            )
    if (
        not isinstance(adjust_to, tuple)
        or len(adjust_to) != 2
        or (not all(isinstance(i, (int, float)) for i in adjust_to[1]))
    ):
        raise TypeError(
            "adjust_to must be a tuple with two elements: a string (either 'boundary', 'edges' or '') and a list with two numeric values."
        )
    if (
        not isinstance(save_png, tuple)
        or len(save_png) != 2
        or not isinstance(save_png[1], str)
    ):
        raise TypeError(
            "save_png must be a tuple with two elements: a boolean and a string (file path)."
        )
    if not isinstance(output_transparency, bool):
        raise TypeError("output_transparency must be a boolean value (True or False).")
    if not isinstance(output_dpi, int):
        raise TypeError("output_dpi must be an integer value.")
    if (
        not isinstance(save_pdf, tuple)
        or len(save_pdf) != 2
        or not isinstance(save_pdf[1], str)
    ):
        raise TypeError(
            "save_pdf must be a tuple with two elements: a boolean and a string (file path)."
        )

    # --------------- DATA PLOT STYLE
    data_linestyle = "-"
    data_linewidth = 0.35
    data_edgecolor = "white"

    # --------------- FOR TENDENCY (temperature_tend)
    # (temperature_tend indicates the data's tendency analysed in all available years)
    if column == "temperature_tend":
        log(
            "plot_temperature() - Identified 'temperature_tend' in column. Creating plot for temperature tendency data."
        )
        # --- TITLE
        # Create plot title
        plot_title = f"Tendency of temperature data"  # Whithout period at the end to add location name if provided
        # --- LEGEND - COLOR BAR
        legend_type = "colorbar"
        # Define cmap and normalization
        vmin = data_gdf[column].min()
        vmax = data_gdf[column].max()
        # Define norm case
        if (
            vmin < 0 < vmax
        ):  # Regular case, there are negative and positive values in tendency
            cmap = plt.get_cmap("RdBu_r")
            # Create symmetrical norm scale (Prevents presenting small vmins or vmaxs with dark colors)
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif (
            vmin < vmax < 0
        ):  # Non-regular case, there are only negative values in tendency
            cmap = plt.get_cmap("Blues_r")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:  # Non-regular case, there are only positive values in tendency
            cmap = plt.get_cmap("Reds")
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # --- PLOT
        # Plot proximity data USING LEGEND=FALSE (SETTING IT  MANUALLY)
        data_gdf.plot(
            ax=ax,
            column=column,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            legend=False,
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )

    # --------------- FOR ANOMALY (temperature_anomaly)
    # (temperature_anomaly indicates the difference between the temperature_mean of each polygon and the overall (city) mean.
    else:
        log("plot_temperature() - Creating plot for temperature anomaly data.")
        # Rewrite column, no other values are permitted.
        column = "temperature_anomaly"
        # --- TITLE
        # Set plot title according to column used
        plot_title = f"Temperature difference between each polygon's mean and overall mean"  # Whithout period at the end to add location name if provided
        # --- LEGEND - CATEGORIZED
        legend_type = "categorized"
        # Calculate anomaly (difference between mean in each polygon and city's mean)
        mean_city_temperature = data_gdf.temperature_mean.mean()
        data_gdf["temperature_anomaly"] = (
            data_gdf["temperature_mean"] - mean_city_temperature
        )
        # Categorize anomaly
        classif_bins = [-100, -3.5, -1.5, -0.5, 0.5, 1.5, 3.5, 100]
        data_gdf["anomaly_class"] = pd.cut(
            data_gdf["temperature_anomaly"],
            bins=classif_bins,
            labels=[-3, -2, -1, 0, 1, 2, 3],
            include_lowest=True,
        ).astype(int)
        classes_dict = {
            3: "1. More temperature",
            2: "2.",
            1: "3.",
            0: "4. Near overall mean",
            -1: "5.",
            -2: "6.",
            -3: f"7. Less temperature",
        }
        data_gdf["anomaly_bins"] = data_gdf["anomaly_class"].map(classes_dict)
        # Define order and convert col into ordered category
        categories = list(classes_dict.values())
        data_gdf["anomaly_bins"] = pd.Categorical(
            data_gdf["anomaly_bins"], categories=categories, ordered=True
        )
        # Force categorical order
        data_gdf.sort_values(by="anomaly_bins", inplace=True)
        # --- PLOT
        # Plot temperature anomaly data using the viridis color palette reversed
        data_gdf.plot(
            ax=ax,
            column="anomaly_bins",
            cmap="RdBu",
            legend=True,
            legend_kwds={"loc": "lower left"},
            linestyle=data_linestyle,
            linewidth=data_linewidth,
            edgecolor=data_edgecolor,
            zorder=1,
        )

    # --------------- OPTIONAL COMPLEMENTS (Area of interest boundary and streets)
    # Plot boundary if available
    if plot_boundary[0]:
        log("plot_temperature() - Plotting area of interest's boundary provided.")
        boundary_gdf = plot_boundary[1].copy()
        boundary_gdf.boundary.plot(
            ax=ax, color="#bebebe", linestyle="--", linewidth=0.75, zorder=2
        )
    # Plot edges if available
    if plot_osmnx_edges[0]:
        log("plot_temperature() - Plotting Open Street Map edges provided.")
        edges_gdf = plot_osmnx_edges[1].copy()
        # Plot edges (Main, relevant on a regional scale)
        edges_shown_a = ["trunk", "trunk_link", "motorway", "motorway_link"]
        edges_gdf_main = edges_gdf[edges_gdf["highway"].isin(edges_shown_a)].copy()
        if len(edges_gdf_main) > 0:
            edges_main_available = True
            edges_gdf_main.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=1.0, zorder=3
            )
        # Plot edges (Other relevant on a city scale)
        edges_shown_b = ["primary", "primary_link"]  # ,'secondary','secondary_link']
        edges_gdf_primary = edges_gdf[edges_gdf["highway"].isin(edges_shown_b)].copy()
        if len(edges_gdf_primary) > 0:
            edges_primary_available = True
            edges_gdf_primary.plot(
                ax=ax, color="#000000", alpha=0.5, linewidth=0.50, zorder=3
            )

    # --------------- FINAL PLOT FORMAT
    # FORMAT - AX SIZE - Edit ax size to fit specified gdf exclusively using square_bounds() function
    # If specified 'boundary' and boundary_gdf available
    if (adjust_to[0] == "boundary") and (plot_boundary[0]):
        log("plot_temperature() - Adjusting ax size to boundary provided.")
        square_bounds(ax, boundary_gdf, adjust_to[1])
    # Elif specified 'edges' and edges_gdf available
    elif (adjust_to[0] == "edges") and (plot_osmnx_edges[0]):
        log("plot_temperature() - Adjusting ax size to edges provided.")
        if edges_main_available and edges_primary_available:
            square_bounds(
                ax, pd.concat([edges_gdf_main, edges_gdf_primary]), adjust_to[1]
            )
        elif edges_main_available:
            square_bounds(ax, edges_gdf_main, adjust_to[1])
        elif edges_primary_available:
            square_bounds(ax, edges_gdf_primary, adjust_to[1])
        else:
            log(
                "plot_temperature() - No edges available to adjust ax size to. Using data_gdf instead."
            )
            square_bounds(ax, data_gdf, adjust_to[1])
    # Else, if specified something else, use data_gdf
    else:
        log("plot_temperature() - Adjusting ax size to data_gdf.")
        square_bounds(ax, data_gdf, adjust_to[1])

    # FORMAT - OBSERVATORY'S PLOT FORMAT - Controls positioning, style and sizing for main title, legend, grid and logo.
    # Add location name to title if provided
    if location_name == "":
        plot_title = plot_title + "."
    else:
        plot_title = plot_title + f" in {location_name.capitalize()}."
    # Call observatory_plot_format() function
    log("plot_temperature() - Formatting plot to observatory's style.")
    if legend_type == "colorbar":
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[cmap, norm],
            grid=False,
        )
    else:
        observatory_plot_format(
            ax=ax,
            plot_title=plot_title,
            legend_title=f"Column: {column}.",
            legend_type=legend_type,
            cmap_args=[],
            grid=False,
        )

    # --------------- SAVING CONFIGURATIONS
    # Save or show plot
    if save_png[0]:
        log(f"plot_temperature() - Saving plot as PNG in {save_png[1]}.")
        plt.savefig(save_png[1], dpi=output_dpi, transparent=output_transparency)
    if save_pdf[0]:
        log(f"plot_temperature() - Saving plot as PDF in {save_pdf[1]}.")
        pdf_name = save_pdf[1]
        if pdf_name != ".pdf":
            pdf_name = pdf_name + ".pdf"
        plt.savefig(
            pdf_name, format="pdf", dpi=output_dpi, transparent=output_transparency
        )


def plot_temperature_anomaly(data_gdf: gpd.GeoDataFrame, ax: Axes, kwargs={}) -> None:
    """
    Creates a plot showing temperature analysis results (data showing hotter or colder areas relative to the overall mean, divided in 7 categories).

    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    ax: matplotlib.axes
        ax to use in the plot.
    kwargs: dict, default {}.
        Dictionary with additional parameters to be passed to the plot_temperature() function.

    Returns
    -------
    None, plots temperature analysis.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.

    """
    # Input validation
    log("plot_temperature_anomaly() - Validating inputs.")
    if not isinstance(data_gdf, gpd.GeoDataFrame):
        raise TypeError("data_gdf must be a geopandas.GeoDataFrame instance.")
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary.")
    if "data_gdf" in kwargs or "ax" in kwargs:
        raise TypeError(
            "kwargs should not contain 'data_gdf' or 'ax' keys, they are already provided as parameters."
        )
    # Set column in order to make sure that a temperature anomaly plot is generated.
    kwargs["column"] = "temperature_mean"
    # Call plot_temperature() function
    log("plot_temperature_anomaly() - Calling plot_temperature() function.")
    plot_temperature(data_gdf=data_gdf, ax=ax, **kwargs)


def plot_temperature_tendency(data_gdf: gpd.GeoDataFrame, ax: Axes, kwargs={}) -> None:
    """
    Creates a plot showing temperature tendency given the available years.

    Parameters
    ----------
    data_gdf: geopandas.GeoDataFrame
        GeoDataFrame with the ndvi analysis data from OdC.
    ax: matplotlib.axes
        ax to use in the plot.
    kwargs: dict, default {}.
        Dictionary with additional parameters to be passed to the plot_temperature() function.

    Returns
    -------
    None, plots temperature analysis.

    Raises
    ------
    TypeError
        If required inputs are missing or if inputs are invalid.

    """
    # Input validation
    log("plot_temperature_tendency() - Validating inputs.")
    if not isinstance(data_gdf, gpd.GeoDataFrame):
        raise TypeError("data_gdf must be a geopandas.GeoDataFrame instance.")
    if not isinstance(ax, (Axes)):
        raise TypeError("ax must be a matplotlib.axes.Axes instance.")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary.")
    if "data_gdf" in kwargs or "ax" in kwargs:
        raise TypeError(
            "kwargs should not contain 'data_gdf' or 'ax' keys, they are already provided as parameters."
        )

    # Set column in order to make sure that a temperature tendency plot is generated.
    kwargs["column"] = "temperature_tend"
    # Call plot_temperature() function
    log("plot_temperature_tendency() - Calling plot_temperature() function.")
    plot_temperature(data_gdf=data_gdf, ax=ax, **kwargs)
