import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, Polygon, MultiPolygon
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    from odc import data
else:
    from odc import data


class TestData:
    """Essential tests for data module functions."""

    # Test fixtures - basic data setup
    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({
            'int_col': ['1', '2', '3', '4', '5'],
            'float_col': ['1.5', '2.5', '3.5', '4.5', '5.5'],
            'string_col': [1, 2, 3, 4, 5],
            'bool_col': ['True', 'False', 'True', 'False', 'True'],
            'date_col': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        })

    @pytest.fixture
    def sample_geodataframe(self):
        """Sample GeoDataFrame for testing."""
        return gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'geometry': [
                Point(-99.170, 19.410),
                Point(-99.165, 19.415),
                Point(-99.160, 19.408)
            ]
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_polygon_gdf(self):
        """Sample polygon GeoDataFrame for testing."""
        polygon = Polygon([
            (-99.17759367, 19.42042723),
            (-99.16243782, 19.42370587),
            (-99.15521145, 19.41136457),
            (-99.15634423, 19.40414356),
            (-99.17110947, 19.40027503),
            (-99.18149982, 19.41243296),
            (-99.17759367, 19.42042723)
        ])
        return gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')

    @pytest.fixture
    def temp_directory(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

    # Test convert_column_types function
    def test_convert_column_types_basic(self, sample_dataframe):
        """Test basic column type conversion."""
        type_mapping = {
            'integer': ['int_col'],
            'float': ['float_col'],
            'string': ['string_col']
        }
        result = data.convert_column_types(sample_dataframe, type_mapping)
        
        assert result['int_col'].dtype in [np.int8,np.int32, np.int64]
        assert result['float_col'].dtype in [np.float32, np.float64]
        assert result['string_col'].dtype == object

    def test_convert_column_types_boolean(self, sample_dataframe):
        """Test boolean conversion."""
        type_mapping = {'boolean': ['bool_col']}
        result = data.convert_column_types(sample_dataframe, type_mapping)
        
        assert result['bool_col'].dtype == bool

    def test_convert_column_types_datetime(self, sample_dataframe):
        """Test datetime conversion."""
        type_mapping = {'datetime': ['date_col']}
        result = data.convert_column_types(sample_dataframe, type_mapping)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date_col'])

    def test_convert_column_types_multiple_types(self, sample_dataframe):
        """Test converting multiple column types at once."""
        type_mapping = {
            'integer': ['int_col'],
            'float': ['float_col'],
            'string': ['string_col'],
            'boolean': ['bool_col']
        }
        result = data.convert_column_types(sample_dataframe, type_mapping)
        
        assert result['int_col'].dtype in [np.int8,np.int32, np.int64]
        assert result['float_col'].dtype in [np.float32, np.float64]
        assert result['string_col'].dtype == object
        assert result['bool_col'].dtype == bool

    def test_convert_column_types_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        type_mapping = {'string': ['col1']}
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            data.convert_column_types(empty_df, type_mapping)

    def test_convert_column_types_invalid_type(self, sample_dataframe):
        """Test handling of invalid data type."""
        type_mapping = {'invalid_type': ['int_col']}
        
        with pytest.raises(ValueError, match="Unsupported data type"):
            data.convert_column_types(sample_dataframe, type_mapping)

    def test_convert_column_types_missing_column(self, sample_dataframe):
        """Test handling of missing column."""
        type_mapping = {'string': ['nonexistent_column']}
        
        with pytest.raises(KeyError, match="not found in DataFrame"):
            data.convert_column_types(sample_dataframe, type_mapping)

    def test_convert_column_types_not_dict(self, sample_dataframe):
        """Test handling of non-dictionary type_mapping."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            data.convert_column_types(sample_dataframe, ['not', 'a', 'dict'])

    def test_convert_column_types_non_list_columns(self, sample_dataframe):
        """Test handling of non-list column specification."""
        type_mapping = {'string': 'int_col'}  # Should be a list
        
        with pytest.raises(ValueError, match="must be a list"):
            data.convert_column_types(sample_dataframe, type_mapping)

    # Test clear_directory function
    def test_clear_directory_basic(self, temp_directory):
        """Test basic directory clearing."""
        # Create some test files
        test_file1 = Path(temp_directory) / "test1.txt"
        test_file2 = Path(temp_directory) / "test2.txt"
        test_file1.write_text("content")
        test_file2.write_text("content")
        
        result = data.clear_directory(temp_directory)
        
        assert result['files'] == 2
        assert result['directories'] == 0
        assert len(list(Path(temp_directory).iterdir())) == 0

    def test_clear_directory_with_subdirectories(self, temp_directory):
        """Test clearing directory with subdirectories."""
        # Create subdirectory with files
        subdir = Path(temp_directory) / "subdir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")
        
        result = data.clear_directory(temp_directory)
        
        assert result['directories'] == 1
        assert len(list(Path(temp_directory).iterdir())) == 0

    def test_clear_directory_nonexistent(self):
        """Test handling of nonexistent directory."""
        with pytest.raises(ValueError, match="does not exist"):
            data.clear_directory("/nonexistent/path")

    def test_clear_directory_not_directory(self, temp_directory):
        """Test handling of non-directory path."""
        file_path = Path(temp_directory) / "test.txt"
        file_path.write_text("content")
        
        with pytest.raises(ValueError, match="not a directory"):
            data.clear_directory(file_path)

    def test_clear_directory_empty(self, temp_directory):
        """Test clearing already empty directory."""
        result = data.clear_directory(temp_directory)
        
        assert result['files'] == 0
        assert result['directories'] == 0

    # Test download_osm_network function
    def test_download_osm_network_from_polygon(self, sample_polygon_gdf):
        """Test OSM network download from polygon."""
        graph, nodes, edges = data.download_osm_network(
            sample_polygon_gdf,
            method='from_polygon',
            network_type='walk'
        )
        
        assert isinstance(graph, nx.MultiDiGraph)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert len(nodes) > 0
        assert len(edges) > 0
        assert nodes.crs == 'EPSG:4326'
        assert edges.crs == 'EPSG:4326'

    def test_download_osm_network_from_bbox(self, sample_polygon_gdf):
        """Test OSM network download from bounding box."""
        graph, nodes, edges = data.download_osm_network(
            sample_polygon_gdf,
            method='from_bbox',
            network_type='walk'
        )
        
        assert isinstance(graph, nx.MultiDiGraph)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert len(nodes) > 0
        assert len(edges) > 0
                           
    def test_download_osm_network_nodes_structure(self, sample_polygon_gdf):
        """Test that nodes have required columns."""
        graph, nodes, edges = data.download_osm_network(
            sample_polygon_gdf,
            method='from_polygon',
            network_type='walk'
        )
        
        required_columns = ["x", "y", "street_count", "geometry"]
        for col in required_columns:
            assert col in nodes.columns

    def test_download_osm_network_edges_structure(self, sample_polygon_gdf):
        """Test that edges have required columns."""
        graph, nodes, edges = data.download_osm_network(
            sample_polygon_gdf,
            method='from_polygon',
            network_type='walk'
        )
        
        required_columns = ["osmid", "oneway", "length", "geometry"]
        for col in required_columns:
            assert col in edges.columns

    def test_download_osm_network_empty_geodataframe(self):
        """Test handling of empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            data.download_osm_network(empty_gdf)

    def test_download_osm_network_invalid_method(self, sample_polygon_gdf):
        """Test handling of invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            data.download_osm_network(
                sample_polygon_gdf,
                method='invalid_method'
            )

    def test_download_osm_network_invalid_network_type(self, sample_polygon_gdf):
        """Test handling of invalid network type."""
        with pytest.raises(ValueError, match="Invalid network_type"):
            data.download_osm_network(
                sample_polygon_gdf,
                network_type='invalid_type'
            )

    def test_download_osm_network_different_network_types(self, sample_polygon_gdf):
        """Test downloading different network types."""
        for net_type in ['drive', 'walk', 'bike']:
            graph, nodes, edges = data.download_osm_network(
                sample_polygon_gdf,
                method='from_polygon',
                network_type=net_type
            )
            assert len(nodes) > 0
            assert len(edges) > 0

    # Test create_hexagonal_grid function
    def test_create_hexagonal_grid_basic(self, sample_polygon_gdf):
        """Test basic hexagonal grid creation."""
        result = data.create_hexagonal_grid(sample_polygon_gdf, resolution=8)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert 'hex_id_8' in result.columns
        assert result.crs == 'EPSG:4326'

    def test_create_hexagonal_grid_different_resolutions(self, sample_polygon_gdf):
        """Test hexagonal grid with different resolutions."""
        result_low = data.create_hexagonal_grid(sample_polygon_gdf, resolution=9)
        result_high = data.create_hexagonal_grid(sample_polygon_gdf, resolution=10)
        
        # Higher resolution should produce more hexagons
        assert len(result_high) > len(result_low)
        assert 'hex_id_9' in result_low.columns
        assert 'hex_id_10' in result_high.columns

    def test_create_hexagonal_grid_geometry_column(self, sample_polygon_gdf):
        """Test that result has geometry column."""
        result = data.create_hexagonal_grid(sample_polygon_gdf, resolution=8)
        
        assert 'geometry' in result.columns
        assert result.geometry.geom_type.iloc[0] == 'Polygon'

    def test_create_hexagonal_grid_no_duplicates(self, sample_polygon_gdf):
        """Test that result has no duplicate hexagons."""
        result = data.create_hexagonal_grid(sample_polygon_gdf, resolution=8)
        
        initial_count = len(result)
        dedup_count = len(result.drop_duplicates(subset=['hex_id_8']))
        assert initial_count == dedup_count

    def test_create_hexagonal_grid_empty_geodataframe(self):
        """Test handling of empty GeoDataFrame."""
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        
        with pytest.raises(ValueError, match="cannot be None or empty"):
            data.create_hexagonal_grid(empty_gdf, resolution=10)

    def test_create_hexagonal_grid_multipolygon(self):
        """Test hexagonal grid with MultiPolygon."""
        poly1 = Polygon([
		(-99.17709913, 19.41851742), 
		(-99.1682861, 19.41873907), 
		(-99.16826651, 19.41292077), 
		(-99.17823503, 19.41168321), 
		(-99.17709913, 19.41851742)
        ])
        poly2 = Polygon([
		(-99.17040191, 19.40765678), 
		(-99.17068946, 19.40244942), 
		(-99.1586123, 19.40434795), 
		(-99.15740458, 19.41036888), 
		(-99.17040191, 19.40765678)
        ])
        multipoly = MultiPolygon([poly1, poly2])
        
        gdf = gpd.GeoDataFrame({'geometry': [multipoly]}, crs='EPSG:4326')
        result = data.create_hexagonal_grid(gdf, resolution=8)
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_create_hexagonal_grid_crs_conversion(self):
        """Test that function handles CRS conversion."""
        # Create polygon in different CRS
        polygon = Polygon([
            (-99.17759367, 19.42042723),
            (-99.16243782, 19.42370587),
            (-99.15521145, 19.41136457),
            (-99.15634423, 19.40414356),
            (-99.17110947, 19.40027503),
            (-99.18149982, 19.41243296),
            (-99.17759367, 19.42042723)
        ])
        gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs='EPSG:4326')
        gdf = gdf.to_crs('EPSG:3857')  # Convert to different CRS
        
        result = data.create_hexagonal_grid(gdf, resolution=8)
        
        assert result.crs == 'EPSG:4326'

    def test_create_hexagonal_grid_point_geometry(self, sample_geodataframe):
        """Test hexagonal grid with point geometries (should fail)."""
        # H3 requires polygon geometries
        with pytest.raises(RuntimeError):
            data.create_hexagonal_grid(sample_geodataframe, resolution=8)


# Simple test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
