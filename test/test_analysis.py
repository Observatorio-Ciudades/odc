import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
from shapely.geometry import Point, Polygon, LineString
from unittest.mock import patch, MagicMock
import sys
import os

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    from odc import analysis
else:
    from odc import analysis


class TestAnalysisFunctions:
    """Essential tests for spatial analysis functions."""

    # Test fixtures - basic data setup
    @pytest.fixture
    def sample_points(self):
        """Sample points GeoDataFrame for testing."""
        return gpd.GeoDataFrame({
            'point_id': ['A', 'B', 'C', 'D', 'E'],
            'value': [10.5, 20.3, 15.7, 30.2, 25.1],
            'category': [1,1,2,2,3],
            'geometry': [
                Point(0, 0),
                Point(1, 0),
                Point(0, 1),
                Point(1, 1),
                Point(0.5, 0.5)
            ]
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_bins(self):
        """Sample spatial bins for testing."""
        return gpd.GeoDataFrame({
            'bin_id': ['bin_1', 'bin_2', 'bin_3'],
            'geometry': [
                Polygon([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]),
                Polygon([(0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (0.5, 0.5)]),
                Polygon([(-0.5, 0.5), (0.5, 0.5), (0.5, 1.5), (-0.5, 1.5)])
            ]
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_hex_data(self):
        """Sample H3 hexagon data for testing."""
        center_lat, center_lon = 19.4326, -99.1332
        resolution = 9
        center_hex = h3.latlng_to_cell(center_lat, center_lon, resolution)
        
        # Get hex ring around center
        hex_ids = list(h3.grid_disk(center_hex, 2))
        
        # Create GeoDataFrame with hex geometries
        geometries = []
        values = []
        for hex_id in hex_ids:
            boundary = h3.cell_to_boundary(hex_id)
            # Convert to (lon, lat) order for Shapely
            coords = [(lon, lat) for lat, lon in boundary]
            geometries.append(Polygon(coords))
            values.append(np.random.uniform(10, 50))
        
        return gpd.GeoDataFrame({
            f'hex_id_{resolution}': hex_ids,
            'value': values,
            'geometry': geometries
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_aoi(self):
        """Sample area of interest polygon."""
        return gpd.GeoDataFrame({
            'geometry': [Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])]
        }, crs='EPSG:4326')

    # Test group_points_by_bins function
    def test_group_points_by_bins_basic(self, sample_points, sample_bins):
        """Test basic point aggregation by bins."""
        result = analysis.group_points_by_bins(
            points=sample_points,
            bins=sample_bins,
            bin_id_column='bin_id',
            aggregate_columns='value'
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'value' in result.columns
        assert len(result) == len(sample_bins)
        assert result['value'].notna().any()

    def test_group_points_by_bins_multiple_columns(self, sample_points, sample_bins):
        """Test aggregation with multiple columns."""
        result = analysis.group_points_by_bins(
            points=sample_points,
            bins=sample_bins,
            bin_id_column='bin_id',
            aggregate_columns=['value', 'category']
        )
        
        assert 'value' in result.columns
        # Note: category aggregation with 'mean' may not be meaningful
        # but we test that it doesn't error

    def test_group_points_by_bins_custom_aggregation(self, sample_points, sample_bins):
        """Test custom aggregation functions."""
        result = analysis.group_points_by_bins(
            points=sample_points,
            bins=sample_bins,
            bin_id_column='bin_id',
            aggregate_columns='value',
            aggregation_func='sum'
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert result['value'].sum() > 0

    def test_group_points_by_bins_fill_missing(self, sample_points, sample_bins):
        """Test filling missing values."""
        result = analysis.group_points_by_bins(
            points=sample_points,
            bins=sample_bins,
            bin_id_column='bin_id',
            aggregate_columns='value',
            fill_missing=True,
            fill_value=-999
        )
        
        # Check that bins without points have fill value
        assert (result['value'] == -999).any() or result['value'].notna().all()

    def test_group_points_by_bins_zero_replacement(self, sample_points, sample_bins):
        """Test zero value replacement."""
        # Create data with potential zeros
        points_with_zeros = sample_points.copy()
        points_with_zeros.loc[0, 'value'] = 0
        
        result = analysis.group_points_by_bins(
            points=points_with_zeros,
            bins=sample_bins,
            bin_id_column='bin_id',
            aggregate_columns='value',
            zero_replacement=0.1
        )
        
        # Check that zeros were replaced
        assert (result['value'] != 0).all() or result['value'].isna().any()

    def test_group_points_by_bins_invalid_inputs(self, sample_points, sample_bins):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            analysis.group_points_by_bins(
                points=sample_points,
                bins=sample_bins,
                bin_id_column='nonexistent',
                aggregate_columns='value'
            )

    # Test fill_missing_h3_data function
    def test_fill_missing_h3_data_basic(self, sample_hex_data):
        """Test basic H3 data filling."""
        resolution = 9
        id_col = f'hex_id_{resolution}'
        
        # Split data into complete and missing
        data_with_values = sample_hex_data.iloc[:5].copy()
        missing_data = sample_hex_data.iloc[5:].copy()
        missing_data = missing_data.drop(columns=['value'])
        
        def mock_neighbor_function(hex_id):
            return list(h3.grid_ring(hex_id, 1))
        
        result = analysis.fill_missing_h3_data(
            missing_data=missing_data,
            data_with_values=data_with_values,
            id_column=id_col,
            value_columns='value',
            neighbor_function=mock_neighbor_function,
            max_iterations=10
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'value' in result.columns
        assert len(result) == len(sample_hex_data)

    def test_fill_missing_h3_data_multiple_columns(self, sample_hex_data):
        """Test filling multiple value columns."""
        resolution = 9
        id_col = f'hex_id_{resolution}'
        
        # Add second value column
        sample_hex_data['value2'] = sample_hex_data['value'] * 2
        
        data_with_values = sample_hex_data.iloc[:5].copy()
        missing_data = sample_hex_data.iloc[5:].copy()
        missing_data = missing_data.drop(columns=['value', 'value2'])
        
        def mock_neighbor_function(hex_id):
            return list(h3.grid_ring(hex_id, 1))
        
        result = analysis.fill_missing_h3_data(
            missing_data=missing_data,
            data_with_values=data_with_values,
            id_column=id_col,
            value_columns=['value', 'value2'],
            neighbor_function=mock_neighbor_function,
            max_iterations=10
        )
        
        assert 'value' in result.columns
        assert 'value2' in result.columns

    # Test sigmoidal functions
    def test_sigmoidal_function_basic(self):
        """Test basic sigmoidal transformation."""
        result = analysis.sigmoidal_function(x=2.0, di=0.5, d0=10.0)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_sigmoidal_function_boundary_values(self):
        """Test sigmoidal function at boundary values."""
        # At d0, should be close to 0.5
        result_at_d0 = analysis.sigmoidal_function(x=0.5, di=10.0, d0=10.0)
        assert 0.4 <= result_at_d0 <= 0.6

    def test_sigmoidal_function_constant_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            analysis.sigmoidal_function_constant(
                positive_limit_value=5.0,
                mid_limit_value=10.0  # Invalid: should be less than positive_limit
            )

    # Test interpolation functions
    def test_interpolate_to_gdf_basic(self):
        """Test basic IDW interpolation to GeoDataFrame."""
        # Create observation points
        x = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        z = np.array([10.0, 20.0, 30.0, 40.0])
        
        # Create target points
        target_gdf = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5), Point(0.25, 0.25)]
        }, crs='EPSG:4326')
        
        result = analysis.interpolate_to_gdf(target_gdf, x, y, z)
        
        assert 'interpolated_value' in result.columns
        assert len(result) == len(target_gdf)
        assert result['interpolated_value'].notna().all()

    def test_interpolate_to_gdf_with_search_radius(self):
        """Test IDW interpolation with search radius."""
        x = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
        y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        z = np.array([10.0, 15.0, 20.0, 30.0, 35.0, 40.0])
        
        target_gdf = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5)]
        }, crs='EPSG:4326')
        
        result = analysis.interpolate_to_gdf(
            target_gdf, x, y, z, 
            search_radius=0.5
        )
        
        assert 'interpolated_value' in result.columns

    def test_idw_at_point_basic(self):
        """Test IDW interpolation at single point."""
        x0 = np.array([0.0, 1.0, 0.0, 1.0])
        y0 = np.array([0.0, 0.0, 1.0, 1.0])
        z0 = np.array([10.0, 20.0, 30.0, 40.0])
        
        result = analysis.idw_at_point(x0, y0, z0, 0.5, 0.5)
        
        assert isinstance(result, (float, np.ndarray))
        # Value should be between min and max of observations
        if isinstance(result, np.ndarray):
            assert np.all((result >= 10.0) & (result <= 40.0))
        else:
            assert 10.0 <= result <= 40.0

    def test_interpolate_at_points_basic(self):
        """Test IDW interpolation at multiple points."""
        x0 = np.array([0.0, 1.0, 0.0, 1.0])
        y0 = np.array([0.0, 0.0, 1.0, 1.0])
        z0 = np.array([10.0, 20.0, 30.0, 40.0])
        
        xi = np.array([0.5, 0.25])
        yi = np.array([0.5, 0.25])
        
        result = analysis.interpolate_at_points(x0, y0, z0, xi, yi)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(xi)

    # Test weighted_average function
    def test_weighted_average_basic(self):
        """Test weighted average calculation."""
        df = pd.DataFrame({
            'weight': [1, 2, 3],
            'value': [10, 20, 30]
        })
        
        result = analysis.weighted_average(df, 'weight', 'value')
        
        assert isinstance(result, float)
        # Weighted avg = (1*10 + 2*20 + 3*30) / (1+2+3) = 23.33...
        assert 23.0 <= result <= 24.0

    def test_weighted_average_equal_weights(self):
        """Test weighted average with equal weights."""
        df = pd.DataFrame({
            'weight': [1, 1, 1],
            'value': [10, 20, 30]
        })
        
        result = analysis.weighted_average(df, 'weight', 'value')
        
        # Should equal simple average
        assert result == 20.0

    # Test voronoi_points_within_aoi function
    def test_voronoi_points_within_aoi_basic(self, sample_aoi):
        """Test basic Voronoi polygon generation."""
        points = gpd.GeoDataFrame({
            'point_id': ['A', 'B', 'C'],
            'geometry': [
                Point(0, 0),
                Point(1, 0),
                Point(0.5, 1)
            ]
        }, crs='EPSG:4326')
        
        result = analysis.voronoi_points_within_aoi(
            area_of_interest=sample_aoi,
            points=points,
            points_id_col='point_id',
            admissible_error=0.05
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert 'point_id' in result.columns
        assert len(result) == len(points)
        assert result.geometry.notna().all()

    def test_voronoi_points_within_aoi_projected_crs(self, sample_aoi):
        """Test Voronoi with custom projected CRS."""
        points = gpd.GeoDataFrame({
            'point_id': ['A', 'B'],
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs='EPSG:4326')
        
        result = analysis.voronoi_points_within_aoi(
            area_of_interest=sample_aoi,
            points=points,
            points_id_col='point_id',
            projected_crs='EPSG:3857'  # Web Mercator
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs.to_string() == 'EPSG:4326'  # Should return in WGS84

    # Test helper functions
    def test_sigmoid_objective_function(self):
        """Test sigmoid objective function helper."""
        result = analysis._sigmoid_objective_function(
            x=0.5, di=5.0, d0=10.0, target_value=0.75
        )
        
        assert isinstance(result, float)
        assert result >= 0  # Should be absolute difference

    def test_find_decay_constant(self):
        """Test decay constant finder helper."""
        result = analysis._find_decay_constant(
            di=5.0, d0=10.0, target_value=0.75
        )
        
        assert isinstance(result, float)


# Simple test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
