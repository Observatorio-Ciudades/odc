import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import tempfile
import os
import sys
from pathlib import Path

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    from odc import census_mx
else:
    from odc import census_mx


class TestCensusMX:
    """Essential tests for census_mx module functions."""

    # Test fixtures - census data setup
    @pytest.fixture
    def sample_ageb_gdf(self):
        """Sample AGEB GeoDataFrame with INEGI structure."""
        return gpd.read_file('data/ageb_test.gpkg') 
      
    @pytest.fixture
    def sample_block_gdf_with_nans(self):
        """Sample block GeoDataFrame with NaN values for testing imputation."""
       
        return gpd.read_file('data/blocks_test.gpkg')

    @pytest.fixture
    def sample_points_gdf(self):
        """Sample points GeoDataFrame for testing."""
        return gpd.read_file("data/nodes_test.gpkg")

    @pytest.fixture
    def sample_polygons_for_aggregation(self):
        """Sample polygons for point aggregation testing."""
        poly1 = Polygon([
            (-99.14213853, 19.3401555), 
            (-99.1410835, 19.33660525), 
            (-99.14031316, 19.33536601), 
            (-99.13808588, 19.33503108), 
            (-99.13731554, 19.33600237), 
            (-99.13771746, 19.3382464), 
            (-99.13811938, 19.33958612), 
            (-99.13865526, 19.33997129), 
            (-99.14213853, 19.3401555)
        ])
        poly2 = Polygon([
            (-99.13669593, 19.33700716), 
            (-99.13646147, 19.33479663), 
            (-99.13584186, 19.33359088), 
            (-99.13393276, 19.33285404), 
            (-99.13389927, 19.33506457), 
            (-99.13415046, 19.33596888), 
            (-99.13450214, 19.33677271), 
            (-99.13669593, 19.33700716)
        ])
        
        return gpd.GeoDataFrame({
            'polygon_id': ['poly1', 'poly2'],
            'geometry': [poly1, poly2]
        }, crs='EPSG:4326')

    # Test socio_polygon_to_points function
    def test_socio_polygon_to_points_basic(self, sample_points_gdf, sample_ageb_gdf):
        """Test basic polygon to points distribution."""
        result = census_mx.socio_polygon_to_points(
            points=sample_points_gdf,
            gdf_socio=sample_ageb_gdf,
            column_start=1,
            column_end=5,
            cve_column='CVE_AGEB'
        )
        result.columns = [x.upper() for x in result.columns]
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_points_gdf)
        assert 'POBTOT' in result.columns

    def test_socio_polygon_to_points_proportional_division(self, sample_points_gdf, sample_ageb_gdf):
        """Test that values are proportionally divided."""
        result = census_mx.socio_polygon_to_points(
            points=sample_points_gdf,
            gdf_socio=sample_ageb_gdf,
            cve_column='CVE_AGEB'
        )
        
        # Check that divided values are less than original totals
        ageb1_total_pop = sample_ageb_gdf[sample_ageb_gdf['CVE_AGEB'] == '0900300010323']['pobtot'].iloc[0]
        result_ageb1 = result[result['CVE_AGEB'] == '0900300010323']
        
        if len(result_ageb1) > 0:
            assert result_ageb1['pobtot'].iloc[0] < ageb1_total_pop

    def test_socio_polygon_to_points_avg_columns(self, sample_points_gdf, sample_ageb_gdf):
        """Test that avg_columns are not divided."""
        # REL_H_M should be averaged, not divided
        result = census_mx.socio_polygon_to_points(
            points=sample_points_gdf,
            gdf_socio=sample_ageb_gdf,
            cve_column='CVE_AGEB',
            avg_columns=['rel_h_m']
        )
        
        # Values in avg_columns should remain close to original
        original_rel = sample_ageb_gdf['rel_h_m'].iloc[0]
        result_with_ageb = result[result['CVE_AGEB'].notna()]
        selected_ageb = sample_ageb_gdf.iloc[0].CVE_AGEB
        result_with_ageb = result_with_ageb.loc[result_with_ageb.CVE_AGEB==selected_ageb]
        if len(result_with_ageb) > 0:
            assert abs(result_with_ageb['rel_h_m'].iloc[0] - original_rel) < 1

    def test_socio_polygon_to_points_missing_column(self, sample_points_gdf, sample_ageb_gdf):
        """Test handling of missing column."""
        with pytest.raises(KeyError):
            census_mx.socio_polygon_to_points(
                points=sample_points_gdf,
                gdf_socio=sample_ageb_gdf,
                cve_column='NONEXISTENT_COLUMN'
            )

    def test_socio_polygon_to_points_crs_handling(self, sample_points_gdf, sample_ageb_gdf):
        """Test CRS handling and conversion."""
        # Convert points to different CRS
        points_3857 = sample_points_gdf.to_crs('EPSG:3857')
        
        result = census_mx.socio_polygon_to_points(
            points=points_3857,
            gdf_socio=sample_ageb_gdf,
            cve_column='CVE_AGEB',
            target_crs='EPSG:4326'
        )
        
        assert result.crs == 'EPSG:4326'

    # Test socio_points_to_polygon function
    def test_socio_points_to_polygon_basic(self, sample_points_gdf, sample_polygons_for_aggregation):
        """Test basic point to polygon aggregation."""
        result = census_mx.socio_points_to_polygon(
            gdf_polygon=sample_polygons_for_aggregation,
            gdf_socio=sample_points_gdf,
            cve_column='polygon_id',
            string_columns=['city']
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'polygon_id' in result.columns
        assert 'value' in result.columns

    def test_socio_points_to_polygon_aggregation(self, sample_points_gdf, sample_polygons_for_aggregation):
        """Test that values are correctly aggregated."""
        result = census_mx.socio_points_to_polygon(
            gdf_polygon=sample_polygons_for_aggregation,
            gdf_socio=sample_points_gdf,
            cve_column='polygon_id',
            string_columns=['city']
        )
        
        # Values should be summed by default
        poly1_value = result[result['polygon_id'] == 'poly1']['value'].iloc[0]
        assert poly1_value > 0

    def test_socio_points_to_polygon_weighted_average(self):
        """Test weighted average calculation."""
        points = gpd.GeoDataFrame({
            'id': [1, 2, 3],
            'rate': [10.0, 20.0, 30.0],
            'population': [100, 200, 300],
            'geometry': [
                Point(-99.169, 19.409),
                Point(-99.167, 19.409),
                Point(-99.164, 19.409)
            ]
        }, crs='EPSG:4326')
        
        polygon = gpd.GeoDataFrame({
            'poly_id': ['p1'],
            'geometry': [Polygon([
                (-99.170, 19.410), (-99.163, 19.410),
                (-99.163, 19.408), (-99.170, 19.408), (-99.170, 19.410)
            ])]
        }, crs='EPSG:4326')
        
        result = census_mx.socio_points_to_polygon(
            gdf_polygon=polygon,
            gdf_socio=points,
            cve_column='poly_id',
            string_columns=['id'],
            wgt_dict={'rate': 'population'}
        )
        
        # Weighted average should be calculated
        expected_wavg = (10*100 + 20*200 + 30*300) / (100 + 200 + 300)
        assert abs(result['rate'].iloc[0] - expected_wavg) < 0.01

    def test_socio_points_to_polygon_empty_inputs(self, sample_points_gdf):
        """Test handling of empty inputs."""
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        
        with pytest.raises(ValueError):
            census_mx.socio_points_to_polygon(
                gdf_polygon=empty_gdf,
                gdf_socio=sample_points_gdf,
                cve_column='id',
                string_columns=[]
            )

    def test_socio_points_to_polygon_missing_columns(self, sample_points_gdf, sample_polygons_for_aggregation):
        """Test handling of missing columns."""
        with pytest.raises(KeyError):
            census_mx.socio_points_to_polygon(
                gdf_polygon=sample_polygons_for_aggregation,
                gdf_socio=sample_points_gdf,
                cve_column='NONEXISTENT',
                string_columns=[]
            )

    # Test group_sociodemographic_data function
    def test_group_sociodemographic_data_sum(self):
        """Test basic sum aggregation."""
        df = pd.DataFrame({
            'pop': [100, 200, 300],
            'households': [20, 30, 40]
        })
        
        result = census_mx.group_sociodemographic_data(
            df_socio=df,
            numeric_cols=['pop', 'households']
        )
        
        assert result['pop'] == 600
        assert result['households'] == 90

    def test_group_sociodemographic_data_average(self):
        """Test simple average calculation."""
        df = pd.DataFrame({
            'rate': [10.0, 20.0, 30.0],
            'pop': [100, 200, 300]
        })
        
        result = census_mx.group_sociodemographic_data(
            df_socio=df,
            numeric_cols=['rate', 'pop'],
            avg_column=['rate']
        )
        
        assert result['rate'] == 20.0  # Simple average
        assert result['pop'] == 600    # Sum

    def test_group_sociodemographic_data_weighted_average(self):
        """Test weighted average calculation."""
        df = pd.DataFrame({
            'rate': [10.0, 20.0, 30.0],
            'weight': [100, 200, 300]
        })
        
        result = census_mx.group_sociodemographic_data(
            df_socio=df,
            numeric_cols=['rate', 'weight'],
            avg_dict={'rate': 'weight'}
        )
        
        expected = (10*100 + 20*200 + 30*300) / 600
        assert abs(result['rate'] - expected) < 0.01

    def test_group_sociodemographic_data_empty_df(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        
        result = census_mx.group_sociodemographic_data(
            df_socio=df,
            numeric_cols=['col1']
        )
        
        assert result == {}

    def test_group_sociodemographic_data_nan_handling(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'value': [10, np.nan, 30]
        })
        
        result = census_mx.group_sociodemographic_data(
            df_socio=df,
            numeric_cols=['value']
        )
        
        assert result['value'] == 40  # NaN should be ignored

    # Test calculate_censo_nan_values function
    def test_calculate_censo_nan_values_basic(self, sample_ageb_gdf, sample_block_gdf_with_nans):
        """Test basic NaN value calculation."""
        result = census_mx.calculate_censo_nan_values(
            pop_ageb_gdf=sample_ageb_gdf,
            pop_mza_gdf=sample_block_gdf_with_nans,
            extended_logs=False
        )
        
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(sample_block_gdf_with_nans)

    def test_calculate_censo_nan_values_reduces_nans(self, sample_ageb_gdf, sample_block_gdf_with_nans):
        """Test that NaN values are reduced."""
        original_nan_count = sample_block_gdf_with_nans.isna().sum().sum()
        
        result = census_mx.calculate_censo_nan_values(
            pop_ageb_gdf=sample_ageb_gdf,
            pop_mza_gdf=sample_block_gdf_with_nans
        )
        
        result_nan_count = result.isna().sum().sum()
        assert result_nan_count < original_nan_count

    def test_calculate_censo_nan_values_preserves_structure(self, sample_ageb_gdf, sample_block_gdf_with_nans):
        """Test that original data structure is preserved."""
        result = census_mx.calculate_censo_nan_values(
            pop_ageb_gdf=sample_ageb_gdf,
            pop_mza_gdf=sample_block_gdf_with_nans
        )
        
        # Check key columns are preserved
        assert 'cvegeo' in result.columns
        assert 'cve_ageb' in result.columns
        assert 'pobtot' in result.columns
        assert 'geometry' in result.columns

    def test_calculate_censo_nan_values_empty_inputs(self):
        """Test handling of empty inputs."""
        empty_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        
        with pytest.raises(ValueError):
            census_mx.calculate_censo_nan_values(
                pop_ageb_gdf=empty_gdf,
                pop_mza_gdf=empty_gdf
            )

    def test_calculate_censo_nan_values_column_validation(self, sample_ageb_gdf):
        """Test validation of required columns."""
        # Create block GDF missing required columns
        bad_block_gdf = gpd.GeoDataFrame({
            'geometry': [Point(0, 0)]
        }, crs='EPSG:4326')
        
        with pytest.raises(KeyError):
            census_mx.calculate_censo_nan_values(
                pop_ageb_gdf=sample_ageb_gdf,
                pop_mza_gdf=bad_block_gdf
            )

    def test_calculate_censo_nan_values_demographic_relationships(self, sample_ageb_gdf, sample_block_gdf_with_nans):
        """Test that demographic relationships are maintained."""
        result = census_mx.calculate_censo_nan_values(
            pop_ageb_gdf=sample_ageb_gdf,
            pop_mza_gdf=sample_block_gdf_with_nans
        )
        
        # Check POBTOT = POBFEM + POBMAS where both are not NaN
        for idx, row in result.iterrows():
            if pd.notna(row['pobfem']) and pd.notna(row['pobmas']):
                total = row['pobfem'] + row['pobmas']
                assert abs(total - row['pobtot']) < 1  # Allow for rounding

    # Test CensusColumns configuration
    def test_census_columns_config(self):
        """Test CensusColumns configuration class."""
        config = census_mx.CensusColumns()
        
        assert 'POBFEM' in config.DEMOGRAPHIC_COLUMNS
        assert 'POBMAS' in config.DEMOGRAPHIC_COLUMNS
        assert 'REL_H_M' in config.AGEB_EXCLUDED_COLUMNS
        assert config.AGEB_ID_COLUMN == 'CVE_AGEB'
        assert config.BLOCK_ID_COLUMN == 'CVEGEO'
        assert config.TOTAL_POPULATION_COLUMN == 'POBTOT'

    def test_census_columns_custom_config(self, sample_ageb_gdf, sample_block_gdf_with_nans):
        """Test using custom CensusColumns configuration."""
        custom_config = census_mx.CensusColumns()
        
        result = census_mx.calculate_censo_nan_values(
            pop_ageb_gdf=sample_ageb_gdf,
            pop_mza_gdf=sample_block_gdf_with_nans,
            columns_config=custom_config
        )
        
        assert isinstance(result, gpd.GeoDataFrame)


# Simple test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])








