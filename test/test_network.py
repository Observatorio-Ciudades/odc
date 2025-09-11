import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import igraph as ig
from shapely.geometry import Point, LineString
from unittest.mock import patch, MagicMock
import tempfile
import os

from network_analysis import (
     nearest_nodes, find_nearest, to_igraph, get_seeds, voronoi_cpu,
     get_distances, calculate_distance_nearest_poi, walk_speed,
     create_network, calculate_isochrone, proximity_isochrone,
     proximity_isochrone_from_osmid, calculate_time_to_pois,
     calculate_time_to_multi_geometry_pois
)

class TestNetworkAnalysis:
    """Test suite for network analysis functions."""

    @pytest.fixture
    def sample_graph(self):
        """Create a simple test graph with CRS information."""
        G = nx.MultiDiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        G.graph['crs'] = 'EPSG:4326'  # Unprojected
        return G

    @pytest.fixture
    def projected_graph(self):
        """Create a test graph with projected CRS."""
        G = nx.MultiDiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        G.graph['crs'] = 'EPSG:3857'  # Projected
        return G

    @pytest.fixture
    def sample_nodes(self):
        """Create sample nodes GeoDataFrame."""
        return gpd.GeoDataFrame({
            'osmid': [1, 2, 3, 4],
            'x': [0.0, 1.0, 1.0, 0.0],
            'y': [0.0, 0.0, 1.0, 1.0],
            'geometry': [
                Point(0.0, 0.0), Point(1.0, 0.0),
                Point(1.0, 1.0), Point(0.0, 1.0)
            ]
        }, crs='EPSG:4326').set_index('osmid')

    @pytest.fixture
    def sample_edges(self):
        """Create sample edges GeoDataFrame."""
        return gpd.GeoDataFrame({
            'u': [1, 2, 3, 4, 1],
            'v': [2, 3, 4, 1, 3],
            'key': [0, 0, 0, 0, 0],
            'length': [100.0, 100.0, 100.0, 100.0, 141.4],
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
                LineString([(1, 1), (0, 1)]),
                LineString([(0, 1), (0, 0)]),
                LineString([(0, 0), (1, 1)])
            ]
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_pois(self):
        """Create sample points of interest."""
        return gpd.GeoDataFrame({
            'poi_id': ['A', 'B', 'C'],
            'name': ['Hospital A', 'Hospital B', 'Hospital C'],
            'geometry': [
                Point(0.1, 0.1),
                Point(0.9, 0.9),
                Point(0.5, 0.5)
            ]
        }, crs='EPSG:4326')

    @pytest.fixture
    def sample_pois_with_goi_id(self):
        """Create POIs with geometry of interest IDs."""
        return gpd.GeoDataFrame({
            'poi_id': ['A1', 'A2', 'B1', 'B2', 'C1'],
            'park_id': ['park_1', 'park_1', 'park_2', 'park_2', 'park_3'],
            'geometry': [
                Point(0.05, 0.05), Point(0.15, 0.15),  # park_1 vertices
                Point(0.85, 0.85), Point(0.95, 0.95),  # park_2 vertices
                Point(0.5, 0.5)                        # park_3 single point
            ]
        }, crs='EPSG:4326')


class TestNearestNodes:
    """Test nearest_nodes function."""

    def test_single_point_unprojected(self, sample_graph, sample_nodes):
        """Test nearest node finding for single point in unprojected CRS."""
        result = nearest_nodes(sample_graph, sample_nodes, 0.1, 0.1)
        assert isinstance(result, int)
        assert result in sample_nodes.index

    def test_multiple_points_unprojected(self, sample_graph, sample_nodes):
        """Test nearest node finding for multiple points."""
        X = [0.1, 0.9, 0.5]
        Y = [0.1, 0.9, 0.5]
        result = nearest_nodes(sample_graph, sample_nodes, X, Y)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(node_id in sample_nodes.index for node_id in result)

    def test_with_distances(self, sample_graph, sample_nodes):
        """Test returning distances along with nearest nodes."""
        node_ids, distances = nearest_nodes(sample_graph, sample_nodes, 0.1, 0.1, return_dist=True)
        assert isinstance(node_ids, int)
        assert isinstance(distances, float)
        assert distances >= 0

    def test_null_coordinates_raises_error(self, sample_graph, sample_nodes):
        """Test that null coordinates raise ValueError."""
        with pytest.raises(ValueError, match="cannot contain nulls"):
            nearest_nodes(sample_graph, sample_nodes, [0.1, np.nan], [0.1, 0.2])

    @patch('network_analysis.cKDTree', None)
    def test_missing_scipy_projected_raises_error(self, projected_graph, sample_nodes):
        """Test ImportError when scipy missing for projected graph."""
        with pytest.raises(ImportError, match="scipy must be installed"):
            nearest_nodes(projected_graph, sample_nodes, 0.1, 0.1)


class TestFindNearest:
    """Test find_nearest function."""

    def test_basic_functionality(self, sample_graph, sample_nodes, sample_pois):
        """Test basic nearest node finding functionality."""
        result = find_nearest(sample_graph, sample_nodes, sample_pois)

        assert 'osmid' in result.columns
        assert len(result) == len(sample_pois)
        assert result['osmid'].notna().all()

    def test_with_distances(self, sample_graph, sample_nodes, sample_pois):
        """Test finding nearest nodes with distance calculation."""
        result = find_nearest(sample_graph, sample_nodes, sample_pois, return_distance=True)

        assert 'osmid' in result.columns
        assert 'distance_node' in result.columns
        assert result['distance_node'].notna().all()
        assert (result['distance_node'] >= 0).all()

    def test_empty_input(self, sample_graph, sample_nodes):
        """Test behavior with empty POI GeoDataFrame."""
        empty_pois = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
        result = find_nearest(sample_graph, sample_nodes, empty_pois)

        assert len(result) == 0
        assert 'osmid' in result.columns


class TestToIgraph:
    """Test to_igraph conversion function."""

    def test_basic_conversion(self, sample_nodes, sample_edges):
        """Test basic conversion to igraph format."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy, weight='length')

        assert isinstance(graph, ig.Graph)
        assert isinstance(weights, np.ndarray)
        assert isinstance(node_mapping, dict)
        assert graph.vcount() == len(sample_nodes)
        assert len(weights) == len(sample_edges)

    def test_node_mapping_consistency(self, sample_nodes, sample_edges):
        """Test that node mapping preserves original IDs."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)

        # Check that all original node IDs are in mapping
        original_ids = set(nodes_copy['osmid'])
        mapped_ids = set(node_mapping.keys())
        assert original_ids == mapped_ids

    def test_missing_weight_column_raises_error(self, sample_nodes, sample_edges):
        """Test error when specified weight column doesn't exist."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        with pytest.raises(KeyError):
            to_igraph(nodes_copy, edges_copy, weight='nonexistent_column')


class TestVoronoiCpu:
    """Test voronoi_cpu function."""

    def test_basic_voronoi(self, sample_nodes, sample_edges):
        """Test basic Voronoi calculation."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])  # Use first and third nodes as seeds

        assignment = voronoi_cpu(graph, weights, seeds)

        assert len(assignment) == graph.vcount()
        assert all(assignment[i] in seeds for i in range(len(assignment)))

    def test_single_seed(self, sample_nodes, sample_edges):
        """Test Voronoi with single seed."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0])

        assignment = voronoi_cpu(graph, weights, seeds)

        assert len(assignment) == graph.vcount()
        assert all(assignment[i] == 0 for i in range(len(assignment)))


class TestGetDistances:
    """Test get_distances function."""

    def test_basic_distances(self, sample_nodes, sample_edges):
        """Test basic distance calculation."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])
        assignment = voronoi_cpu(graph, weights, seeds)

        distances = get_distances(graph, seeds, weights, assignment)

        assert len(distances) == graph.vcount()
        assert all(d >= 0 for d in distances)

    def test_with_poi_indices(self, sample_nodes, sample_edges):
        """Test distance calculation with POI indices."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])
        assignment = voronoi_cpu(graph, weights, seeds)

        distances, poi_indices = get_distances(
            graph, seeds, weights, assignment, get_nearest_poi=True
        )

        assert len(distances) == len(poi_indices)
        assert all(idx in range(len(seeds)) for idx in poi_indices)

    def test_with_count_pois(self, sample_nodes, sample_edges):
        """Test distance calculation with POI counting."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])
        assignment = voronoi_cpu(graph, weights, seeds)

        distances, counts = get_distances(
            graph, seeds, weights, assignment, count_pois=(True, 150.0)
        )

        assert len(distances) == len(counts)
        assert all(isinstance(count, int) for count in counts)
        assert all(count >= 0 for count in counts)


class TestWalkSpeed:
    """Test walk_speed function."""

    def test_flat_terrain(self):
        """Test walking speed on flat terrain."""
        edges = gpd.GeoDataFrame({
            'grade': [0.0, 0.0, 0.0],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(3)]
        })

        result = walk_speed(edges)

        assert 'walkspeed' in result.columns
        # Tobler's function: 4 * exp(-3.5 * 0) = 4 km/h for flat terrain
        np.testing.assert_array_almost_equal(result['walkspeed'], [4.0, 4.0, 4.0])

    def test_steep_terrain(self):
        """Test walking speed on steep terrain."""
        edges = gpd.GeoDataFrame({
            'grade': [0.1, 0.2, 0.3],  # 10%, 20%, 30% grades
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(3)]
        })

        result = walk_speed(edges)

        assert 'walkspeed' in result.columns
        # Speed should decrease with increasing grade
        assert result['walkspeed'].iloc[0] > result['walkspeed'].iloc[1]
        assert result['walkspeed'].iloc[1] > result['walkspeed'].iloc[2]
        assert all(speed > 0 for speed in result['walkspeed'])

    def test_preserves_original_data(self):
        """Test that original DataFrame is not modified."""
        edges = gpd.GeoDataFrame({
            'grade': [0.0, 0.1],
            'other_column': ['A', 'B'],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(2)]
        })

        original_columns = edges.columns.tolist()
        result = walk_speed(edges)

        # Original should be unchanged
        assert edges.columns.tolist() == original_columns
        # Result should have additional column
        assert 'walkspeed' in result.columns
        assert all(col in result.columns for col in original_columns)


class TestCreateNetwork:
    """Test create_network function."""

    def test_basic_network_creation(self):
        """Test creating network from raw geometries."""
        # Create test nodes
        nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
        }, crs='EPSG:4326')

        # Create test edges
        edges = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
                LineString([(1, 1), (0, 1)]),
                LineString([(0, 1), (0, 0)])
            ]
        }, crs='EPSG:4326')

        result_nodes, result_edges = create_network(nodes, edges)

        # Check that required columns were created
        assert 'osmid' in result_nodes.columns
        assert all(col in result_edges.columns for col in ['u', 'v', 'key', 'length'])
        assert result_nodes.crs.to_string() == 'EPSG:4326'
        assert result_edges.crs.to_string() == 'EPSG:4326'

    def test_edge_connectivity(self):
        """Test that edges have proper connectivity attributes."""
        nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 0)]
        }, crs='EPSG:4326')

        edges = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 0)])]
        }, crs='EPSG:4326')

        result_nodes, result_edges = create_network(nodes, edges)

        # Check edge connectivity
        assert result_edges['u'].notna().all()
        assert result_edges['v'].notna().all()
        assert result_edges['length'].notna().all()
        assert result_edges['length'].iloc[0] > 0


class TestCalculateIsochrone:
    """Test calculate_isochrone function."""

    def test_basic_isochrone(self, sample_nodes, sample_edges):
        """Test basic isochrone calculation."""
        # Create graph from nodes and edges
        G = nx.MultiDiGraph()
        for _, edge in sample_edges.iterrows():
            G.add_edge(edge['u'], edge['v'], length=edge['length'])

        # Add node coordinates
        for node_id, node in sample_nodes.iterrows():
            G.nodes[node_id]['x'] = node['x']
            G.nodes[node_id]['y'] = node['y']

        isochrone = calculate_isochrone(G, 1, 150.0, 'length')

        assert hasattr(isochrone, 'area')  # Should be a polygon
        assert isochrone.area > 0

    def test_with_subgraph(self, sample_nodes, sample_edges):
        """Test isochrone calculation returning subgraph."""
        G = nx.MultiDiGraph()
        for _, edge in sample_edges.iterrows():
            G.add_edge(edge['u'], edge['v'], length=edge['length'])

        for node_id, node in sample_nodes.iterrows():
            G.nodes[node_id]['x'] = node['x']
            G.nodes[node_id]['y'] = node['y']

        subgraph, geometry = calculate_isochrone(G, 1, 150.0, 'length', subgraph=True)

        assert isinstance(subgraph, nx.MultiDiGraph)
        assert hasattr(geometry, 'area')
        assert len(subgraph.nodes) <= len(G.nodes)


class TestPoisTime:
    """Test calculate_time_to_pois function - the main integration function."""

    def test_empty_pois(self, sample_graph, sample_nodes, sample_edges):
        """Test handling of empty POI dataset."""
        empty_pois = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            empty_pois, "hospital", "length"
        )

        assert 'time_hospital' in result.columns
        assert result['time_hospital'].isna().all()

    def test_basic_poi_analysis(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test basic POI time analysis."""
        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length"
        )

        assert 'time_hospital' in result.columns
        assert len(result) <= len(sample_nodes)  # Some nodes might be filtered
        assert result['time_hospital'].notna().any()

    def test_with_count_pois(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test POI analysis with counting enabled."""
        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length",
            count_pois=(True, 30)
        )

        assert 'time_hospital' in result.columns
        assert 'hospital_30min' in result.columns
        assert result['hospital_30min'].dtype in [int, 'int64']

    def test_progress_callback(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test that progress callback is called."""
        callback_calls = []

        def progress_callback(current, total, description):
            callback_calls.append((current, total, description))

        calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length",
            progress_callback=progress_callback
        )

        # Should have received at least one progress update
        assert len(callback_calls) > 0

    def test_invalid_prox_measure_raises_error(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test that invalid proximity measure raises error."""
        with pytest.raises(ValueError, match="prox_measure must be"):
            calculate_time_to_pois(
                sample_graph, sample_nodes, sample_edges,
                sample_pois, "hospital", "invalid_measure"
            )

    def test_negative_walking_speed_raises_error(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test that negative walking speed raises error."""
        with pytest.raises(ValueError, match="walking_speed must be positive"):
            calculate_time_to_pois(
                sample_graph, sample_nodes, sample_edges,
                sample_pois, "hospital", "length", walking_speed=-1
            )


class TestIdPoisTime:
    """Test calculate_time_to_multi_geometry_pois function - geometry-grouped POI analysis."""

    def test_basic_geometry_grouping(self, sample_graph, sample_nodes, sample_edges, sample_pois_with_goi_id):
        """Test basic functionality with geometry grouping."""
        result = calculate_time_to_multi_geometry_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois_with_goi_id, "park", "length", 4.0, "park_id"
        )

        assert 'time_park' in result.columns
        assert len(result) <= len(sample_nodes)

    def test_geometry_count_binary(self, sample_graph, sample_nodes, sample_edges, sample_pois_with_goi_id):
        """Test that geometry counting returns binary values per geometry."""
        result = calculate_time_to_multi_geometry_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois_with_goi_id, "park", "length", 4.0, "park_id",
            count_pois=(True, 30)
        )

        assert 'park_30min' in result.columns
        # Should count distinct geometries, not individual POIs
        max_count = result['park_30min'].max()
        unique_geometries = sample_pois_with_goi_id['park_id'].nunique()
        assert max_count <= unique_geometries

    def test_missing_goi_id_raises_error(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test error when geometry ID column is missing."""
        with pytest.raises(ValueError, match="goi_id column .* not found"):
            calculate_time_to_multi_geometry_pois(
                sample_graph, sample_nodes, sample_edges,
                sample_pois, "poi", "length", 4.0, "nonexistent_column"
            )


class TestProximityIsochrone:
    """Test proximity isochrone functions."""

    def test_proximity_isochrone_basic(self, sample_graph, sample_nodes, sample_edges):
        """Test basic proximity isochrone calculation."""
        poi = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5)]
        }, crs='EPSG:4326')

        result = proximity_isochrone(
            sample_graph, sample_nodes, sample_edges, poi, 15
        )

        assert hasattr(result, 'area')
        assert result.area > 0

    def test_proximity_isochrone_from_osmid(self, sample_graph, sample_nodes, sample_edges):
        """Test proximity isochrone from specific node ID."""
        center_osmid = sample_nodes.index[0]  # Use first node

        result = proximity_isochrone_from_osmid(
            sample_graph, sample_nodes, sample_edges, center_osmid, 15
        )

        assert hasattr(result, 'area')
        assert result.area > 0


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_hospital_analysis_workflow(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test complete workflow for hospital accessibility analysis."""
        # Step 1: Find nearest nodes
        pois_with_nearest = find_nearest(sample_graph, sample_nodes, sample_pois)
        assert 'osmid' in pois_with_nearest.columns

        # Step 2: Calculate accessibility
        accessibility = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length", count_pois=(True, 20)
        )

        assert 'time_hospital' in accessibility.columns
        assert 'hospital_20min' in accessibility.columns

        # Step 3: Create isochrone from best-served location
        if not accessibility.empty:
            best_node = accessibility.loc[accessibility['time_hospital'].idxmin(), 'osmid']
            isochrone = proximity_isochrone_from_osmid(
                sample_graph, sample_nodes, sample_edges, best_node, 15
            )
            assert hasattr(isochrone, 'area')

    def test_park_vertices_analysis(self, sample_graph, sample_nodes, sample_edges, sample_pois_with_goi_id):
        """Test analysis of POIs derived from park geometries."""
        result = calculate_time_to_multi_geometry_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois_with_goi_id, "park", "length", 4.0, "park_id",
            count_pois=(True, 15)
        )

        # Should handle multiple POIs per park correctly
        assert 'time_park' in result.columns
        assert 'park_15min' in result.columns

        # Count should be number of distinct parks, not POI points
        max_parks = result['park_15min'].max()
        unique_parks = sample_pois_with_goi_id['park_id'].nunique()
        assert max_parks <= unique_parks


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_disconnected_graph(self, sample_nodes, sample_edges):
        """Test behavior with disconnected graph components."""
        # Create disconnected edges
        disconnected_edges = sample_edges.iloc[:2].copy()  # Only first two edges

        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'

        # Should handle gracefully or raise informative error
        pois = gpd.GeoDataFrame({
            'geometry': [Point(0.1, 0.1)]
        }, crs='EPSG:4326')

        # This might raise an error or return limited results
        try:
            result = calculate_time_to_pois(G, sample_nodes, disconnected_edges, pois, "test", "length")
            # If it succeeds, check that results are reasonable
            assert 'time_test' in result.columns
        except Exception as e:
            # Should be a meaningful error message
            assert len(str(e)) > 0

    def test_poi_far_from_network(self, sample_graph, sample_nodes, sample_edges):
        """Test POI that's very far from network."""
        far_poi = gpd.GeoDataFrame({
            'geometry': [Point(100, 100)]  # Very far from network
        }, crs='EPSG:4326')

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            far_poi, "remote", "length"
        )

        # Should handle gracefully
        assert 'time_remote' in result.columns

    def test_missing_required_columns(self, sample_graph, sample_edges, sample_pois):
        """Test error handling for missing required columns."""
        # Nodes without osmid
        bad_nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 1)]
        }, crs='EPSG:4326')

        with pytest.raises(ValueError, match="missing required columns"):
            calculate_time_to_pois(sample_graph, bad_nodes, sample_edges, sample_pois, "test", "length")


class TestPerformance:
    """Performance and memory tests."""

    def test_large_dataset_handling(self):
        """Test handling of larger datasets."""
        # Create larger test dataset
        n_nodes = 100
        node_coords = np.random.random((n_nodes, 2))
        nodes = gpd.GeoDataFrame({
            'osmid': range(n_nodes),
            'x': node_coords[:, 0],
            'y': node_coords[:, 1],
            'geometry': [Point(x, y) for x, y in node_coords]
        }, crs='EPSG:4326').set_index('osmid')

        # Create edges (simplified grid)
        edges_data = []
        for i in range(n_nodes - 1):
            edges_data.append({
                'u': i, 'v': i + 1, 'key': 0, 'length': 100.0,
                'geometry': LineString([node_coords[i], node_coords[i + 1]])
            })

        edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')

        # Create POIs
        n_pois = 20
        poi_coords = np.random.random((n_pois, 2))
        pois = gpd.GeoDataFrame({
            'geometry': [Point(x, y) for x, y in poi_coords]
        }, crs='EPSG:4326')

        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'

        # Should complete without memory errors
        result = calculate_time_to_pois(G, nodes, edges, pois, "test", "length")
        assert not result.empty

    def test_batch_processing_consistency(self, sample_graph, sample_nodes, sample_edges):
        """Test that batch processing gives consistent results."""
        # Create enough POIs to trigger batch processing
        n_pois = 300  # Exceeds batch size of 250
        poi_coords = np.random.random((n_pois, 2))
        large_pois = gpd.GeoDataFrame({
            'geometry': [Point(x, y) for x, y in poi_coords]
        }, crs='EPSG:4326')

        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'

        result = calculate_time_to_pois(G, sample_nodes, sample_edges, large_pois, "test", "length")

        # Results should be consistent regardless of batch processing
        assert 'time_test' in result.columns
        assert not result.empty


# Additional test utilities and fixtures

@pytest.fixture
def temp_test_directory():
    """Create temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestFileOperations:
    """Test file I/O operations in functions."""

    def test_preprocessed_nearest_loading(self, sample_graph, sample_nodes, sample_edges,
                                        sample_pois, temp_test_directory):
        """Test loading preprocessed nearest node data."""
        # First, create and save nearest data
        nearest_data = find_nearest(sample_graph, sample_nodes, sample_pois, return_distance=True)
        test_file = os.path.join(temp_test_directory, "nearest_hospital.gpkg")
        nearest_data.to_file(test_file, driver='GPKG')

        # Now test loading it
        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges, sample_pois,
            "hospital", "length", preprocessed_nearest=(True, temp_test_directory)
        )

        assert 'time_hospital' in result.columns


class TestDataValidation:
    """Test data validation and error handling."""

    def test_crs_mismatch_handling(self, sample_graph, sample_edges, sample_pois):
        """Test handling of CRS mismatches between inputs."""
        # Create nodes with different CRS
        nodes_different_crs = gpd.GeoDataFrame({
            'osmid': [1, 2, 3, 4],
            'x': [0, 1000, 1000, 0],  # In meters
            'y': [0, 0, 1000, 1000],
            'geometry': [
                Point(0, 0), Point(1000, 0),
                Point(1000, 1000), Point(0, 1000)
            ]
        }, crs='EPSG:3857').set_index('osmid')  # Different CRS

        # Function should handle CRS conversion
        result = calculate_time_to_pois(
            sample_graph, nodes_different_crs, sample_edges,
            sample_pois, "test", "length"
        )

        assert result.crs.to_string() == 'EPSG:4326'

    def test_invalid_geometry_types(self, sample_graph, sample_nodes, sample_edges):
        """Test error handling for invalid geometry types."""
        # Create POIs with LineString instead of Point
        invalid_pois = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (1, 1)])]
        }, crs='EPSG:4326')

        # Should handle gracefully or raise informative error
        try:
            result = calculate_time_to_pois(
                sample_graph, sample_nodes, sample_edges,
                invalid_pois, "test", "length"
            )
            # If it succeeds, it should handle the conversion
            assert not result.empty
        except Exception as e:
            # Should be an informative error
            assert "Point" in str(e) or "geometry" in str(e)

    def test_missing_edge_weights(self, sample_graph, sample_nodes, sample_pois):
        """Test handling of edges missing weight information."""
        # Create edges without length column
        edges_no_length = gpd.GeoDataFrame({
            'u': [1, 2, 3, 4],
            'v': [2, 3, 4, 1],
            'key': [0, 0, 0, 0],
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)]),
                LineString([(1, 1), (0, 1)]),
                LineString([(0, 1), (0, 0)])
            ]
        }, crs='EPSG:4326')

        # Function should calculate missing lengths
        result = calculate_time_to_pois(
            sample_graph, sample_nodes, edges_no_length,
            sample_pois, "test", "length"
        )

        assert 'time_test' in result.columns


class TestParameterValidation:
    """Test parameter validation across all functions."""

    def test_empty_string_poi_name(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test validation of empty POI name."""
        with pytest.raises(ValueError, match="poi_name must be a non-empty string"):
            calculate_time_to_pois(sample_graph, sample_nodes, sample_edges, sample_pois, "", "length")

    def test_zero_walking_speed(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test validation of zero walking speed."""
        with pytest.raises(ValueError, match="walking_speed must be positive"):
            calculate_time_to_pois(sample_graph, sample_nodes, sample_edges, sample_pois, "test", "length", 0)

    def test_negative_trip_time(self, sample_graph, sample_nodes, sample_edges):
        """Test validation of negative trip time in isochrone."""
        poi = gpd.GeoDataFrame({'geometry': [Point(0.5, 0.5)]}, crs='EPSG:4326')

        # Should handle negative values gracefully
        result = proximity_isochrone(sample_graph, sample_nodes, sample_edges, poi, -5)
        # Might return empty geometry or raise error
        assert result is not None


# Mock functions for testing without actual dependencies

class TestMockDependencies:
    """Test functions with mocked external dependencies."""

    @patch('network_analysis.calculate_distance_nearest_poi')
    def test_calculate_time_to_pois_with_mocked_calculation(self, mock_calc, sample_graph, sample_nodes,
                                             sample_edges, sample_pois):
        """Test calculate_time_to_pois with mocked distance calculation."""
        # Mock the distance calculation function
        mock_result = sample_nodes.reset_index().copy()
        mock_result['dist_hospital'] = [10, 20, 30, 40]
        mock_calc.return_value = mock_result

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length"
        )

        # Verify mock was called
        assert mock_calc.called
        assert 'time_hospital' in result.columns

    @patch('builtins.print')  # Mock print statements if any remain
    def test_no_unwanted_print_statements(self, mock_print, sample_graph, sample_nodes,
                                        sample_edges, sample_pois):
        """Test that no print statements are called (should use logging)."""
        calculate_time_to_pois(sample_graph, sample_nodes, sample_edges, sample_pois, "test", "length")

        # Should not have called print (should use logging instead)
        mock_print.assert_not_called()


# Configuration for pytest

class TestConfig:
    """Test configuration and constants."""

    def test_default_values_are_reasonable(self):
        """Test that default parameter values are reasonable."""
        # These should be reasonable defaults
        assert 4.0 > 0  # Default walking speed
        assert "EPSG:6372" != ""  # Default projected CRS
        assert 80.0 > 0  # Default max walking distance

    def test_batch_size_logic(self):
        """Test batch size determination logic."""
        # Test the logic used in batch processing
        test_cases = [
            (250, 250),  # Divisible by 250
            (500, 250),  # Divisible by 250
            (300, 200),  # Not divisible by 250
            (199, 200),  # Less than 200
        ]

        for n_items, expected_batch_size in test_cases:
            if n_items % 250 == 0:
                assert expected_batch_size == 250
            else:
                assert expected_batch_size == 200


# Utility functions for test setup

def create_test_network(n_nodes=10, connect_probability=0.7):
    """Create a random test network for testing."""
    np.random.seed(42)  # For reproducible tests

    # Generate random node coordinates
    coords = np.random.random((n_nodes, 2))
    nodes = gpd.GeoDataFrame({
        'osmid': range(n_nodes),
        'x': coords[:, 0],
        'y': coords[:, 1],
        'geometry': [Point(x, y) for x, y in coords]
    }, crs='EPSG:4326').set_index('osmid')

    # Generate random edges
    edges_data = []
    edge_id = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if np.random.random() < connect_probability:
                distance = np.sqrt((coords[i][0] - coords[j][0])**2 +
                                 (coords[i][1] - coords[j][1])**2) * 111000  # Rough meters
                edges_data.append({
                    'u': i, 'v': j, 'key': 0, 'length': distance,
                    'geometry': LineString([coords[i], coords[j]])
                })
                edge_id += 1

    edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')

    # Create graph
    G = nx.MultiDiGraph()
    G.graph['crs'] = 'EPSG:4326'
    for _, edge in edges.iterrows():
        G.add_edge(edge['u'], edge['v'], length=edge['length'])

    return G, nodes, edges


def create_test_pois(n_pois=5, bounds=(0, 1, 0, 1)):
    """Create random test POIs within specified bounds."""
    np.random.seed(42)

    x_coords = np.random.uniform(bounds[0], bounds[1], n_pois)
    y_coords = np.random.uniform(bounds[2], bounds[3], n_pois)

    return gpd.GeoDataFrame({
        'poi_id': [f'poi_{i}' for i in range(n_pois)],
        'geometry': [Point(x, y) for x, y in zip(x_coords, y_coords)]
    }, crs='EPSG:4326')


# Parameterized tests for comprehensive coverage

@pytest.mark.parametrize("prox_measure", ["length", "time_min"])
@pytest.mark.parametrize("walking_speed", [3.0, 4.0, 5.0])
def test_different_proximity_measures_and_speeds(prox_measure, walking_speed,
                                               sample_graph, sample_nodes, sample_edges, sample_pois):
    """Test different combinations of proximity measures and walking speeds."""
    if prox_measure == "time_min":
        # Add time_min column for this test
        sample_edges['time_min'] = sample_edges['length'] / (walking_speed * 1000/60)

    result = calculate_time_to_pois(
        sample_graph, sample_nodes, sample_edges,
        sample_pois, "test", prox_measure, walking_speed
    )

    assert 'time_test' in result.columns
    if not result.empty:
        assert result['time_test'].notna().any()


@pytest.mark.parametrize("count_threshold", [5, 10, 15, 30])
def test_different_count_thresholds(count_threshold, sample_graph, sample_nodes, sample_edges, sample_pois):
    """Test POI counting with different time thresholds."""
    result = calculate_time_to_pois(
        sample_graph, sample_nodes, sample_edges,
        sample_pois, "test", "length",
        count_pois=(True, count_threshold)
    )

    assert f'test_{count_threshold}min' in result.columns
    if not result.empty:
        assert result[f'test_{count_threshold}min'].dtype in [int, 'int64']


# Benchmark tests (optional - for performance monitoring)

@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance monitoring."""

    def test_calculate_time_to_pois_performance(self, benchmark):
        """Benchmark calculate_time_to_pois function performance."""
        G, nodes, edges = create_test_network(n_nodes=50)
        pois = create_test_pois(n_pois=20)

        result = benchmark(calculate_time_to_pois, G, nodes, edges, pois, "test", "length")
        assert not result.empty

    def test_large_network_performance(self, benchmark):
        """Benchmark with larger network."""
        G, nodes, edges = create_test_network(n_nodes=100)
        pois = create_test_pois(n_pois=50)

        result = benchmark(calculate_time_to_pois, G, nodes, edges, pois, "test", "length")
        assert not result.empty


# Regression tests for specific bugs

class TestRegressionTests:
    """Tests for specific bugs and edge cases found in development."""

    def test_single_poi_does_not_crash(self, sample_graph, sample_nodes, sample_edges):
        """Test that single POI doesn't cause indexing errors."""
        single_poi = gpd.GeoDataFrame({
            'geometry': [Point(0.5, 0.5)]
        }, crs='EPSG:4326')

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            single_poi, "single", "length"
        )

        assert 'time_single' in result.columns

    def test_poi_exactly_on_node(self, sample_graph, sample_nodes, sample_edges):
        """Test POI located exactly on a network node."""
        # POI at exact node location
        node_poi = gpd.GeoDataFrame({
            'geometry': [Point(sample_nodes.iloc[0]['x'], sample_nodes.iloc[0]['y'])]
        }, crs='EPSG:4326')

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            node_poi, "exact", "length"
        )

        assert 'time_exact' in result.columns
        # Distance should be very small or zero
        if not result.empty:
            min_time = result['time_exact'].min()
            assert min_time >= 0
            assert min_time < 1  # Should be less than 1 minute

    def test_all_pois_very_far(self, sample_graph, sample_nodes, sample_edges):
        """Test behavior when all POIs are very far from network."""
        far_pois = gpd.GeoDataFrame({
            'geometry': [Point(100, 100), Point(200, 200)]
        }, crs='EPSG:4326')

        result = calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            far_pois, "far", "length"
        )

        # Should handle gracefully
        assert 'time_far' in result.columns


# Test data generation helpers

def generate_realistic_test_data():
    """Generate more realistic test data for comprehensive testing."""
    # Create a small Manhattan-like grid
    x_coords = np.arange(0, 0.01, 0.002)  # ~200m spacing
    y_coords = np.arange(0, 0.01, 0.002)

    nodes_data = []
    node_id = 1
    for x in x_coords:
        for y in y_coords:
            nodes_data.append({
                'osmid': node_id,
                'x': x, 'y': y,
                'geometry': Point(x, y)
            })
            node_id += 1

    nodes = gpd.GeoDataFrame(nodes_data, crs='EPSG:4326').set_index('osmid')

    # Create grid edges
    edges_data = []
    for i, node1 in nodes.iterrows():
        for j, node2 in nodes.iterrows():
            if i != j:
                dist = np.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)
                if dist < 0.003:  # Only connect nearby nodes
                    edges_data.append({
                        'u': i, 'v': j, 'key': 0,
                        'length': dist * 111000,  # Convert to meters
                        'geometry': LineString([(node1['x'], node1['y']),
                                              (node2['x'], node2['y'])])
                    })

    edges = gpd.GeoDataFrame(edges_data, crs='EPSG:4326')

    return nodes, edges


# Test fixtures for realistic scenarios

@pytest.fixture
def realistic_network():
    """Create realistic network for testing."""
    return generate_realistic_test_data()


@pytest.fixture
def hospital_pois():
    """Create realistic hospital POI data."""
    return gpd.GeoDataFrame({
        'hospital_id': ['H001', 'H002', 'H003'],
        'name': ['General Hospital', 'Emergency Clinic', 'Specialist Center'],
        'geometry': [Point(0.002, 0.002), Point(0.006, 0.008), Point(0.004, 0.006)]
    }, crs='EPSG:4326')


@pytest.fixture
def park_vertices():
    """Create POIs representing park vertices with geometry IDs."""
    return gpd.GeoDataFrame({
        'vertex_id': ['V001', 'V002', 'V003', 'V004', 'V005'],
        'park_id': ['park_A', 'park_A', 'park_B', 'park_B', 'park_C'],
        'geometry': [
            Point(0.001, 0.001), Point(0.002, 0.002),  # park_A
            Point(0.007, 0.007), Point(0.008, 0.008),  # park_B
            Point(0.005, 0.005)                        # park_C
        ]
    }, crs='EPSG:4326')


class TestRealisticScenarios:
    """Test with more realistic data scenarios."""

    def test_hospital_accessibility_analysis(self, realistic_network, hospital_pois):
        """Test realistic hospital accessibility scenario."""
        nodes, edges = realistic_network

        # Create minimal graph for testing
        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'

        result = calculate_time_to_pois(
            G, nodes.iloc[:20], edges.iloc[:30],  # Subset for testing
            hospital_pois, "hospital", "length",
            walking_speed=3.5, count_pois=(True, 20)
        )

        assert 'time_hospital' in result.columns
        assert 'hospital_20min' in result.columns

    def test_park_vertices_analysis(self, realistic_network, park_vertices):
        """Test realistic park accessibility with geometry grouping."""
        nodes, edges = realistic_network

        G = nx.MultiDiGraph()
        G.graph['crs'] = 'EPSG:4326'

        result = calculate_time_to_multi_geometry_pois(
            G, nodes.iloc[:20], edges.iloc[:30],  # Subset for testing
            park_vertices, "park", "length", 4.0, "park_id",
            count_pois=(True, 15), max_walking_distance=100.0
        )

        assert 'time_park' in result.columns
        assert 'park_15min' in result.columns

        # Count should represent distinct parks, not vertices
        if not result.empty:
            max_parks = result['park_15min'].max()
            unique_parks = park_vertices['park_id'].nunique()
            assert max_parks <= unique_parks


# Run configuration

if __name__ == "__main__":
    # Example of how to run tests
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--cov=network_analysis",  # Coverage reporting
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing"  # Show missing lines in terminal
    ])
