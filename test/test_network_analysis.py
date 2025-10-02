import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import igraph as ig
from shapely.geometry import Point, LineString
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
    from odc import network_analysis
else:
    from odc import network_analysis

class TestNetworkAnalysis:
    """Essential tests for network analysis functions."""

    # Test fixtures - basic data setup
    @pytest.fixture
    def sample_graph(self):
        """Simple test graph."""
        center_point = (19.3984893,-99.1417539)
        G = ox.graph_from_point(center_point, dist=500, network_type='all')
        return G

    @pytest.fixture
    def sample_nodes(self):
        """Sample nodes GeoDataFrame."""
        center_point = (19.3984893,-99.1417539)
        G = ox.graph_from_point(center_point, dist=500, network_type='all')
        nodes, _ = ox.graph_to_gdfs(G)
        nodes = nodes.set_crs('EPSG:4326')

        return nodes

    @pytest.fixture
    def sample_edges(self):
        """Sample edges GeoDataFrame."""
        center_point = (19.3984893,-99.1417539)
        G = ox.graph_from_point(center_point, dist=500, network_type='all')
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.set_crs('EPSG:4326')
        edges = edges.reset_index()

        return edges

    @pytest.fixture
    def sample_pois(self):
        """Sample points of interest."""
        return gpd.GeoDataFrame({
            'poi_id': ['A', 'B'],
            'geometry': [Point(19.3987151,-99.1439583),
                Point(19.3934174,-99.1392214)]
        }, crs='EPSG:4326')

    # Test nearest_nodes function
    def test_nearest_nodes_basic(self, sample_graph, sample_nodes):
        """Test basic nearest node functionality."""
        result = network_analysis.nearest_nodes(sample_graph, sample_nodes, -99.1439583, 19.3987151)
        assert isinstance(result, int)
        assert result in sample_nodes.index

    def test_nearest_nodes_with_distance(self, sample_graph, sample_nodes):
        """Test nearest nodes returning distances."""
        node_id, distance = network_analysis.nearest_nodes(sample_graph, sample_nodes, -99.1439583, 19.3987151, return_dist=True)
        assert isinstance(node_id, int)
        assert isinstance(distance, float)
        assert distance >= 0

    # Test find_nearest function
    def test_find_nearest_basic(self, sample_graph, sample_nodes, sample_pois):
        """Test basic find nearest functionality."""
        result = network_analysis.find_nearest_point_to_node(sample_graph, 
                                                             sample_nodes,
                                                             sample_pois)
        assert 'osmid' in result.columns
        assert len(result) == len(sample_pois)
        assert result['osmid'].notna().all()

    # Test to_igraph function
    def test_to_igraph_conversion(self, sample_nodes, sample_edges):
        """Test conversion to igraph format."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = network_analysis.to_igraph(nodes_copy, edges_copy, weight='length')

        assert isinstance(graph, ig.Graph)
        assert isinstance(weights, np.ndarray)
        assert isinstance(node_mapping, dict)
        assert graph.vcount() == len(sample_nodes)

    # Test get_seeds function
    def test_get_seeds_basic(self, sample_nodes, sample_edges, sample_pois, sample_graph):
        """Test seed generation for Voronoi calculation."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = network_analysis.to_igraph(nodes_copy, edges_copy)

        # Create test data with osmid column
        pois_test = sample_pois.copy()
        G_copy = sample_graph.copy()
        test_gdf = network_analysis.find_nearest_point_to_node(G_copy, 
                                                               nodes_copy,
                                                               pois_test)
        seeds = network_analysis.get_seeds(test_gdf, node_mapping, 'osmid')

        assert isinstance(seeds, np.ndarray)
        assert len(seeds) == 2

    # Test voronoi_cpu function
    def test_voronoi_cpu_basic(self, sample_nodes, sample_edges):
        """Test basic Voronoi calculation."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = network_analysis.to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])  # Use first and third nodes as seeds

        assignment = network_analysis.voronoi_cpu(graph, weights, seeds)

        assert len(assignment) == graph.vcount()
        assert all(assignment[i] in seeds for i in range(len(assignment)))

    # Test get_distances function
    def test_get_distances_basic(self, sample_nodes, sample_edges):
        """Test distance calculation."""
        nodes_copy = sample_nodes.reset_index()
        edges_copy = sample_edges.copy()

        graph, weights, node_mapping = network_analysis.to_igraph(nodes_copy, edges_copy)
        seeds = np.array([0, 2])
        assignment = network_analysis.voronoi_cpu(graph, weights, seeds)

        distances = network_analysis.get_distances(graph, seeds, weights, assignment)

        assert len(distances) == graph.vcount()
        assert all(d >= 0 for d in distances)

    # Test calculate_distance_nearest_poi function
    def test_calculate_distance_nearest_poi_basic(self, sample_nodes, sample_edges, sample_pois, sample_graph):
        """Test POI distance calculation."""
        # Create test POI data with osmid assignments
        test_pois = sample_pois.copy()
        nodes_test = sample_nodes.reset_index()
        
        G_copy = sample_graph.copy()
        test_pois = network_analysis.find_nearest_point_to_node(G_copy, 
                                                               sample_nodes,
                                                               test_pois)
        
        result = network_analysis.calculate_distance_nearest_poi(
            test_pois, nodes_test, sample_edges,
            'test_poi', 'osmid', weight='length'
        )

        assert 'dist_test_poi' in result.columns
        assert len(result) <= len(sample_nodes)

    # Test walk_speed function
    def test_walk_speed_basic(self):
        """Test walking speed calculation."""
        edges = gpd.GeoDataFrame({
            'grade': [0.0, 0.1, 0.2],
            'geometry': [LineString([(0, 0), (1, 0)]) for _ in range(3)]
        })

        result = network_analysis.walk_speed(edges)

        assert 'walkspeed' in result.columns
        assert all(speed > 0 for speed in result['walkspeed'])
        # Speed should decrease with increasing grade
        assert result['walkspeed'].iloc[0] > result['walkspeed'].iloc[2]

    # Test create_network function
    def test_create_network_basic(self):
        """Test network creation from raw geometries."""
        nodes = gpd.GeoDataFrame({
            'geometry': [Point(0, 0), Point(1, 0), Point(1, 1)]
        }, crs='EPSG:4326')

        edges = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (1, 1)])
            ]
        }, crs='EPSG:4326')

        result_nodes, result_edges = network_analysis.create_network(nodes, edges)

        assert 'osmid' in result_nodes.columns
        assert all(col in result_edges.columns for col in ['u', 'v', 'key', 'length'])

    # Test calculate_isochrone function
    def test_calculate_isochrone_basic(self, sample_nodes, sample_edges, sample_graph):
        """Test isochrone calculation."""
        center_node_osmid = 268684269
        isochrone = network_analysis.calculate_isochrone(sample_graph,
                                                         center_node_osmid,
                                                         500.0, 'length')

        assert hasattr(isochrone, 'area')
        assert isochrone.area > 0

    # Test proximity_isochrone function
    def test_proximity_isochrone_basic(self, sample_graph, sample_nodes, sample_edges):
        """Test proximity-based isochrone."""
        poi = gpd.GeoDataFrame({
            'geometry': [Point(-99.1428684, 19.3976424)]
        }, crs='EPSG:4326')

        result = network_analysis.proximity_isochrone(sample_graph, sample_nodes, sample_edges, poi, 15)

        assert hasattr(result, 'area')

    # Test pois_time function (main integration function)
    def test_pois_time_basic(self, sample_graph, sample_nodes, sample_edges, sample_pois):
        """Test basic POI time analysis."""
        result = network_analysis.calculate_time_to_pois(
            sample_graph, sample_nodes, sample_edges,
            sample_pois, "hospital", "length"
        )

        assert 'time_hospital' in result.columns
        assert len(result) <= len(sample_nodes)

    def test_pois_time_empty_pois(self, sample_graph, sample_nodes, sample_edges):
        """Test handling of empty POI dataset."""
        empty_pois = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')

        result = network_analysis.calculate_time_to_pois(
            sample_graph, sample_nodes.reset_index(), sample_edges,
            empty_pois, "hospital", "length"
        )

        assert 'time_hospital' in result.columns
        assert result['time_hospital'].isna().all()

    # Test id_pois_time function (geometry-grouped analysis)
    def test_multi_geometry_pois_time_basic(self, sample_graph, sample_nodes, sample_edges):
        """Test geometry-grouped POI analysis."""
        pois_with_goi = gpd.GeoDataFrame({
            'park_id': ['park_1', 'park_1', 'park_2'],
            'geometry': [Point(-99.1415698, 19.3964740),
                Point(-99.1449053, 19.3964485),
                Point(-99.1377683, 19.4010425)]
        }, crs='EPSG:4326')

        result = network_analysis.calculate_time_to_multi_geometry_pois(
            sample_graph, sample_nodes.reset_index(), sample_edges,
            pois_with_goi, "park", "length", 4.0, "park_id"
        )

        assert 'time_park' in result.columns


# Simple test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
