import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.transform
from shapely.geometry import Point, Polygon, box

module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)
    from odc import raster
else:
    from odc import raster


class TestRasterModule:
    """Collection of sanity‑checks for raster.py."""

    @pytest.fixture(scope="function")
    def temp_dir(self):
        """Create a fresh temporary directory for each test."""
        dirpath = Path(tempfile.mkdtemp())
        yield dirpath
        shutil.rmtree(dirpath, ignore_errors=True)

    @pytest.fixture(scope="function")
    def dummy_aoi_gdf(self):
        """A tiny AOI (single square polygon) in EPSG:4326."""
        poly = box(-1.0, -1.0, 1.0, 1.0)  # 2 × 2 degree square
        return gpd.GeoDataFrame({"fid": [0], "geometry": [poly]}, crs="EPSG:4326")

    @pytest.fixture(scope="function")
    def dummy_raster(self, tmp_path):
        """
        Write a 3 × 3 raster (single band) with known values.
        Returned path points to the .tif file.
        """
        data = np.arange(9, dtype="float32").reshape((1, 3, 3))
        transform = rasterio.transform.from_origin(
            -1.0, 1.0, 1.0, 1.0
        )  # pixel size = 1°
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": 3,
            "height": 3,
            "crs": "EPSG:4326",
            "transform": transform,
        }

        raster_path = tmp_path / "dummy.tif"
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(data)

        return raster_path

    def _make_fake_items(self, dates):
        """
        Produce a list of very small mock STAC items that look enough like the
        real ones for the code paths we hit (they only need .datetime and
        .properties accessed in `available_datasets`).
        """
        items = []
        for d in dates:
            mock_item = MagicMock()
            mock_item.datetime = pd.Timestamp(d)
            mock_item.properties = {
                "s2:mgrs_tile": "31TGL",
                "s2:high_proba_clouds_percentage": 5,
                "landsat:wrs_path": 31,
                "landsat:wrs_row": 33,
                "landsat:cloud_cover_land": 5,
            }
            items.append(mock_item)
        return items

    def test_pcrasterdata_initialisation_success(self, dummy_aoi_gdf, temp_dir):
        """Happy‑path construction with all required arguments."""
        band_dict = {
            "red": [False, "red"],
            "nir": [False, "nir"],
            "quality": [False, "qa"],
        }

        p = raster.PCRasterData(
            gdf=dummy_aoi_gdf,
            index_analysis="ndvi",
            area_of_analysis_name="test_area",
            start_date="2020-01-01",
            end_date="2020-03-01",
            tmp_dir=temp_dir,
            band_name_dict=band_dict,
            index_equation=["(nir-red)/(nir+red)"],
        )

        # Spot‑check a few attributes that are set during __init__
        assert p.gdf.equals(dummy_aoi_gdf)
        assert p.start_date == "2020-01-01"
        assert p.end_date == "2020-03-01"
        assert p.band_name_dict == band_dict
        assert p.freq == "MS"
        assert p.satellite == "sentinel-2-l2a"

    def test_pcrasterdata_invalid_dates(self, dummy_aoi_gdf, temp_dir):
        """Start date after end date should raise a ValueError."""
        band_dict = {"red": [False, "red"], "nir": [False, "nir"]}

        with pytest.raises(ValueError, match="start_date must be before end_date"):
            raster.PCRasterData(
                gdf=dummy_aoi_gdf,
                index_analysis="ndvi",
                area_of_analysis_name="bad_dates",
                start_date="2021-01-01",
                end_date="2020-01-01",
                tmp_dir=temp_dir,
                band_name_dict=band_dict,
                index_equation=[],
            )

    def test_define_processing_directory_creates_files(self, dummy_aoi_gdf, temp_dir):
        """Helper should write a CSV inventory and create the expected dirs."""
        band_dict = {"red": [False, "red"], "nir": [False, "nir"]}

        p = raster.PCRasterData(
            gdf=dummy_aoi_gdf,
            index_analysis="ndvi",
            area_of_analysis_name="inventory_test",
            start_date="2020-01-01",
            end_date="2020-02-01",
            tmp_dir=temp_dir,
            band_name_dict=band_dict,
            index_equation=[],
        )

        # Tiny inventory DataFrame – content does not matter for the test
        inv = pd.DataFrame(
            {
                "year": [2020, 2020],
                "month": [1, 2],
                "data_id": [0, 0],
                "able_to_download": [np.nan, np.nan],
            }
        )

        csv_path = p._define_processing_directory(inv)

        # CSV must exist and contain the same rows
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(loaded, inv)

        # Expected directories
        assert (temp_dir / "inventory_test").exists()
        assert (temp_dir / "inventory_test" / "temporary_files").exists()

    def test_clean_mask_returns_array(self, dummy_raster):
        """Mask a simple square geometry against a tiny raster."""
        geom = box(-0.5, -0.5, 0.5, 0.5)  # overlaps centre pixel of the dummy raster

        with rasterio.open(dummy_raster) as src:
            masked = raster.clean_mask(geom, dataset=src, outside_value=-999)

        assert isinstance(masked, np.ndarray)
        # Centre pixel (value 4) should be present in the masked result
        assert 4 in masked

    def test_clean_mask_outside_geometry_returns_outside_value(self, dummy_raster):
        """When the geometry lies completely outside the raster, the fallback value is used."""
        geom = box(10, 10, 11, 11)  # far away from the dummy raster

        with rasterio.open(dummy_raster) as src:
            masked = raster.clean_mask(geom, dataset=src, outside_value=-999)

        assert masked.shape == (1,)
        assert masked[0] == -999

    def _write_dummy_inventory(self, tmp_dir, area_name, index_name):
        """Utility that writes a minimal inventory CSV for the given area/index."""
        df = pd.DataFrame(
            {
                "year": [2020],
                "month": [1],
                "data_id": [1],
                "able_to_download": [1],
            }
        )
        csv_path = tmp_dir / f"{area_name}_{index_name}_dataframe.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_raster_to_polygon_summary(self, dummy_aoi_gdf, dummy_raster, temp_dir):
        """
        Verify that ``RasterToPolygon.raster_summary`` produces the expected
        statistical columns.  The heavy raster‑reading step is mocked – we supply
        a tiny DataFrame that mimics the output of ``raster_to_polygon_multiprocess``.
        """
        area_name = "test"
        index_name = "temperature"

        hex_gdf = gpd.read_file("data/hex_test.gpkg")

        # Create the RasterToPolygon instance
        rtp = raster.RasterToPolygon(
            gdf=hex_gdf,
            feature_unique_id="hex_id_9",
            df_raster_inventory=pd.read_csv(
                f"data/{index_name}_{area_name}_dataframe.csv"
            ),
            index_analysis=index_name,
            area_of_analysis_name=area_name,
            tmp_dir="data/",
        )

        #  Run the summary routine (now operating on the mock DataFrame)
        gdf_summary, df_detail = rtp.raster_summary()

        # ------------------------------------------------------------------
        # Assertions – unchanged from the original test
        # ------------------------------------------------------------------
        # Types
        assert isinstance(gdf_summary, gpd.GeoDataFrame)
        assert isinstance(df_detail, pd.DataFrame)

        # Core statistical columns produced by the pipeline
        expected_stats = [
            f"{index_name}_max",
            f"{index_name}_min",
            f"{index_name}_mean",
            f"{index_name}_std",
            f"{index_name}_median",
            f"{index_name}_tend",
            f"{index_name}_diff",
            f"{index_name}_tend",
        ]
        for col in expected_stats:
            assert col in gdf_summary.columns

        # Year‑specific column (one column per year present in the inventory)
        assert f"{index_name}_2025" in gdf_summary.columns

        # No NaNs in the mean column for our single polygon
        assert not gdf_summary[f"{index_name}_mean"].isna().any()

    def test_nanvalues_exception_is_raised(self, dummy_aoi_gdf, temp_dir):
        """
        The private ``_raster_nan_test`` method should raise ``NanValues`` when
        NaNs are detected inside the raster after masking.

        We create a tiny raster that deliberately contains a NaN value, then
        invoke the method and confirm the exception propagates.
        """
        # ------------------------------------------------------------------
        # Create a raster with a NaN cell
        # ------------------------------------------------------------------
        raster_path = temp_dir / "nan_raster.tif"
        data = np.array([[1.0, 2.0], [np.nan, 4.0]], dtype="float32").reshape((1, 2, 2))
        transform = rasterio.transform.from_origin(0, 2, 1, 1)  # arbitrary
        meta = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": 2,
            "height": 2,
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(data)

        # ------------------------------------------------------------------
        # Minimal AOI that fully covers the raster
        # ------------------------------------------------------------------
        aoi = gpd.GeoDataFrame({"geometry": [box(-1, -1, 3, 3)]}, crs="EPSG:4326")

        # ------------------------------------------------------------------
        #  Initialise PCRasterData (most args are irrelevant for this test)
        # ------------------------------------------------------------------
        band_dict = {"red": [False, "red"], "nir": [False, "nir"]}

        p = raster.PCRasterData(
            gdf=aoi,
            index_analysis="ndvi",
            area_of_analysis_name="nan_test",
            start_date="2020-01-01",
            end_date="2020-03-01",
            tmp_dir=temp_dir,
            band_name_dict=band_dict,
            index_equation=[],
        )

        # ------------------------------------------------------------------
        #  Open the raster and trigger the null‑test
        # ------------------------------------------------------------------
        with rasterio.open(raster_path) as src:
            with pytest.raises(raster.NanValues, match="NaN values are still present"):
                p._raster_nan_test(aoi, src)
