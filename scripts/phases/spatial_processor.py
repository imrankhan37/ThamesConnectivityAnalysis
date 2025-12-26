"""
Run from repo root:
  uv run python scripts/phases/spatial_processor.py

Processing steps:
1. Export London LSOA subset to GeoJSON
2. Spatial join stations to LSOA polygons
3. Bank classification and Thames crossing detection

Outputs:
- data/processed/boundaries/lsoa_london.geojson
- data/processed/spatial/station_lsoa.csv
- data/processed/spatial/station_bank.csv
- data/processed/spatial/edge_is_thames_crossing.csv
- data/processed/spatial/crossing_count.csv
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

# Ensure repo root is on sys.path so `import src...` works when executing this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import CRS_BNG, CRS_WGS84, configure_logging, get_paths
from src.geo.spatial import (
    classify_station_bank_by_local_orientation,
    label_thames_crossing_edges,
    spatial_join_stations_to_lsoa,
    stations_to_gdf,
)
from src.io import ensure_unzipped, read_csv_validated
from src.models.schemas import (
    EDGE_CROSSING,
    EDGES,
    STATION_BANK,
    STATION_LSOA,
    STATIONS,
)
from src.models.validate import validate_df

LOGGER = logging.getLogger("spatial_processor")


@dataclass(frozen=True)
class SpatialConfig:
    """Configuration for spatial processing."""

    # File paths
    stations_csv: Path
    edges_csv: Path
    lsoa_gpkg: Path
    river_geojson: Path
    boundary_gpkg: Path

    # Processing parameters
    lsoa_simplify_tolerance_m: float = 25.0
    station_lsoa_nearest_max_m: float = 750.0
    crossing_buffer_m: float = 75.0
    figure_size: tuple[float, float] = (9.5, 9.5)
    plot_dpi: int = 300

    @classmethod
    def from_paths(cls, paths) -> SpatialConfig:
        """Create config with default London paths."""
        return cls(
            stations_csv=paths.processed_transit / "stations_london.csv",
            edges_csv=paths.processed_transit / "edges_london.csv",
            lsoa_gpkg=paths.data_raw / "lsoa_2021_ew_bfe_v10_london_bbox.gpkg",
            river_geojson=paths.data_raw / "thames_centerline.geojson",
            boundary_gpkg=paths.data_raw / "ons_regions_2021_en_bgc.gpkg",
        )


class SpatialProcessor:
    """Fluent interface for spatial processing operations."""

    def __init__(self, config: SpatialConfig):
        self.config = config
        self.paths = None
        self.datasets = None

    def setup(self):
        """Initialize logging and paths."""
        configure_logging()
        self.paths = get_paths()
        self.paths.docs_qa.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting spatial processing with:")
        LOGGER.info("  Stations: %s", self.config.stations_csv)
        LOGGER.info("  Edges: %s", self.config.edges_csv)
        return self

    def export_lsoa(self):
        """Export London LSOA subset to GeoJSON."""
        LOGGER.info("Step: LSOA GeoJSON export")

        out_path = self.paths.processed_boundaries / "lsoa_london.geojson"

        lsoa_gpkg = ensure_unzipped(self.config.lsoa_gpkg)
        gdf = gpd.read_file(lsoa_gpkg, layer="lsoa2021").to_crs(CRS_BNG)
        LOGGER.info("Loaded LSOA polygons: %d", len(gdf))

        # Topology-preserving simplification in meters
        gdf["geometry"] = gdf["geometry"].simplify(
            self.config.lsoa_simplify_tolerance_m, preserve_topology=True
        )
        gdf = gdf.to_crs(CRS_WGS84)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON")
        LOGGER.info(
            "Wrote %s (simplify_tol_m=%.1f)", out_path, self.config.lsoa_simplify_tolerance_m
        )
        return self

    def load_datasets(self):
        """Load all London datasets needed for spatial processing."""
        LOGGER.info("Loading London datasets...")

        # Support zipped raw artifacts for upload-size constraints.
        lsoa_gpkg = ensure_unzipped(self.config.lsoa_gpkg)
        river_geojson = ensure_unzipped(self.config.river_geojson)
        boundary_gpkg = ensure_unzipped(self.config.boundary_gpkg)

        # Load and validate stations/edges
        stations = read_csv_validated(
            self.config.stations_csv, dtype={"station_id": "string"}, schema=STATIONS
        )
        edges = read_csv_validated(
            self.config.edges_csv, dtype={"u": "string", "v": "string"}, schema=EDGES
        )

        # Convert stations to GeoDataFrame
        stations_gdf = stations_to_gdf(stations)

        # Load LSOA data
        lsoa_gdf = gpd.read_file(lsoa_gpkg, layer="lsoa2021")

        # Load Thames centerline
        river_gdf = gpd.read_file(river_geojson)

        LOGGER.info(
            "Loaded: %d stations, %d edges, %d LSOA polygons",
            len(stations),
            len(edges),
            len(lsoa_gdf),
        )

        self.datasets = {
            "stations": stations,
            "edges": edges,
            "stations_gdf": stations_gdf,
            "lsoa_gdf": lsoa_gdf,
            "river_gdf": river_gdf,
            "boundary_gpkg": boundary_gpkg,
        }
        return self

    def process_station_lsoa(self):
        """Spatial join stations to LSOA polygons with QA report."""
        LOGGER.info("Step: Station to LSOA spatial join")

        station_lsoa, qa = spatial_join_stations_to_lsoa(
            stations_gdf_wgs84=self.datasets["stations_gdf"],
            lsoa_gdf_wgs84=self.datasets["lsoa_gdf"],
            lsoa_code_col="LSOA21CD",
            nearest_max_m=self.config.station_lsoa_nearest_max_m,
        )

        station_lsoa = validate_df(station_lsoa, STATION_LSOA)
        out_csv = self.paths.processed_spatial / "station_lsoa.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        station_lsoa.to_csv(out_csv, index=False)
        LOGGER.info("Wrote %s", out_csv)

        # QA report
        self._write_station_lsoa_qa_report(qa)
        return self

    def process_bank_crossings(self):
        """Bank classification, Thames crossing detection, and visualisation."""
        LOGGER.info("Step: Bank classification and Thames crossings")

        self.paths.processed_spatial.mkdir(parents=True, exist_ok=True)

        # Bank classification
        station_bank = self._classify_banks()

        # Thames crossing detection
        edge_cross = self._detect_crossings(station_bank)

        # Generate outputs
        self._write_crossing_table(edge_cross)
        self._create_visualization(station_bank, edge_cross)

        return self

    def complete(self):
        """Finalise processing."""
        LOGGER.info("Spatial processing completed successfully")
        return self

    def _classify_banks(self):
        """Classify station bank labels (deterministic; no manual overrides)."""
        station_bank = classify_station_bank_by_local_orientation(
            stations_wgs84=self.datasets["stations_gdf"],
            river_centerline_wgs84=self.datasets["river_gdf"],
            bank_col="bank",
        )
        station_bank = validate_df(station_bank, STATION_BANK)

        out_bank = self.paths.processed_spatial / "station_bank.csv"
        station_bank.to_csv(out_bank, index=False)
        LOGGER.info("Wrote %s", out_bank)

        return station_bank

    def _detect_crossings(self, station_bank):
        """Detect Thames crossing edges."""
        stations_bng = self.datasets["stations_gdf"].to_crs(CRS_BNG)
        edge_cross = label_thames_crossing_edges(
            edges=self.datasets["edges"],
            stations_bng=stations_bng[["station_id", "geometry"]],
            station_bank=station_bank[["station_id", "bank"]],
            river_centerline_wgs84=self.datasets["river_gdf"],
            buffer_m=self.config.crossing_buffer_m,
        )
        edge_cross = validate_df(edge_cross, EDGE_CROSSING)

        out_edges = self.paths.processed_spatial / "edge_is_thames_crossing.csv"
        edge_cross.to_csv(out_edges, index=False)
        LOGGER.info("Wrote %s", out_edges)

        return edge_cross

    def _write_crossing_table(self, edge_cross):
        """Generate crossing count table."""
        n_cross = int(edge_cross["is_thames_crossing"].sum()) if not edge_cross.empty else 0
        tbl = pd.DataFrame(
            [
                {"metric": "edges_total", "value": int(len(self.datasets["edges"]))},
                {"metric": "edges_crossing", "value": n_cross},
                {"metric": "buffer_m", "value": float(self.config.crossing_buffer_m)},
            ]
        )

        out_tbl = self.paths.processed_spatial / "crossing_count.csv"
        out_tbl.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(out_tbl, index=False)
        LOGGER.info("Wrote %s", out_tbl)

    def _create_visualization(self, station_bank, edge_cross):
        """Create banks and crossings visualisation."""
        fig_out = self.paths.figures / "fig02_banks_and_crossings.png"
        self._plot_banks_and_crossings(station_bank, edge_cross, fig_out)
        LOGGER.info("Wrote %s", fig_out)

    def _plot_banks_and_crossings(
        self, station_bank: pd.DataFrame, edge_cross: pd.DataFrame, out_path: Path
    ) -> None:
        """Create Figure 2: Banks and Thames-crossing edges visualisation."""
        crs_plot = CRS_BNG

        # Load and project geospatial data
        london = gpd.read_file(self.datasets["boundary_gpkg"]).to_crs(crs_plot)
        river = self.datasets["river_gdf"].to_crs(crs_plot)
        stations_bng = self.datasets["stations_gdf"].to_crs(crs_plot)

        st = stations_bng.set_index("station_id")
        bank_map = station_bank.set_index("station_id")["bank"].to_dict()

        # Build edge segments
        ok = self.datasets["edges"]["u"].isin(st.index) & self.datasets["edges"]["v"].isin(st.index)
        e = self.datasets["edges"].loc[ok, ["u", "v"]].copy()
        x0 = st.loc[e["u"], "geometry"].x.to_numpy()
        y0 = st.loc[e["u"], "geometry"].y.to_numpy()
        x1 = st.loc[e["v"], "geometry"].x.to_numpy()
        y1 = st.loc[e["v"], "geometry"].y.to_numpy()
        segs = list(
            zip(
                zip(x0, y0, strict=True),
                zip(x1, y1, strict=True),
                strict=True,
            )
        )

        # Identify crossing segments
        cross_set = {
            (str(u), str(v))
            for u, v in edge_cross.loc[
                edge_cross["is_thames_crossing"].fillna(False).astype(bool), ["u", "v"]
            ].itertuples(index=False, name=None)
        }
        cross_mask = [
            ((u, v) in cross_set) or ((v, u) in cross_set)
            for u, v in e[["u", "v"]].astype(str).values
        ]
        segs_cross = [s for s, is_c in zip(segs, cross_mask, strict=True) if is_c]
        segs_other = [s for s, is_c in zip(segs, cross_mask, strict=True) if not is_c]

        # Create plot
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        ax.set_facecolor("white")

        # Geographic layers
        london.boundary.plot(ax=ax, color="#111827", linewidth=1.0, alpha=0.55, zorder=1)
        river.plot(ax=ax, color="#374151", linewidth=2.0, alpha=0.65, zorder=2)

        # Network edges: non-crossing light grey; crossing highlighted red
        ax.add_collection(
            LineCollection(segs_other, colors=(0, 0, 0, 0.08), linewidths=0.8, zorder=3)
        )
        ax.add_collection(
            LineCollection(
                segs_cross, colors=(220 / 255, 38 / 255, 38 / 255, 0.75), linewidths=1.8, zorder=4
            )
        )

        # Stations coloured by bank
        banks = pd.Series([bank_map.get(sid, "") for sid in st.index], index=st.index)
        north = banks == "north"
        south = banks == "south"
        ax.scatter(
            st.loc[north, "geometry"].x,
            st.loc[north, "geometry"].y,
            s=14,
            color="#2563eb",
            alpha=0.9,
            linewidths=0,
            zorder=5,
            label="North bank",
        )
        ax.scatter(
            st.loc[south, "geometry"].x,
            st.loc[south, "geometry"].y,
            s=14,
            color="#16a34a",
            alpha=0.9,
            linewidths=0,
            zorder=5,
            label="South bank",
        )

        # Styling
        ax.set_title("Banks and detected Thames-crossing edges (projected: EPSG:27700)")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower center", ncol=3, frameon=False)

        # Set bounds based on London boundary
        minx, miny, maxx, maxy = london.total_bounds
        pad_x = (maxx - minx) * 0.08
        pad_y = (maxy - miny) * 0.08
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        # Save figure
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=self.config.plot_dpi)
        plt.close(fig)


def main() -> None:
    paths = get_paths()
    config = SpatialConfig.from_paths(paths)

    (
        SpatialProcessor(config)
        .setup()
        .export_lsoa()
        .load_datasets()
        .process_station_lsoa()
        .process_bank_crossings()
        .complete()
    )


if __name__ == "__main__":
    main()
