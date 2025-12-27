"""Shared visualisation utilities for London map plotting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Constants from existing scripts
CRS_BNG = "EPSG:27700"


@dataclass(frozen=True)
class MapStyle:
    """Consistent styling configuration for London map plot"""

    # Figure configuration
    figsize: tuple[float, float] = (10.0, 10.0)
    figsize_dual: tuple[float, float] = (14.0, 6.5)
    facecolor: str = "white"

    # London boundary styling
    london_color: str = "#111827"
    london_linewidth: float = 1.0
    london_alpha: float = 0.45

    # Thames river styling
    river_color: str = "#374151"
    river_linewidth: float = 2.0
    river_alpha: float = 0.55

    # Bank-specific colors
    station_colors: dict[str, str] = field(
        default_factory=lambda: {"north": "#2563eb", "south": "#16a34a", "unknown": "#6b7280"}
    )

    # Crossing edge highlighting
    crossing_color: str = "#dc2626"

    # Layout padding
    extent_padding: float = 0.08


@dataclass(frozen=True)
class PlotContext:
    """Spatial context data for London map plotting."""

    london: gpd.GeoDataFrame
    river: gpd.GeoDataFrame
    extent: tuple[float, float, float, float, float, float]  # minx, miny, maxx, maxy, pad_x, pad_y

    @classmethod
    def from_paths(
        cls, boundary_gpkg: Path, river_geojson: Path, style: MapStyle = None
    ) -> PlotContext:
        """Load spatial context from file paths."""
        style = style or MapStyle()

        LOGGER.debug("Loading London boundary from %s", boundary_gpkg)
        london = gpd.read_file(boundary_gpkg).to_crs(CRS_BNG)

        LOGGER.debug("Loading Thames river from %s", river_geojson)
        river = gpd.read_file(river_geojson).to_crs(CRS_BNG)

        # Clip river to London boundary for cleaner visualisation
        try:
            river = gpd.clip(river, london.geometry.union_all())
            LOGGER.debug("Successfully clipped river to London boundary")
        except Exception as e:
            LOGGER.warning("River clipping failed, using full geometry: %s", e)

        # Calculate extent with padding
        minx, miny, maxx, maxy = london.total_bounds
        pad_x = (maxx - minx) * style.extent_padding
        pad_y = (maxy - miny) * style.extent_padding
        extent = (minx, miny, maxx, maxy, pad_x, pad_y)

        return cls(london=london, river=river, extent=extent)


class LondonMapPlotter:
    """Reusable London map plotting with Thames context."""

    def __init__(self, context: PlotContext, style: MapStyle = None):
        self.context = context
        self.style = style or MapStyle()

    @classmethod
    def from_paths(
        cls, boundary_gpkg: Path, river_geojson: Path, style: MapStyle = None
    ) -> LondonMapPlotter:
        context = PlotContext.from_paths(boundary_gpkg, river_geojson, style)
        return cls(context, style)

    def setup_london_axes(self, figsize: tuple[float, float] = None) -> tuple[plt.Figure, plt.Axes]:
        """Setup London map axes (with no visible x/y axes)."""
        figsize = figsize or self.style.figsize

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(self.style.facecolor)

        # Plot boundary and river background
        self.context.london.boundary.plot(
            ax=ax,
            color=self.style.london_color,
            linewidth=self.style.london_linewidth,
            alpha=self.style.london_alpha,
            zorder=1,
        )

        self.context.river.plot(
            ax=ax,
            color=self.style.river_color,
            linewidth=self.style.river_linewidth,
            alpha=self.style.river_alpha,
            zorder=2,
        )

        # Set extent and remove axis formatting
        minx, miny, maxx, maxy, pad_x, pad_y = self.context.extent
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        ax.set_aspect("equal", adjustable="box")

        # Ensure the map has no x and y axis: remove ticks, labels, and axis border
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        return fig, ax


def prepare_station_geometries(stations: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert stations DataFrame to BNG-projected GeoDataFrame."""
    required_cols = {"station_id", "lon", "lat"}
    if not required_cols.issubset(stations.columns):
        missing = required_cols - set(stations.columns)
        raise ValueError(f"Missing required columns: {missing}")

    stations_gdf = gpd.GeoDataFrame(
        stations, geometry=gpd.points_from_xy(stations["lon"], stations["lat"]), crs="EPSG:4326"
    ).to_crs(CRS_BNG)

    stations_gdf["x"] = stations_gdf.geometry.x
    stations_gdf["y"] = stations_gdf.geometry.y

    return stations_gdf


def create_edge_segments(
    stations_gdf: gpd.GeoDataFrame, edges: pd.DataFrame
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Create LineCollection segments from edge DataFrame."""
    if not {"u", "v"}.issubset(edges.columns):
        raise ValueError("edges DataFrame must have 'u' and 'v' columns")

    if not {"x", "y", "station_id"}.issubset(stations_gdf.columns):
        raise ValueError("stations_gdf must have 'x', 'y', and 'station_id' columns")

    coord_lookup = stations_gdf.set_index("station_id")[["x", "y"]].to_dict("index")

    segments = []
    for _, edge in edges.iterrows():
        u, v = str(edge["u"]), str(edge["v"])

        if u not in coord_lookup or v not in coord_lookup:
            LOGGER.warning("Skipping edge (%s, %s): station not found in geometries", u, v)
            continue

        u_coords = coord_lookup[u]
        v_coords = coord_lookup[v]
        segments.append(((u_coords["x"], u_coords["y"]), (v_coords["x"], v_coords["y"])))

    return segments
