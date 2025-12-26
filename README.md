# Thames Connectivity Analysis

A network analysis of London's transit system examining connectivity patterns, bottlenecks, resilience, and equity across the Thames divide.

## Quick Start

### Prerequisites
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/) (handles Python 3.11 automatically)

### Setup
```bash
# Clone and setup
git clone <repository-url>
cd ThamesConnectivityAnalysis
chmod +x setup.sh
./setup.sh
```

### If raw GeoPackages are provided as .zip (upload-size constraint)
If you received `data/raw/*.zip` files (e.g. `oprvrs_gb.gpkg.zip`), unpack them first:

```bash
bash scripts/unzip_raw_data.sh
```

### Running the Full Analysis Pipeline
```bash
# Phase 1: Data Ingestion
uv run python scripts/phases/ingest_data.py

# Phase 2: Network Construction
uv run python scripts/phases/transform_load_network_data.py

# Phase 3: Spatial Processing
uv run python scripts/phases/spatial_processor.py

# Phase 4: Analysis (via Jupyter notebooks)
uv run jupyter notebook notebooks/
```

## Project Structure

### Core Phases
1. **Data Ingestion** (`scripts/phases/ingest_data.py`)
   - TfL transit data (stations, routes, sequences)
   - Thames centerline geometry
   - LSOA boundaries and IMD deprivation indices

2. **Network Construction** (`scripts/phases/transform_load_network_data.py`)
   - Station-edge graph from TfL route sequences
   - London boundary filtering
   - Network validation and visualization

3. **Spatial Processing** (`scripts/phases/spatial_processor.py`)
   - Station-LSOA spatial joins
   - Thames bank classification
   - Cross-river connectivity detection

4. **Analysis Notebooks** 
   - `01_network_metrics.ipynb`: Basic network properties and centrality
   - `02_h1_bottleneck_analysis.ipynb`: Critical node identification
   - `03_h2_resilience_analysis.ipynb`: Failure cascade modeling
   - `04_h3_equity_analysis.ipynb`: Cross-Thames accessibility patterns

### Directory Structure
```
ThamesConnectivityAnalysis/
├── data/
│   ├── raw/                # Original raw datasets
│   ├── processed/          # Cleaned, validated datasets
│   │   ├── transit/       
│   │   ├── boundaries/    
│   │   ├── spatial/       
│   │   └── _meta/         
│   └── analysis/          
├── src/                    # Python source code modules
│   ├── core/               # Config, logging, base utilities
│   ├── data_processing/    # Data ingestion and cleaning logic
│   ├── geo/                # Spatial operations and helpers
│   ├── connectivity/       # Network analysis algorithms
│   ├── graph/              
│   └── vis/                
├── scripts/
│   └── phases/             
├── notebooks/             
├── artifacts/              
└── config/                 # Pipeline and environment configuration files
```


### Key Outputs
- **Network Data**: `data/processed/transit/` (stations, edges, London subset)
- **Spatial Mappings**: `data/processed/spatial/` (LSOA joins, bank classifications)
- **Visualistions**: `artifacts/` (network maps, analysis figures)

## Data Sources

### Required raw data (OS Open Rivers is manual)
- **TfL API (auto-ingested)**: stop points, route sequences, line information (downloaded by `scripts/phases/ingest_data.py`).
- **ONS Geography + IMD (auto-ingested)**: LSOA boundaries + IMD (downloaded by `scripts/phases/ingest_data.py`).
- **OS Open Rivers (manual)**: you must download the OS Open Rivers GeoPackage and place it at:
  - `data/raw/oprvrs_gb.gpkg`

Because of upload-size constraints, we also provide the raw GeoPackage in compressed form:
- `data/raw/oprvrs_gb.gpkg.zip`

To use the zipped version, run:

```bash
bash scripts/unzip_raw_data.sh
```

### Data Pipeline
All raw datasets are preprocessed through validated schemas ensuring:
- Consistent coordinate reference systems (WGS84 → BNG)
- Type safety with pandas nullable dtypes
- Spatial topology validation
- Comprehensive QA reporting

## Research Hypotheses

### H1: Bottleneck Identification
Critical stations whose removal disproportionately impacts network connectivity, particularly at Thames crossing points.

### H2: Resilience Analysis
Network vulnerability to cascading failures, with focus on cross-Thames connectivity preservation.

### H3: Equity Assessment
Systematic accessibility disparities between north and south London, correlated with deprivation indices.

## Reproducibility Notes

### Dependencies
- All dependencies pinned in `pyproject.toml` with lockfile `uv.lock`
- Tested on Python 3.11 with UV package manager
- Geographic operations use EPSG:27700 (British National Grid)